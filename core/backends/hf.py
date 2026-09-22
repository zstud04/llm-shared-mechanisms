"""Hugging Face `transformers` backend: inference + interpretability.

Interpretability on plain `transformers` works by mapping each `Component` to
a concrete submodule plus a site ("input" or "output" of that module). The
mapping is chosen so the tensors match TransformerLens' hook semantics, which
keeps results comparable between the two backends:

    attn_head   input of o_proj, reshaped to [batch, seq, n_heads, d_head]
                (== TL `hook_z`)
    attn_out    output of o_proj                     (== TL `hook_attn_out`)
    mlp_pre     output of gate_proj / c_fc           (== TL `hook_pre`)
    mlp_post    input of down_proj / c_proj          (== TL `hook_post`)
    mlp_out     output of the MLP block              (== TL `hook_mlp_out`)
    resid_post  output of the decoder layer          (== TL `hook_resid_post`)

Those submodules are located by *searching* the loaded module tree rather than
by a fixed attribute path, so multimodal wrappers (Gemma-3, Llava, Qwen-VL,
...) that bury the text stack under `model.language_model` work without any
per-architecture entry.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

import core.chat_templates as chat_templates
from core.backends.base import (
    GenerationResult,
    InterpBackend,
    ScoredTokens,
    resolve_cache_dir,
)
from core.components import Component

# Attribute paths tried first when locating the decoder-layer list. If none
# match we fall back to searching the module tree (`_discover_layers`).
_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "language_model.layers",
    "model.language_model.layers",
)

# Submodule names to skip when searching for the *text* stack: multimodal
# checkpoints carry a full vision transformer with its own decoder layers.
_NON_TEXT_HINTS = ("vision", "visual", "image", "audio", "speech", "multi_modal", "projector")

# Per-role name candidates, most specific first. A role is resolved by walking
# a decoder layer and matching child module names against these.
_ROLE_NAMES: Dict[str, Tuple[str, ...]] = {
    "attn": ("self_attn", "attn", "attention", "self_attention"),
    "attn_o": ("o_proj", "out_proj", "c_proj", "dense", "wo", "proj"),
    "mlp": ("mlp", "feed_forward", "ffn", "feedforward", "mlp_block"),
    # Gated MLPs: the pre-activation TL exposes is the *gate* branch.
    "mlp_in": ("gate_proj", "c_fc", "dense_h_to_4h", "w1", "fc_in", "fc1", "up_proj"),
    "mlp_down": ("down_proj", "c_proj", "dense_4h_to_h", "w2", "fc_out", "fc2"),
}

# Per-kind (role path, site). "input" hooks the module's first positional
# argument; "output" hooks its return value. An empty role path means the
# decoder layer itself.
_SITES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "attn_head": (("attn", "attn_o"), "input"),
    "attn_out": (("attn", "attn_o"), "output"),
    "mlp_pre": (("mlp", "mlp_in"), "output"),
    "mlp_post": (("mlp", "mlp_down"), "input"),
    "mlp_out": (("mlp",), "output"),
    "resid_post": ((), "output"),
}

# Hook kinds that read the dense feed-forward branch, and so cannot describe a
# sparse layer's full FFN. See `_is_moe_layer`.
_DENSE_MLP_KINDS = ("mlp_pre", "mlp_post", "mlp_out")


def _getattr_path(obj, path: str):
    for part in path.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def _is_text_name(name: str) -> bool:
    lowered = name.lower()
    return not any(hint in lowered for hint in _NON_TEXT_HINTS)


def _resolve_role(parent: torch.nn.Module, role: str) -> Optional[torch.nn.Module]:
    """Find the child of `parent` playing `role`, by name.

    Exact name matches win; otherwise a child whose name contains a candidate
    (e.g. `mlp_dense_4h_to_h`) is accepted, so unusual namings still resolve.
    """
    children = dict(parent.named_children())
    for candidate in _ROLE_NAMES[role]:
        if candidate in children:
            return children[candidate]
    for candidate in _ROLE_NAMES[role]:
        for name, child in children.items():
            if candidate in name.lower():
                return child
    return None


def _is_moe_layer(layer: torch.nn.Module) -> bool:
    """True when this decoder layer routes tokens through sparse experts.

    Gemma-4's MoE layers (`gemma4`, e.g. 26B-A4B) keep a *dense* `mlp` and the
    routed `experts` as siblings, and add the two branches together:

        h = post_ffn_ln_1(mlp(h)) + post_ffn_ln_2(experts(router(h)))

    So `mlp` still resolves on these layers, but reading it captures only the
    dense branch and silently drops the 128 routed experts. Detect the sparse
    layers so `_resolve_site` can refuse rather than return a partial number.
    `enable_moe_block` is per-layer: a model may mix dense and sparse layers.
    """
    children = dict(layer.named_children())
    if "experts" not in children:
        return False
    return bool(getattr(layer, "enable_moe_block", True))


def _find_text_config(config):
    """Return the sub-config describing the text transformer.

    Multimodal configs (Gemma-3, Llava, ...) keep the language-model
    hyperparameters under `text_config` / `llm_config` / `decoder`.
    """
    for attr in ("text_config", "llm_config", "language_config", "decoder"):
        sub = getattr(config, attr, None)
        if sub is not None and getattr(sub, "num_hidden_layers", None) is not None:
            return sub
    return config


def _discover_layers(model: torch.nn.Module, n_expected: Optional[int]):
    """Search the module tree for the decoder-layer `ModuleList`.

    Candidates are `ModuleList`s of transformer blocks (a child with an
    attention role) that do not sit under a vision/audio tower. When the config
    reports a layer count we require a match; otherwise the longest list wins.
    """
    best = None
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if not _is_text_name(name):
            continue
        block = module[0]
        if _resolve_role(block, "attn") is None and _resolve_role(block, "mlp") is None:
            continue
        if n_expected is not None and len(module) != n_expected:
            continue
        if best is None or len(module) > len(best[1]):
            best = (name, module)
    return best


def _first_tensor(value):
    """Modules may return a tensor or a tuple whose first element is one."""
    return value[0] if isinstance(value, (tuple, list)) else value


def _replace_first_tensor(value, new):
    if isinstance(value, tuple):
        return (new,) + tuple(value[1:])
    if isinstance(value, list):
        return [new] + list(value[1:])
    return new


class HFBackend(InterpBackend):
    """Wraps `AutoModelForCausalLM` (or `AutoModelForMaskedLM` for BERT).

    Models above `shard_threshold_b` parameters are loaded with
    `device_map="auto"` so `accelerate` shards them across every visible GPU;
    that is what makes >30B runs possible on CHTC multi-GPU slots.
    """

    def __init__(
        self,
        name: str,
        cfg: Dict[str, Any],
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        cache_dir: Optional[str] = None,
        device_map: Optional[str] = "auto",
        max_memory: Optional[Dict[Any, str]] = None,
        enable_templates: bool = True,
    ):
        super().__init__(name, cfg)
        self.device = device
        self.enable_templates = enable_templates
        self.prompt_template = chat_templates.template_for(cfg)
        self.is_masked = bool(cfg.get("masked", "bert" in cfg["path"].lower()))

        model_cls = AutoModelForMaskedLM if self.is_masked else AutoModelForCausalLM
        resolved_cache = resolve_cache_dir(cache_dir)

        # Default to trusting the repo's custom modeling code, but let a model
        # opt out. Some repos (e.g. Phi-3) ship a modeling_*.py pinned to an old
        # cache API that breaks on current transformers; those load correctly
        # with the native class when trust_remote_code is false.
        trust_remote = bool(cfg.get("trust_remote_code", True))

        # Some repos (e.g. Phi-3.5-vision) default their custom code to
        # flash_attention_2, which isn't in the container; let a model force a
        # different kernel ("eager"/"sdpa"). None lets transformers choose.
        attn_impl = cfg.get("attn_implementation")

        self.model = model_cls.from_pretrained(
            cfg["path"],
            cache_dir=resolved_cache,
            dtype=dtype,  # transformers>=5 renamed torch_dtype to dtype
            trust_remote_code=trust_remote,
            device_map=device_map,  # "auto" shards across every visible GPU
            max_memory=max_memory,
            attn_implementation=attn_impl,
        )
        if device_map is None:
            self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"], cache_dir=resolved_cache, trust_remote_code=trust_remote
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Text hyperparameters live in a sub-config on multimodal checkpoints.
        self.text_config = _find_text_config(self.model.config)
        self._layers = self._resolve_layers()

    # ---- module resolution ----

    def _resolve_layers(self) -> List[torch.nn.Module]:
        n_expected = getattr(self.text_config, "num_hidden_layers", None)
        for path in _LAYER_PATHS:
            try:
                layers = _getattr_path(self.model, path)
            except AttributeError:
                continue
            if isinstance(layers, torch.nn.ModuleList) and (
                n_expected is None or len(layers) == n_expected
            ):
                return list(layers)

        found = _discover_layers(self.model, n_expected)
        if found is None:
            raise RuntimeError(
                f"Could not locate decoder layers on {self.cfg['path']}; tried "
                f"{_LAYER_PATHS} and a tree search for a ModuleList of "
                f"{n_expected} transformer blocks."
            )
        return list(found[1])

    def _resolve_site(self, component: Component) -> Tuple[torch.nn.Module, str]:
        """Return (module, site) for a component in its layer."""
        roles, site = _SITES[component.kind]
        module: torch.nn.Module = self._layers[component.layer]
        if component.kind in _DENSE_MLP_KINDS and _is_moe_layer(module):
            raise RuntimeError(
                f"{self.cfg['path']} layer {component.layer} is a "
                f"mixture-of-experts layer, so '{component.kind}' would read "
                "only the dense shared-expert branch and ignore the routed "
                "experts. Use attn_head/attn_out/resid_post on this model."
            )
        for role in roles:
            child = _resolve_role(module, role)
            if child is None:
                raise RuntimeError(
                    f"No '{role}' submodule for {component.kind} on "
                    f"{self.cfg['path']}; {type(module).__name__} has children "
                    f"{[n for n, _ in module.named_children()]}. Add a name to "
                    f"_ROLE_NAMES[{role!r}]."
                )
            module = child
        return module, site

    # ---- shape ----

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def n_heads(self) -> int:
        return int(self.text_config.num_attention_heads)

    @property
    def d_head(self) -> int:
        head_dim = getattr(self.text_config, "head_dim", None)
        if head_dim:
            return int(head_dim)
        return int(self.text_config.hidden_size) // self.n_heads

    def _to_heads(self, act: torch.Tensor) -> torch.Tensor:
        """[batch, seq, n_heads*d_head] -> [batch, seq, n_heads, d_head].

        d_head is read from the tensor rather than ``self.d_head`` because some
        models (e.g. Gemma 4) use a larger head dim on global-attention layers
        (``global_head_dim``) than on sliding layers (``head_dim``); the number
        of query heads stays constant across layers.
        """
        b, s, hidden = act.shape
        d_head, rem = divmod(hidden, self.n_heads)
        if rem:
            raise RuntimeError(
                f"o_proj input width {hidden} is not divisible by n_heads="
                f"{self.n_heads} on {self.cfg['path']}."
            )
        return act.view(b, s, self.n_heads, d_head)

    def _from_heads(self, act: torch.Tensor) -> torch.Tensor:
        b, s, h, d = act.shape
        return act.reshape(b, s, h * d)

    # ---- prompts / tokens ----

    def _format(self, instruction: str) -> str:
        if self.prompt_template and self.enable_templates:
            return self.prompt_template.format(instruction=instruction)
        return instruction

    def _encode(self, prompt: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in enc.items()}

    def to_tokens(self, prompt: str) -> torch.Tensor:
        return self._encode(self._format(prompt))["input_ids"]

    def token_id(self, s: str) -> int:
        return self.tokenizer.encode(s, add_special_tokens=False)[0]

    # ---- forward passes ----

    def logits(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            return self.model(**self._encode(self._format(prompt))).logits

    def generate(self, prompt: str, max_new_tokens: int = 10) -> GenerationResult:
        enc = self._encode(self._format(prompt))
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.model.config.pad_token_id,
            )
        text = self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        return GenerationResult(text=text, provider="hf", model=self.name)

    def score_options(self, prompt: str, option_a: str, option_b: str) -> ScoredTokens:
        logits = self.logits(prompt)
        a_id, b_id = self.token_id(option_a), self.token_id(option_b)
        last = logits[0, -1].float()
        return ScoredTokens(
            logit_a=last[a_id].item(),
            logit_b=last[b_id].item(),
            logit_diff=(last[a_id] - last[b_id]).item(),
            entropy=Categorical(logits=last).entropy().item(),
        )

    # ---- hook plumbing ----

    @contextmanager
    def _hooks(self, specs: List[Tuple[Component, Callable[[torch.Tensor], torch.Tensor]]]):
        """Install per-component transforms on the right module and site.

        Each `fn` receives the component's activation (head-reshaped for
        `attn_head`) and returns a replacement. Multiple components sharing a
        module are composed into one hook so the tensor is only touched once.
        """
        grouped: Dict[Tuple[int, str], List] = {}
        for comp, fn in specs:
            module, site = self._resolve_site(comp)
            grouped.setdefault((id(module), site), []).append((module, comp, fn))

        handles = []
        for (_, site), entries in grouped.items():
            module = entries[0][0]

            def apply(tensor, entries=entries):
                for _, comp, fn in entries:
                    if comp.kind == "attn_head":
                        heads = self._to_heads(tensor)
                        heads = fn(heads, comp)
                        tensor = self._from_heads(heads)
                    else:
                        tensor = fn(tensor, comp)
                return tensor

            if site == "input":
                def pre_hook(mod, args, apply=apply):
                    return (apply(args[0]),) + tuple(args[1:])

                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def fwd_hook(mod, args, output, apply=apply):
                    return _replace_first_tensor(output, apply(_first_tensor(output)))

                handles.append(module.register_forward_hook(fwd_hook))

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def cache(
        self, prompt: str, components: Sequence[Component]
    ) -> Dict[Component, torch.Tensor]:
        store: Dict[Component, torch.Tensor] = {}

        def make_saver(target):
            def saver(tensor, comp):
                if comp.index is not None:
                    store[target] = tensor[0, :, comp.index, :].detach().clone()
                else:
                    store[target] = tensor[0].detach().clone()
                return tensor

            return saver

        specs = [(c, make_saver(c)) for c in components]
        with self._hooks(specs), torch.no_grad():
            self.model(**self._encode(self._format(prompt)))
        return store

    def cache_layers(
        self,
        prompt: str,
        kinds: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        layer_list = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        targets = [
            Component(kind, layer, 0 if kind == "attn_head" else None)
            for kind in kinds
            for layer in layer_list
        ]
        store: Dict[Tuple[str, int], torch.Tensor] = {}

        def make_saver(key):
            def saver(tensor, comp):
                store[key] = tensor.detach().clone()
                return tensor

            return saver

        specs = [(t, make_saver((t.kind, t.layer))) for t in targets]
        with self._hooks(specs), torch.no_grad():
            self.model(**self._encode(self._format(prompt)))
        return store

    def attention_patterns(
        self, prompt: str, layers: Optional[Iterable[int]] = None
    ) -> Dict[int, torch.Tensor]:
        # SDPA/flash kernels never materialize the probability matrix, so we
        # swap to the eager implementation for the duration of this pass.
        # Multimodal configs carry their own copy on each sub-config.
        configs = {id(c): c for c in (self.model.config, self.text_config)}.values()
        previous = {id(c): getattr(c, "_attn_implementation", None) for c in configs}
        for c in configs:
            c._attn_implementation = "eager"
        try:
            with torch.no_grad():
                out = self.model(
                    **self._encode(self._format(prompt)), output_attentions=True
                )
        finally:
            for c in configs:
                if previous[id(c)] is not None:
                    c._attn_implementation = previous[id(c)]

        if not out.attentions or out.attentions[0] is None:
            raise RuntimeError(
                f"{self.cfg['path']} returned no attention probabilities; "
                "reload it with attn_implementation='eager'."
            )
        layer_list = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        return {l: out.attentions[l][0].detach().float().cpu() for l in layer_list}

    def run_with_ablation(
        self,
        prompt: str,
        components: Sequence[Component],
        mode: str = "zero",
        replacements: Optional[Dict[Component, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if mode != "zero" and not replacements:
            raise ValueError(f"mode={mode!r} requires `replacements`")
        values = {c: (None if mode == "zero" else replacements[c]) for c in components}
        return self._run_with_writes(prompt, values)

    def run_with_patch(
        self, prompt: str, patches: Dict[Component, torch.Tensor]
    ) -> torch.Tensor:
        return self._run_with_writes(prompt, dict(patches))

    def _run_with_writes(
        self, prompt: str, values: Dict[Component, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        def make_writer(value):
            def writer(tensor, comp):
                tensor = tensor.clone()
                new = 0 if value is None else value.to(tensor.dtype).to(tensor.device)
                if comp.index is not None:
                    tensor[:, :, comp.index, :] = new
                else:
                    tensor[:, :, :] = new
                return tensor

            return writer

        specs = [(c, make_writer(v)) for c, v in values.items()]
        with self._hooks(specs), torch.no_grad():
            return self.model(**self._encode(self._format(prompt))).logits

    def act_and_grad(
        self,
        prompt: str,
        a_id: int,
        b_id: int,
        kinds: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ):
        layer_list = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        targets = [
            Component(kind, layer, 0 if kind == "attn_head" else None)
            for kind in kinds
            for layer in layer_list
        ]

        stored: Dict[Tuple[str, int], torch.Tensor] = {}

        def make_saver(key, is_head):
            def saver(tensor, comp):
                # Keep the graph reference so autograd.grad can reach it.
                stored[key] = tensor
                return tensor

            return saver

        specs = [
            (t, make_saver((t.kind, t.layer), t.kind == "attn_head")) for t in targets
        ]

        # Deliberately outside no_grad: the backward pass needs the graph.
        with self._hooks(specs):
            logits = self.model(**self._encode(self._format(prompt))).logits
        metric = logits[0, -1, a_id] - logits[0, -1, b_id]

        keys = list(stored)
        act_list = [stored[k] for k in keys]
        grad_list = torch.autograd.grad(metric, act_list)

        acts = {k: a.detach() for k, a in zip(keys, act_list)}
        grads = {k: g.detach() for k, g in zip(keys, grad_list)}
        baseline = metric.item()

        del stored, logits, metric, act_list, grad_list
        return acts, grads, baseline

    # ---- surprisal (used by the behavioral evals; BERT lives here) ----

    def _replace_nth(self, s: str, old: str, new: str, n: int) -> str:
        parts = s.split(old)
        if n < 0 or n >= len(parts) - 1:
            raise ValueError(f"occurrence {n} requested but '{old}' appears {len(parts)-1}x")
        return old.join(parts[: n + 1]) + new + old.join(parts[n + 1:])

    def get_surprisal(
        self,
        prompt: str,
        tok_a: str,
        tok_b: str,
        mask_str: Optional[str] = None,
        crop_str: Optional[str] = None,
        n_mask_pos: int = 1,
    ) -> Tuple[float, float]:
        """Per-token surprisal of each option: (-logp(a), -logp(b)).

        `mask_str` scores at a [MASK] position (masked LMs); `crop_str` crops
        the prompt before the placeholder and scores the next token
        (autoregressive LMs). Only the first token of each option is scored.
        """
        a_id, b_id = self.token_id(tok_a), self.token_id(tok_b)

        if mask_str is not None:
            if not self.is_masked:
                raise RuntimeError("mask_str scoring requires a masked LM")
            masked = self._replace_nth(prompt, mask_str, self.tokenizer.mask_token, n_mask_pos)
            enc = self.tokenizer(masked, return_tensors="pt", add_special_tokens=True)
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            pos = (enc["input_ids"][0] == self.tokenizer.mask_token_id).nonzero()[0, 0].item()
            logits_at = logits[0, pos]
        else:
            if crop_str is not None:
                prompt = prompt.split(crop_str, 1)[0]
            with torch.no_grad():
                logits = self.model(**self._encode(prompt)).logits
            logits_at = logits[0, -1]

        logp = torch.log_softmax(logits_at.float(), dim=-1)
        return (-logp[a_id]).item(), (-logp[b_id]).item()

    def get_mean_surprisal(self, prompt: str) -> float:
        """Mean -log p(token_t | token_<t) over the prompt, in nats."""
        if self.is_masked:
            raise RuntimeError("get_mean_surprisal is autoregressive-only")
        enc = self._encode(prompt)
        with torch.no_grad():
            logits = self.model(**enc).logits
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = enc["input_ids"][0, 1:]
        return (-logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean().item()

    def close(self) -> None:
        for attr in ("model", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
