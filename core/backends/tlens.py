"""TransformerLens backend: inference + full interpretability support."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.distributions.categorical import Categorical

import core.chat_templates as chat_templates
from core.backends.base import (
    GenerationResult,
    InterpBackend,
    ScoredTokens,
    resolve_cache_dir,
)
from core.components import TL_HOOK_NAMES, Component


class TransformerLensBackend(InterpBackend):
    """Wraps `HookedTransformer`.

    Only models with a TransformerLens conversion are usable here; the model
    config's `tl_support` flag gates that. Large models are sharded across
    GPUs with `n_devices`, which TransformerLens handles by splitting blocks
    layer-wise.
    """

    def __init__(
        self,
        name: str,
        cfg: Dict[str, Any],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        n_devices: int = 1,
        enable_templates: bool = True,
    ):
        super().__init__(name, cfg)
        from transformer_lens import HookedTransformer

        self.device = device
        self.enable_templates = enable_templates
        self.prompt_template = chat_templates.template_for(cfg)

        self.model = HookedTransformer.from_pretrained(
            cfg["path"],
            device=device,
            cache_dir=resolve_cache_dir(cache_dir),
            dtype=dtype,
            n_devices=n_devices,
        )
        self.model.eval()

    # ---- shape ----

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def n_heads(self) -> int:
        return self.model.cfg.n_heads

    # ---- prompts / tokens ----

    def _format(self, instruction: str) -> str:
        if self.prompt_template and self.enable_templates:
            return self.prompt_template.format(instruction=instruction)
        return instruction

    def to_tokens(self, prompt: str) -> torch.Tensor:
        return self.model.to_tokens(self._format(prompt)).to(self.device)

    def token_id(self, s: str) -> int:
        return self.model.to_tokens(s, prepend_bos=False)[0, 0].item()

    # ---- forward passes ----

    def logits(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            return self.model(self.to_tokens(prompt), return_type="logits")

    def generate(self, prompt: str, max_new_tokens: int = 10) -> GenerationResult:
        toks = self.to_tokens(prompt)
        out = self.model.generate(
            toks, max_new_tokens=max_new_tokens, do_sample=False, prepend_bos=False
        )
        text = self.model.to_string(out[0, toks.shape[1]:])
        return GenerationResult(text=text, provider="transformer_lens", model=self.name)

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

    # ---- activations ----

    @staticmethod
    def _slice(component: Component, act: torch.Tensor) -> torch.Tensor:
        """Drop the batch dim and select a head when the component has one."""
        if component.index is not None:
            return act[0, :, component.index, :]
        return act[0]

    def cache(
        self, prompt: str, components: Sequence[Component]
    ) -> Dict[Component, torch.Tensor]:
        names = sorted({c.tl_hook for c in components})
        with torch.no_grad():
            _, cache = self.model.run_with_cache(self.to_tokens(prompt), names_filter=names)
        out = {c: self._slice(c, cache[c.tl_hook]).detach().clone() for c in components}
        del cache
        return out

    def cache_layers(
        self,
        prompt: str,
        kinds: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        layer_list = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        wanted = {
            TL_HOOK_NAMES[kind].format(layer=layer): (kind, layer)
            for kind in kinds
            for layer in layer_list
        }
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                self.to_tokens(prompt), names_filter=list(wanted)
            )
        out = {key: cache[name].detach().clone() for name, key in wanted.items()}
        del cache
        return out

    def attention_patterns(
        self, prompt: str, layers: Optional[Iterable[int]] = None
    ) -> Dict[int, torch.Tensor]:
        layer_list = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        names = [TL_HOOK_NAMES["attn_pattern"].format(layer=l) for l in layer_list]
        with torch.no_grad():
            _, cache = self.model.run_with_cache(self.to_tokens(prompt), names_filter=names)
        out = {l: cache[n][0].detach().float().cpu() for l, n in zip(layer_list, names)}
        del cache
        return out

    def _write_hooks(self, writes: Dict[Component, Optional[torch.Tensor]]):
        """Build TL fwd_hooks that overwrite each component's activation.

        A value of None means "write zeros" (ablation); a tensor is copied in
        (patching / mean-ablation).
        """
        by_hook: Dict[str, List[Tuple[Component, Optional[torch.Tensor]]]] = {}
        for comp, value in writes.items():
            if not comp.is_writable:
                raise ValueError(f"{comp.kind} is a read-only site and cannot be written")
            by_hook.setdefault(comp.tl_hook, []).append((comp, value))

        def make_hook(entries):
            # TransformerLens calls fwd hooks as hook(act, hook=hook_point),
            # so the second parameter must be named `hook`.
            def hook(act, hook):
                for comp, value in entries:
                    if comp.index is not None:
                        act[:, :, comp.index, :] = (
                            0 if value is None else value.to(act.dtype).to(act.device)
                        )
                    else:
                        act[:, :, :] = (
                            0 if value is None else value.to(act.dtype).to(act.device)
                        )
                return act

            return hook

        return [(name, make_hook(entries)) for name, entries in by_hook.items()]

    def run_with_ablation(
        self,
        prompt: str,
        components: Sequence[Component],
        mode: str = "zero",
        replacements: Optional[Dict[Component, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if mode == "zero":
            writes = {c: None for c in components}
        else:
            if not replacements:
                raise ValueError(f"mode={mode!r} requires `replacements`")
            writes = {c: replacements[c] for c in components}

        with self.model.hooks(fwd_hooks=self._write_hooks(writes)):
            with torch.no_grad():
                return self.model(self.to_tokens(prompt), return_type="logits")

    def run_with_patch(
        self, prompt: str, patches: Dict[Component, torch.Tensor]
    ) -> torch.Tensor:
        with self.model.hooks(fwd_hooks=self._write_hooks(dict(patches))):
            with torch.no_grad():
                return self.model(self.to_tokens(prompt), return_type="logits")

    def act_and_grad(
        self,
        prompt: str,
        a_id: int,
        b_id: int,
        kinds: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ):
        layer_list = list(range(self.n_layers)) if layers is None else sorted(set(layers))
        wanted = {
            TL_HOOK_NAMES[kind].format(layer=layer): (kind, layer)
            for kind in kinds
            for layer in layer_list
        }

        self.model.reset_hooks()
        stored: Dict[str, torch.Tensor] = {}

        def save_hook(act, hook):
            stored[hook.name] = act  # keep the graph reference; do NOT detach
            return act

        for name in wanted:
            self.model.add_hook(name, save_hook, "fwd")

        # Deliberately outside no_grad: we need the graph for autograd.grad.
        logits = self.model(self.to_tokens(prompt), return_type="logits")
        metric = logits[0, -1, a_id] - logits[0, -1, b_id]

        names = list(wanted)
        act_list = [stored[n] for n in names]
        grad_list = torch.autograd.grad(metric, act_list)

        self.model.reset_hooks()
        acts = {wanted[n]: a.detach() for n, a in zip(names, act_list)}
        grads = {wanted[n]: g.detach() for n, g in zip(names, grad_list)}
        baseline = metric.item()

        del stored, logits, metric, act_list, grad_list
        return acts, grads, baseline

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            try:
                model.reset_hooks()
            except Exception:
                pass
            del self.model
        torch.cuda.empty_cache()
