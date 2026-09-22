"""vLLM backend: batched inference only (no hooks, so no interpretability).

vLLM never exposes the unnormalized logit vector, so option scores come back
as log-probabilities. That is not a problem for the analyses here: the softmax
normalizer cancels in `logit_a - logit_b`, so `logit_diff` is numerically
identical to the HF/TransformerLens value. Individual `logit_a`/`logit_b` are
log-probs (flagged via `ScoredTokens.is_logprob`) and full-vocab entropy is
unavailable, so it is reported as NaN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import core.chat_templates as chat_templates
from core.backends.base import GenerationResult, LMBackend, ScoredTokens, resolve_cache_dir


class VLLMBackend(LMBackend):
    """Wraps `vllm.LLM`.

    `tensor_parallel_size` shards one model across GPUs within a job, which is
    the path for >30B inference on a multi-GPU CHTC slot. For models that fit
    on one GPU, prefer several single-GPU jobs instead (see the CHTC docs in
    `chtc/README.md`).
    """

    supports_interp = False

    def __init__(
        self,
        name: str,
        cfg: Dict[str, Any],
        cache_dir: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        enable_templates: bool = True,
    ):
        super().__init__(name, cfg)
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self.enable_templates = enable_templates
        self.prompt_template = chat_templates.template_for(cfg)

        self.llm = LLM(
            model=cfg["path"],
            download_dir=resolve_cache_dir(cache_dir),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _format(self, instruction: str) -> str:
        if self.prompt_template and self.enable_templates:
            return self.prompt_template.format(instruction=instruction)
        return instruction

    def generate(self, prompt: str, max_new_tokens: int = 10) -> GenerationResult:
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(
        self, prompts: List[str], max_new_tokens: int = 10
    ) -> List[GenerationResult]:
        """Batched greedy decode. This is the reason to use vLLM at all —
        callers with many prompts should prefer it over `generate`."""
        params = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = self.llm.generate([self._format(p) for p in prompts], params)
        return [
            GenerationResult(
                text=o.outputs[0].text, raw=o, provider="vllm", model=self.name
            )
            for o in outputs
        ]

    def score_options(self, prompt: str, option_a: str, option_b: str) -> ScoredTokens:
        """Score both options by appending each and reading its prompt logprob."""
        base = self._format(prompt)
        params = self._SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

        scored = []
        for option in (option_a, option_b):
            first_tok = self.tokenizer.encode(option, add_special_tokens=False)[0]
            text = base + self.tokenizer.decode([first_tok])
            out = self.llm.generate([text], params)[0]
            # prompt_logprobs[i] holds the logprob of token i given tokens <i;
            # the final entry is the appended option token.
            last = out.prompt_logprobs[-1]
            scored.append(float(last[first_tok].logprob))

        return ScoredTokens(
            logit_a=scored[0],
            logit_b=scored[1],
            logit_diff=scored[0] - scored[1],
            entropy=float("nan"),  # vLLM does not expose the full distribution
            is_logprob=True,
        )

    def close(self) -> None:
        if hasattr(self, "llm"):
            del self.llm
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
