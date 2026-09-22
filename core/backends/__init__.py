"""Backend registry.

Backends are imported lazily so that, e.g., a TransformerLens run on a machine
without vLLM installed still works.
"""

from core.backends.base import (
    ABLATION_MODES,
    GenerationResult,
    InterpBackend,
    LMBackend,
    ScoredTokens,
)

#: backends usable for inference (behavioral evals)
INFERENCE_BACKENDS = ("hf", "tlens", "vllm", "api")
#: backends usable for interpretability (hooks required)
INTERP_BACKENDS = ("hf", "tlens")


def get_backend_cls(backend: str):
    """Import and return the backend class for `backend`."""
    if backend == "hf":
        from core.backends.hf import HFBackend

        return HFBackend
    if backend == "tlens":
        from core.backends.tlens import TransformerLensBackend

        return TransformerLensBackend
    if backend == "vllm":
        from core.backends.vllm import VLLMBackend

        return VLLMBackend
    if backend == "api":
        from core.backends.api import APIBackend

        return APIBackend
    raise ValueError(f"Unknown backend {backend!r}; expected one of {INFERENCE_BACKENDS}")


__all__ = [
    "ABLATION_MODES",
    "GenerationResult",
    "INFERENCE_BACKENDS",
    "INTERP_BACKENDS",
    "InterpBackend",
    "LMBackend",
    "ScoredTokens",
    "get_backend_cls",
]
