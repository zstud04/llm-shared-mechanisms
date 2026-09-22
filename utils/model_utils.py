"""Model loading, backend selection, and multi-GPU sizing."""

from __future__ import annotations

import gc
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from core.backends import INFERENCE_BACKENDS, INTERP_BACKENDS, LMBackend, get_backend_cls

CONFIG_DIR = os.getenv("WORLD_MODELS_CONFIG", "config")

with open(os.path.join(CONFIG_DIR, "models.json"), "r", encoding="utf-8") as f:
    # Keys starting with "_" are documentation, not models.
    MODELS_CFG: Dict[str, Dict[str, Any]] = {
        k: v for k, v in json.load(f).items() if not k.startswith("_")
    }

# Above this parameter count a model will not fit one 80GB GPU in bf16, so we
# shard it across every GPU the job was allocated. See chtc/README.md for how
# to request a multi-GPU slot.
SHARD_THRESHOLD_B = 30

API_TYPES = ("openai", "gemini", "deepseek")

API_KEY_ENV = {
    "openai": "OPENAI_KEY",
    "gemini": "GEMINI_KEY",
    "deepseek": "DEEPSEEK_KEY",
}


def get_model_cfg(model_str: str) -> Dict[str, Any]:
    cfg = MODELS_CFG.get(model_str)
    if cfg is None:
        raise KeyError(
            f"Model not found in {CONFIG_DIR}/models.json: {model_str}. "
            f"Known models: {sorted(MODELS_CFG)}"
        )
    return cfg


def visible_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def needs_sharding(cfg: Dict[str, Any]) -> bool:
    """True when the model is large enough to need multiple GPUs."""
    return float(cfg.get("n_params_b", 0)) > SHARD_THRESHOLD_B


def resolve_backend(cfg: Dict[str, Any], backend: Optional[str], interp: bool) -> str:
    """Pick and validate a backend for this model and run type."""
    if cfg["type"] in API_TYPES:
        if backend not in (None, "api"):
            raise ValueError(f"{cfg['name']} is an API model; backend must be 'api'")
        if interp:
            raise ValueError(f"{cfg['name']} is an API model and cannot run interp methods")
        return "api"

    chosen = backend or cfg.get("default_backend") or ("tlens" if cfg.get("tl_support") else "hf")

    allowed = INTERP_BACKENDS if interp else INFERENCE_BACKENDS
    if chosen not in allowed:
        raise ValueError(
            f"backend={chosen!r} cannot run {'interp' if interp else 'inference'} methods; "
            f"choose from {allowed}"
        )
    if chosen == "tlens" and not cfg.get("tl_support"):
        raise ValueError(
            f"{cfg['name']} has no TransformerLens conversion (tl_support=false); "
            "use backend='hf'"
        )
    return chosen


def load_model(
    model_str: str,
    backend: Optional[str] = None,
    interp: bool = False,
    n_gpus: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **backend_kwargs,
) -> Tuple[LMBackend, Dict[str, Any]]:
    """Load `model_str` on a suitable backend.

    `backend` overrides the config default. `interp=True` restricts the choice
    to backends that support hooks and makes the error explicit when a model
    cannot run interp methods at all.
    """
    cfg = get_model_cfg(model_str)
    chosen = resolve_backend(cfg, backend, interp)
    cls = get_backend_cls(chosen)

    if chosen == "api":
        env_var = API_KEY_ENV[cfg["type"]]
        key = os.getenv(env_var)
        if not key:
            raise ValueError(f"Missing API key for '{cfg['type']}'; set ${env_var}")
        return cls(model_str, cfg, api_key=key, **backend_kwargs), cfg

    gpus = n_gpus if n_gpus is not None else visible_gpu_count()
    if needs_sharding(cfg) and gpus < 2:
        print(
            f"[warn] {model_str} is ~{cfg.get('n_params_b')}B but only {gpus} GPU(s) "
            "are visible; expect OOM. Request a multi-GPU slot (see chtc/README.md)."
        )

    if chosen == "hf":
        # device_map="auto" lets accelerate shard across all visible GPUs.
        # WORLD_MODELS_DEVICE_MAP=none loads onto a single device instead,
        # which is handy for local debugging without accelerate installed.
        env_map = os.getenv("WORLD_MODELS_DEVICE_MAP")
        default_map = None if (env_map or "").lower() == "none" else (env_map or "auto")
        backend_kwargs.setdefault("device_map", default_map)
    elif chosen == "tlens":
        # TransformerLens splits blocks layer-wise over n_devices.
        backend_kwargs.setdefault("n_devices", max(1, gpus) if needs_sharding(cfg) else 1)
    elif chosen == "vllm":
        backend_kwargs.setdefault(
            "tensor_parallel_size", max(1, gpus) if needs_sharding(cfg) else 1
        )

    return cls(model_str, cfg, cache_dir=cache_dir, **backend_kwargs), cfg


def mem_cleanup(llm: Optional[LMBackend]) -> None:
    """Release a loaded model and its GPU memory."""
    if llm is None:
        return
    llm.close()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
