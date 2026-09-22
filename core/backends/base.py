"""Backend interfaces shared by every model wrapper.

Two levels:

`LMBackend`      inference only (generation + optional next-token logits).
                 Implemented by HF, TransformerLens, vLLM and the API wrapper.
`InterpBackend`  adds caching, ablation, patching and gradient attribution.
                 Implemented by HF and TransformerLens only.

Keeping the interp surface in a separate base class is what lets
`exp_battery` reject an interp run on a vLLM/API model with a clear error
instead of failing deep inside a hook.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from core.components import Component

# Ablation modes understood by every InterpBackend.
ABLATION_MODES = ("zero", "mean", "resample")


@dataclass
class GenerationResult:
    """Uniform generation return so CoT extraction works across backends.

    `reasoning` holds chain-of-thought when the provider exposes it separately
    (OpenAI reasoning summaries, DeepSeek `reasoning_content`, Gemini thought
    parts). Local models put everything in `text`.
    """

    text: str
    reasoning: Optional[str] = None
    raw: Any = None
    usage: Optional[Dict[str, Any]] = None
    provider: str = ""
    model: str = ""
    response_id: Optional[str] = None


@dataclass
class ScoredTokens:
    """Scores for the two answer options at the final prompt position.

    `logit_a`/`logit_b` are raw logits for local backends and log-probabilities
    for vLLM (which never exposes the unnormalized vector). `logit_diff` is
    identical either way, since the softmax normalizer cancels in the
    difference. `entropy` is NaN when the full distribution is unavailable.
    """

    logit_a: float
    logit_b: float
    logit_diff: float
    entropy: float = float("nan")
    is_logprob: bool = False


def resolve_cache_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Model cache directory: explicit arg, then HF_HOME/CACHE_DIR env vars."""
    return explicit or os.getenv("CACHE_DIR") or os.getenv("HF_HOME") or None


class LMBackend(ABC):
    """Minimum surface every model wrapper provides."""

    #: whether this backend can report next-token logits at all
    supports_logits: bool = True
    #: whether this backend can run hooks/interventions
    supports_interp: bool = False

    def __init__(self, name: str, cfg: Dict[str, Any]):
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 10) -> GenerationResult:
        """Greedy-decode `prompt`."""

    @abstractmethod
    def score_options(self, prompt: str, option_a: str, option_b: str) -> ScoredTokens:
        """Score both answer options at the final position of `prompt`."""

    def close(self) -> None:
        """Release GPU memory. Safe to call more than once."""


class InterpBackend(LMBackend):
    """Backend that supports activation access and interventions."""

    supports_interp = True

    # ---- model shape ----

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int: ...

    # ---- tokenization ----

    @abstractmethod
    def to_tokens(self, prompt: str) -> torch.Tensor:
        """Tokenize to a [1, seq] tensor on the model's input device."""

    @abstractmethod
    def token_id(self, s: str) -> int:
        """First token id of `s`, without a BOS prefix."""

    # ---- forward passes ----

    @abstractmethod
    def logits(self, prompt: str) -> torch.Tensor:
        """Logits for `prompt`, shape [1, seq, vocab]."""

    @abstractmethod
    def cache(
        self, prompt: str, components: Sequence[Component]
    ) -> Dict[Component, torch.Tensor]:
        """Activations for each component.

        Head-indexed components are returned already sliced to their head:
        `attn_head` -> [seq, d_head]. Whole-layer components keep their full
        trailing dimension: [seq, d].
        """

    @abstractmethod
    def cache_layers(
        self,
        prompt: str,
        kinds: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        """Full activations keyed by `(kind, layer)`, batch dim kept.

        Unlike `cache`, nothing is sliced to a head — this is the shape
        `act_and_grad` returns, so the two can be combined for attribution.
        """

    @abstractmethod
    def attention_patterns(
        self, prompt: str, layers: Optional[Iterable[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """Post-softmax attention probabilities per layer, shape [n_heads, seq, seq].

        Separate from `cache` because attention probabilities are not a module
        output in `transformers` — they need `output_attentions=True`.
        """

    @abstractmethod
    def run_with_ablation(
        self,
        prompt: str,
        components: Sequence[Component],
        mode: str = "zero",
        replacements: Optional[Dict[Component, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with `components` ablated. Returns [1, seq, vocab].

        `mode="zero"` writes zeros; `"mean"`/`"resample"` write the tensors in
        `replacements` (precomputed by the caller from a corpus or a source
        prompt).
        """

    @abstractmethod
    def run_with_patch(
        self, prompt: str, patches: Dict[Component, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with each component overwritten by a donor activation."""

    @abstractmethod
    def act_and_grad(
        self,
        prompt: str,
        a_id: int,
        b_id: int,
        kinds: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ) -> Tuple[Dict[Tuple[str, int], torch.Tensor],
               Dict[Tuple[str, int], torch.Tensor],
               float]:
        """One fwd + one bwd of the logit-diff metric.

        Returns `(activations, gradients, baseline_logit_diff)` keyed by
        `(kind, layer)`, both detached. This is the primitive behind
        logit-attribution ablation and patching: a first-order estimate of the
        effect of changing an activation by `delta` is `sum(grad * delta)`.
        """

    # ---- shared helpers ----

    def logit_diff(self, logits: torch.Tensor, a_id: int, b_id: int) -> float:
        """logit(a) - logit(b) at the final position."""
        return (logits[0, -1, a_id] - logits[0, -1, b_id]).item()
