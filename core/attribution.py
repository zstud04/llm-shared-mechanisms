"""Reducing activation x gradient tensors to per-component scores.

Both ablation and patching support a `logit_attribution` method: instead of
re-running the model once per intervention, take one forward + one backward
pass of the metric (logit_a - logit_b) and estimate the effect of a change
`delta` in an activation as `sum(grad * delta)`.

For zero-ablation `delta = -act`, so the first-order estimate of
`baseline_diff - ablated_diff` is `sum(grad * act)` — the same sign convention
the direct method uses. For patching, `delta = act_source - act_null`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.components import HEAD_INDEXED_KINDS, Component


def reduce_to_components(
    kind: str,
    layer: int,
    tensor: torch.Tensor,
) -> Dict[Component, float]:
    """Sum a per-activation tensor down to one score per component.

    Head-indexed activations are [batch, pos, n_heads, d_head] and reduce over
    (batch, pos, d_head), leaving one value per head. Whole-layer activations
    are [batch, pos, d] and reduce to a single value.
    """
    t = tensor.float()
    if kind in HEAD_INDEXED_KINDS:
        if t.ndim != 4:
            raise ValueError(f"{kind} expects a 4-D [b, pos, head, d] tensor, got {tuple(t.shape)}")
        per_head = t.sum(dim=(0, 1, 3)).cpu()
        return {Component(kind, layer, h): per_head[h].item() for h in range(per_head.shape[0])}
    return {Component(kind, layer): t.sum().item()}


def ablation_attribution(
    acts: Dict[tuple, torch.Tensor],
    grads: Dict[tuple, torch.Tensor],
) -> Dict[Component, float]:
    """First-order effect of zero-ablating each component.

    Matches the direct method's sign: positive means ablating the component
    lowers the logit difference (the component supports the correct answer).
    """
    scores: Dict[Component, float] = {}
    for key, act in acts.items():
        kind, layer = key
        scores.update(reduce_to_components(kind, layer, act * grads[key].float()))
    return scores


def patch_attribution(
    null_acts: Dict[tuple, torch.Tensor],
    null_grads: Dict[tuple, torch.Tensor],
    source_acts: Dict[tuple, torch.Tensor],
) -> Dict[Component, float]:
    """First-order effect of patching each component from a source prompt."""
    scores: Dict[Component, float] = {}
    for key, null_act in null_acts.items():
        kind, layer = key
        delta = source_acts[key].float() - null_act.float()
        scores.update(reduce_to_components(kind, layer, delta * null_grads[key].float()))
    return scores
