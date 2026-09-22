"""Activity measures collected from an unmodified forward pass.

Two families:

*Pattern measures* read the [seq, seq] attention probability matrix and only
apply to attention heads: `entropy`, `betweenness`, `effective_rank`.

*Activation measures* read any component's activation vector — attention head
output, MLP pre- or post-activation, residual stream — as [seq, d]:
`l2_norm`, `mean_abs`, `sparsity`, `unit_entropy`.

Results are written long-format (one row per prompt x component x measure),
which replaces the old list-valued columns and joins cleanly against ablation
and behavioral tables.
"""

from __future__ import annotations

import gc
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import utils.model_utils as model_utils
import utils.stim_utils as stim_utils
from core.backends import InterpBackend
from core.components import Component, enumerate_components, parse_components
from utils.io_utils import write_output


# ---------------------------------------------------------------- pattern measures


def _entropy(pattern: np.ndarray) -> float:
    """Mean row-entropy of a [seq, seq] attention matrix.

    `pattern` is post-softmax (rows sum to 1), so no further softmax is
    applied. Zero entries use the standard limit 0*log(0) = 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(pattern > 0, np.log(pattern), 0.0)
    return float((-(pattern * log_p).sum(axis=-1)).mean())


def _effective_rank(pattern: np.ndarray) -> float:
    """Mean participation ratio per row: "how many tokens does this row attend to".

    For a row p summing to 1 this is 1 / sum(p^2): 1 for a one-hot row, k for
    a uniform row over k tokens.
    """
    row_sums = pattern.sum(axis=-1)
    valid = np.isfinite(row_sums) & (row_sums > 0)
    if not valid.any():
        return 0.0
    p = pattern[valid]
    p = p / p.sum(axis=-1, keepdims=True)
    return float((1.0 / (p * p).sum(axis=-1)).mean())


def _betweenness(pattern: np.ndarray, threshold: Optional[float] = None) -> float:
    """Mean node betweenness centrality of the attention graph.

    Treats the pattern as a directed weighted graph (i -> j with weight a_ij)
    and converts weights to distances via -log(w), so shortest paths are paths
    of maximum attention flow.
    """
    import networkx as nx

    n = pattern.shape[0]
    finite = np.isfinite(pattern)
    mask = finite & (pattern >= threshold) if threshold is not None else finite & (pattern > 0)

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return 0.0

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        zip(rows.tolist(), cols.tolist(), (-np.log(pattern[rows, cols])).tolist())
    )
    centrality = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    return float(np.mean([centrality[i] for i in range(n)]))


# ------------------------------------------------------------- activation measures


def _l2_norm(act: np.ndarray) -> float:
    """Mean L2 norm of the activation vector across positions."""
    return float(np.linalg.norm(act, axis=-1).mean())


def _mean_abs(act: np.ndarray) -> float:
    return float(np.abs(act).mean())


def _sparsity(act: np.ndarray, tol: float = 1e-6) -> float:
    """Fraction of units at (near) zero — meaningful for post-GELU/SiLU MLPs."""
    return float((np.abs(act) <= tol).mean())


def _unit_entropy(act: np.ndarray) -> float:
    """Entropy of the normalized |activation| profile over units, averaged over positions.

    Low entropy means a few units carry the activation; high means it is spread out.
    """
    mag = np.abs(act)
    totals = mag.sum(axis=-1, keepdims=True)
    valid = (totals > 0).squeeze(-1)
    if not valid.any():
        return 0.0
    p = mag[valid] / totals[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    return float((-(p * log_p).sum(axis=-1)).mean())


PATTERN_MEASURES: Dict[str, Callable[[np.ndarray], float]] = {
    "entropy": _entropy,
    "effective_rank": _effective_rank,
    "betweenness": _betweenness,
}

ACTIVATION_MEASURES: Dict[str, Callable[[np.ndarray], float]] = {
    "l2_norm": _l2_norm,
    "mean_abs": _mean_abs,
    "sparsity": _sparsity,
    "unit_entropy": _unit_entropy,
}

ALL_MEASURES = tuple(PATTERN_MEASURES) + tuple(ACTIVATION_MEASURES)


# ------------------------------------------------------------------------ driver


def _resolve_components(
    lm: InterpBackend,
    components,
    component_kind: str,
    ablation_csv_path: Optional[str],
    n_components: int,
) -> List[Component]:
    """Components to measure: explicit spec, top-k from an ablation sweep, or all."""
    if components:
        return parse_components(components, default_kind=component_kind)
    if ablation_csv_path:
        from core.patches import top_components

        return top_components(ablation_csv_path, n_components)
    return list(enumerate_components(component_kind, lm.n_layers, lm.n_heads))


def collect_activity(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    measure: str = "entropy",
    component_kind: str = "attn_head",
    components=None,
    ablation_csv_path: Optional[str] = None,
    n_components: int = 5,
    backend: Optional[str] = None,
    subset_categories=None,
):
    """Measure per-component activity for every stimulus.

    Which components are measured, in priority order: an explicit `components`
    spec, else the top `n_components` from `ablation_csv_path`, else every
    component of `component_kind`.
    """
    if measure not in ALL_MEASURES:
        raise ValueError(f"measure must be one of {ALL_MEASURES}, got {measure!r}")

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=False
    )
    lm: InterpBackend = model_utils.load_model(model_str, backend=backend, interp=True)[0]

    comps = _resolve_components(
        lm, components, component_kind, ablation_csv_path, int(n_components)
    )

    is_pattern = measure in PATTERN_MEASURES
    if is_pattern:
        bad = [c for c in comps if c.kind != "attn_head"]
        if bad:
            raise ValueError(
                f"measure {measure!r} reads attention patterns and only applies to "
                f"attention heads; got {[c.label for c in bad]}"
            )
        fn = PATTERN_MEASURES[measure]
        layers = sorted({c.layer for c in comps})
    else:
        fn = ACTIVATION_MEASURES[measure]

    print(f"Collecting {measure} for {len(comps)} components")

    rows: List[dict] = []
    for _, row in tqdm(formatted_df.iterrows(), total=len(formatted_df), desc=measure):
        prompt = row["formatted_prompt"]

        if is_pattern:
            patterns = lm.attention_patterns(prompt, layers)
            values = {c: fn(patterns[c.layer][c.index].numpy()) for c in comps}
            del patterns
        else:
            acts = lm.cache(prompt, comps)
            values = {c: fn(acts[c].float().cpu().numpy()) for c in comps}
            del acts

        for comp, value in values.items():
            rows.append(
                {
                    "category": row.get("Category"),
                    "prompt": row["prompt"],
                    "a": row["a"],
                    "b": row["b"],
                    "component": comp.label,
                    "component_kind": comp.kind,
                    "layer": comp.layer,
                    "head": comp.index,
                    "measure": measure,
                    "value": value,
                }
            )

        gc.collect()
        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    filename = (
        f"activity_{measure}_{component_kind}_{model_str}"
        f"{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(results, out_dest_path, filename, in_place=False)
    print(f"Wrote {len(results)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path
