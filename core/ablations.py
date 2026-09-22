"""Ablation experiments.

Two entry points:

`sweep_ablations`     score every component of one kind (all attention heads,
                      all MLP layers, ...) for causal importance.
`run_with_ablation`   ablate a *given* set of components together and record
                      the downstream change in logits / logit difference.

Both support `method="direct"` (re-run the model per intervention; exact) and
`method="attribution"` (one fwd + bwd, first-order estimate; far cheaper on
large grids). See `core/attribution.py` for the estimator.
"""

from __future__ import annotations

import gc
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import utils.model_utils as model_utils
import utils.stim_utils as stim_utils
from core.attribution import ablation_attribution
from core.backends import InterpBackend
from core.components import Component, enumerate_components, parse_components
from utils.io_utils import write_output

METHODS = ("direct", "attribution")


def check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _load_interp_model(model_str: str, backend: Optional[str]) -> InterpBackend:
    lm, _ = model_utils.load_model(model_str, backend=backend, interp=True)
    return lm


def _direct_scores(
    lm: InterpBackend,
    prompt: str,
    components: Sequence[Component],
    a_id: int,
    b_id: int,
    baseline_diff: float,
) -> Dict[Component, float]:
    """Exact effect of ablating each component on its own."""
    scores = {}
    for comp in components:
        ablated = lm.run_with_ablation(prompt, [comp])
        scores[comp] = baseline_diff - lm.logit_diff(ablated, a_id, b_id)
    return scores


def sweep_ablations(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    component_kind: str = "attn_head",
    method: str = "direct",
    backend: Optional[str] = None,
    subset_categories=None,
    in_place: bool = False,
):
    """Score every component of `component_kind` for causal importance.

    Impact is `baseline_logit_diff - ablated_logit_diff`, averaged over the
    prompts in each category: positive means the component supports the
    correct answer.
    """
    check_method(method)

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=False
    )
    lm = _load_interp_model(model_str, backend)

    components = list(enumerate_components(component_kind, lm.n_layers, lm.n_heads))
    print(
        f"Ablating {len(components)} {component_kind} components "
        f"({lm.n_layers} layers) via {method}"
    )

    subset_str = stim_utils.category_suffix(subset_categories).lstrip("_") or None
    rows: List[dict] = []

    for category in formatted_df["Category"].unique():
        category_df = formatted_df[formatted_df["Category"] == category]
        prompts = category_df["formatted_prompt"].tolist()
        per_prompt: List[Dict[Component, float]] = []

        for _, row in tqdm(category_df.iterrows(), total=len(category_df), desc=category):
            prompt = row["formatted_prompt"]
            a_id, b_id = lm.token_id(row["a"]), lm.token_id(row["b"])

            if method == "direct":
                baseline = lm.logit_diff(lm.logits(prompt), a_id, b_id)
                per_prompt.append(_direct_scores(lm, prompt, components, a_id, b_id, baseline))
            else:
                acts, grads, _ = lm.act_and_grad(prompt, a_id, b_id, [component_kind])
                per_prompt.append(ablation_attribution(acts, grads))
                del acts, grads

            gc.collect()
            torch.cuda.empty_cache()

        for comp in components:
            effects = [p[comp] for p in per_prompt]
            rows.append(
                {
                    "category": category,
                    "component": comp.label,
                    "component_kind": comp.kind,
                    "layer": comp.layer,
                    "head": comp.index,
                    "method": method,
                    "avg_impact": float(np.mean(effects)),
                    "individual_prompt_effects": effects,
                    "prompts": prompts,
                    "subset_categories": subset_str,
                }
            )

    results = pd.DataFrame(rows)
    filename = (
        f"ablation_{component_kind}_{method}_{model_str}"
        f"{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(results, out_dest_path, filename, in_place=False)
    print(f"Wrote {len(results)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path


def get_n_components_variance(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    component_kind: str = "attn_head",
    variance_threshold: float = 0.90,
    backend: Optional[str] = None,
    subset_categories=None,
    in_place: bool = False,
):
    """Estimate how many top components carry most of the causal signal.
    """
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold!r}"
        )

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=False
    )
    lm = _load_interp_model(model_str, backend)

    components = list(enumerate_components(component_kind, lm.n_layers, lm.n_heads))
    print(
        f"Scoring {len(components)} {component_kind} components via DLA over "
        f"{len(formatted_df)} stimuli"
    )

    # Sum DLA impact per component across every stimulus (mean taken at the end).
    totals: Dict[Component, float] = {comp: 0.0 for comp in components}
    for _, row in tqdm(formatted_df.iterrows(), total=len(formatted_df), desc=model_str):
        prompt = row["formatted_prompt"]
        a_id, b_id = lm.token_id(row["a"]), lm.token_id(row["b"])

        acts, grads, _ = lm.act_and_grad(prompt, a_id, b_id, [component_kind])
        scores = ablation_attribution(acts, grads)
        for comp in components:
            totals[comp] += scores[comp]
        del acts, grads
        gc.collect()
        torch.cuda.empty_cache()

    n_stim = len(formatted_df)
    mean_impact = {comp: totals[comp] / n_stim for comp in components}
    sq_impact = {comp: mean_impact[comp] ** 2 for comp in components}

    # Rank by squared impact (variance contribution), descending.
    ranked = sorted(components, key=lambda c: sq_impact[c], reverse=True)
    total_sq = sum(sq_impact.values())
    cumulative = np.cumsum([sq_impact[c] for c in ranked])
    cum_frac = cumulative / total_sq if total_sq > 0 else np.zeros(len(ranked))

    # Smallest k whose cumulative variance reaches the threshold.
    reached = np.searchsorted(cum_frac, variance_threshold, side="left")
    n_components = int(reached) + 1 if len(cum_frac) else 0

    subset_str = stim_utils.category_suffix(subset_categories).lstrip("_") or None
    rows: List[dict] = []
    for rank, comp in enumerate(ranked):
        rows.append(
            {
                "rank": rank + 1,
                "component": comp.label,
                "component_kind": comp.kind,
                "layer": comp.layer,
                "head": comp.index,
                "mean_impact": mean_impact[comp],
                "squared_impact": sq_impact[comp],
                "cumulative_variance": float(cum_frac[rank]),
                "variance_threshold": variance_threshold,
                "n_components_for_threshold": n_components,
                "n_stimuli": n_stim,
                "method": "attribution",
                "subset_categories": subset_str,
            }
        )

    results = pd.DataFrame(rows)
    filename = (
        f"n_components_variance_{component_kind}_{model_str}"
        f"{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(results, out_dest_path, filename, in_place=False)
    print(
        f"{n_components}/{len(components)} {component_kind} components explain "
        f">= {variance_threshold:.0%} of causal variance; wrote {path}"
    )
    model_utils.mem_cleanup(lm)
    return path


def run_with_ablation(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    components,
    ablation_name: Optional[str] = None,
    method: str = "direct",
    backend: Optional[str] = None,
    subset_categories=None,
    in_place: bool = False,
):
    """Ablate a specific set of components together, per stimulus.

    `components` is a spec string (e.g. "L38H12;L20.mlp_post"), a list of such
    strings, or `Component` objects. All listed components are ablated in the
    *same* forward pass, so the result is their joint effect.

    `ablation_name` labels the run and is embedded in the output column names,
    which makes several ablation sets comparable side by side in one CSV.
    """
    check_method(method)
    comps = parse_components(components)
    if not comps:
        raise ValueError("no components to ablate")

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=False
    )
    lm = _load_interp_model(model_str, backend)

    for comp in comps:
        if comp.layer >= lm.n_layers:
            raise ValueError(f"{comp} exceeds model depth ({lm.n_layers} layers)")

    # Column suffix: model alone, or name + model when the run is named.
    suffix = f"{ablation_name}_{model_str}" if ablation_name else model_str
    print(f"Ablating {[c.label for c in comps]} -> columns *_{suffix}")

    out = formatted_df.copy()
    out[f"ablation_components_{suffix}"] = ";".join(c.label for c in comps)
    out[f"ablation_method_{suffix}"] = method

    base_a, base_b, base_diff = [], [], []
    abl_a, abl_b, abl_diff, delta = [], [], [], []

    for _, row in tqdm(out.iterrows(), total=len(out), desc=suffix):
        prompt = row["formatted_prompt"]
        a_id, b_id = lm.token_id(row["a"]), lm.token_id(row["b"])

        baseline_logits = lm.logits(prompt)
        b_logit_a = baseline_logits[0, -1, a_id].item()
        b_logit_b = baseline_logits[0, -1, b_id].item()
        base_a.append(b_logit_a)
        base_b.append(b_logit_b)
        base_diff.append(b_logit_a - b_logit_b)

        if method == "direct":
            ablated_logits = lm.run_with_ablation(prompt, comps)
            a_logit_a = ablated_logits[0, -1, a_id].item()
            a_logit_b = ablated_logits[0, -1, b_id].item()
            abl_a.append(a_logit_a)
            abl_b.append(a_logit_b)
            abl_diff.append(a_logit_a - a_logit_b)
            delta.append((b_logit_a - b_logit_b) - (a_logit_a - a_logit_b))
        else:
            # First-order estimate: individual post-ablation logits are not
            # recoverable from the metric gradient, only their difference.
            kinds = sorted({c.kind for c in comps})
            layers = sorted({c.layer for c in comps})
            acts, grads, baseline = lm.act_and_grad(prompt, a_id, b_id, kinds, layers)
            scores = ablation_attribution(acts, grads)
            est = sum(scores[c] for c in comps)
            abl_a.append(float("nan"))
            abl_b.append(float("nan"))
            abl_diff.append(baseline - est)
            delta.append(est)
            del acts, grads

        gc.collect()
        torch.cuda.empty_cache()

    out[f"baseline_logit_a_{suffix}"] = base_a
    out[f"baseline_logit_b_{suffix}"] = base_b
    out[f"baseline_logit_diff_{suffix}"] = base_diff
    out[f"ablated_logit_a_{suffix}"] = abl_a
    out[f"ablated_logit_b_{suffix}"] = abl_b
    out[f"ablated_logit_diff_{suffix}"] = abl_diff
    out[f"delta_logit_diff_{suffix}"] = delta

    filename = (
        f"ablation_run_{model_str}"
        f"{('_' + ablation_name) if ablation_name else ''}"
        f"{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(
        out, out_dest_path, filename, in_place=in_place, source_path=stim_csv_path
    )
    print(f"Wrote {len(out)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path
