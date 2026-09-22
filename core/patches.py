"""Causal interchange (activation patching).

For each stimulus we build "source" prompts by substituting either a *content*
word (semantically irrelevant to the answer) or a *critical* word (which
changes the correct answer), then copy one component's activation from the
source into the original forward pass and measure the change in logit
difference.

`method="direct"` copies the activation and re-runs the model.
`method="attribution"` estimates the same quantity from one fwd + bwd on the
original prompt, as `sum(grad * (act_source - act_null))`.

The set of components patched is chosen from a prior ablation sweep (top-k by
mean absolute impact), and can be any component kind, not just attention heads.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

import utils.model_utils as model_utils
import utils.stim_utils as stim_utils
from core.ablations import METHODS, check_method
from core.attribution import patch_attribution
from core.backends import InterpBackend
from core.components import Component
from utils.io_utils import write_output


def top_components(ablation_csv_path: str, n: int) -> List[Component]:
    """Top-n components from an ablation sweep, by mean |impact| across categories."""
    df = pd.read_csv(ablation_csv_path)

    # Sweeps written before the component refactor only have layer/head.
    kind_col = df["component_kind"] if "component_kind" in df.columns else "attn_head"
    df = df.assign(component_kind=kind_col)

    ranked = (
        df.groupby(["component_kind", "layer", "head"], dropna=False)["avg_impact"]
        .agg(lambda x: x.abs().mean())
        .reset_index()
        .sort_values("avg_impact", ascending=False)
        .head(int(n))
    )
    return [
        Component(
            r["component_kind"],
            int(r["layer"]),
            None if pd.isna(r["head"]) else int(r["head"]),
        )
        for _, r in ranked.iterrows()
    ]


def run_content_patching(
    model_str: str,
    stim_csv_path: str,
    patch_stim_csv_path: str,
    ablation_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    n_components: int = 5,
    method: str = "direct",
    backend: Optional[str] = None,
    subset_categories=None,
    in_place: bool = False,
):
    """Patch content vs critical substitutions through the top-k components."""
    check_method(method)
    n_components = int(n_components)

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories=None, random_shuffle=False
    )
    substitutions = pd.read_csv(patch_stim_csv_path)
    formatted_df = formatted_df.merge(substitutions, on="prompt", how="inner")
    formatted_df = stim_utils.apply_category_subset(formatted_df, subset_categories)

    lm: InterpBackend = model_utils.load_model(model_str, backend=backend, interp=True)[0]

    comps = top_components(ablation_csv_path, n_components)
    kinds = sorted({c.kind for c in comps})
    layers = sorted({c.layer for c in comps})
    print(f"Patching top-{n_components} components via {method}: {[c.label for c in comps]}")

    instruction = stim_utils.get_instruction(prompt_key)

    def source_prompt(base_prompt, sub_word, match_word, option_a, option_b) -> str:
        return instruction.format(
            prompt=base_prompt.replace(match_word, sub_word), a=option_a, b=option_b
        )

    results: List[dict] = []

    for category in formatted_df["Category"].unique():
        category_df = formatted_df[formatted_df["Category"] == category].reset_index(drop=True)

        for _, row in tqdm(category_df.iterrows(), total=len(category_df), desc=category):
            required = ("content_match", "critical_match", "content_subs", "critical_subs")
            if any(pd.isna(row.get(col)) for col in required):
                continue

            option_a, option_b = row["a"], row["b"]
            a_id, b_id = lm.token_id(option_a), lm.token_id(option_b)

            null_prompt = row["formatted_prompt"]
            null_tokens = lm.to_tokens(null_prompt)
            null_len = null_tokens.shape[1]

            if method == "direct":
                null_ref_ld = lm.logit_diff(lm.logits(null_prompt), a_id, b_id)
            else:
                # One fwd+bwd covers every component and every substitution.
                null_acts, null_grads, null_ref_ld = lm.act_and_grad(
                    null_prompt, a_id, b_id, kinds, layers
                )

            for patch_kind, subs_field, match_word in (
                ("content", row["content_subs"], row["content_match"]),
                ("critical", row["critical_subs"], row["critical_match"]),
            ):
                for sub in (s.strip() for s in str(subs_field).split(",")):
                    src_prompt = source_prompt(
                        row["prompt"], sub, match_word, option_a, option_b
                    )
                    # Patching requires token-aligned prompts.
                    if lm.to_tokens(src_prompt).shape[1] != null_len:
                        continue

                    if method == "direct":
                        donors = lm.cache(src_prompt, comps)
                        deltas = {}
                        for comp in comps:
                            patched = lm.run_with_patch(null_prompt, {comp: donors[comp]})
                            deltas[comp] = lm.logit_diff(patched, a_id, b_id) - null_ref_ld
                        del donors
                    else:
                        src_acts = lm.cache_layers(src_prompt, kinds, layers)
                        scores = patch_attribution(null_acts, null_grads, src_acts)
                        deltas = {comp: scores[comp] for comp in comps}
                        del src_acts

                    for comp in comps:
                        results.append(
                            {
                                "category": category,
                                "prompt": row["prompt"],
                                "patch_kind": patch_kind,
                                "sub_word": sub,
                                "component": comp.label,
                                "component_kind": comp.kind,
                                "layer": comp.layer,
                                "head": comp.index,
                                "method": method,
                                "baseline_logit_diff": null_ref_ld,
                                "delta_logit_diff": deltas[comp],
                            }
                        )

            if method == "attribution":
                del null_acts, null_grads
            gc.collect()
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    filename = (
        f"patching_{method}_{model_str}"
        f"{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(results_df, out_dest_path, filename, in_place=False)
    print(f"Wrote {len(results_df)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path
