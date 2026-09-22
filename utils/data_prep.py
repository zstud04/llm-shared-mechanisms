"""Rebuild the organized human-data tree from the raw collection exports.

Run with:  python -m utils.data_prep [--raw-dir data_raw] [--out-dir data/human]

Three evaluations, each getting a long file (one row per subject x item) and a
wide file (one row per item, with means and SDs across subjects):

    base     the original behavioral evaluation (rounds 1 and 2)
    retest   the re-test / justification session on the base stimuli
    minimal  the minimal-difference evaluation

Every item carries a deterministic `prompt_uuid` (uuid5 over the normalized
prompt text), so ids are stable across reruns, joinable to the stimulus files
without a fragile string match, and identical for the same prompt in different
evaluations — which is what makes base-vs-retest comparisons a clean join.
"""

from __future__ import annotations

import argparse
import os
import re
import uuid

import pandas as pd

# Fixed namespace: regenerating the tree must reproduce identical ids.
UUID_NAMESPACE = uuid.UUID("6f3d5f4e-9a1b-4c8d-b2e7-1d0c9a5b3e77")

RAW_DIR_DEFAULT = "data_raw"
OUT_DIR_DEFAULT = os.path.join("data", "human")

# Curly quotes appear in some exports but not others; normalize before hashing.
_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def normalize_prompt(text) -> str:
    s = str(text)
    for src, dst in _QUOTES.items():
        s = s.replace(src, dst)
    return re.sub(r"\s+", " ", s).strip()


def prompt_uuid(prompt: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, normalize_prompt(prompt)))


def add_uuid(df: pd.DataFrame, eval_name: str, prompt_col: str = "prompt") -> pd.DataFrame:
    out = df.copy()
    out["prompt_norm"] = out[prompt_col].map(normalize_prompt)
    out["prompt_uuid"] = out["prompt_norm"].map(prompt_uuid)
    out["eval"] = eval_name
    return out


def aggregate_wide(
    long_df: pd.DataFrame,
    correct_col: str,
    measure_cols: tuple = (),
    id_cols: tuple = (),
) -> pd.DataFrame:
    """One row per item: mean and SD across subjects for accuracy and each measure.

    Grouping is on `prompt_uuid` alone so an item stays one row; `id_cols` are
    item-level attributes carried through unchanged.
    """
    keys = ["prompt_uuid", "prompt_norm"]
    grouped = long_df.groupby(keys, dropna=False)

    wide = grouped.agg(
        accuracy=(correct_col, "mean"),
        accuracy_sd=(correct_col, "std"),
        n_subjects=(correct_col, "count"),
    ).reset_index()

    for col in measure_cols:
        if col not in long_df.columns:
            continue
        stats = grouped[col].agg(["mean", "std"]).reset_index()
        stats = stats.rename(columns={"mean": f"{col}_mean", "std": f"{col}_sd"})
        wide = wide.merge(stats, on=keys, how="left")

    carried = [c for c in id_cols if c in long_df.columns]
    if carried:
        attrs = grouped[carried].first().reset_index()
        wide = wide.merge(attrs, on=keys, how="left")

    return wide.rename(columns={"prompt_norm": "prompt"})


# ------------------------------------------------------------------ per-eval builds


def build_base(raw: str) -> tuple:
    """Base evaluation: subject-level rounds 1 and 2, pooled."""
    round_1 = pd.read_csv(os.path.join(raw, "subj_data_long_round_1_behavioral.csv"))
    round_2 = pd.read_csv(os.path.join(raw, "subj_data_long_round_2_behavioral.csv"))
    round_1["round"] = 1
    round_2["round"] = 2

    keep = [
        "subj_code", "round", "bin", "type", "prompt", "response",
        "category", "category_type", "correct_res", "incorrect_res",
    ]
    long_df = pd.concat(
        [df[[c for c in keep if c in df.columns]] for df in (round_1, round_2)],
        ignore_index=True,
    )
    long_df = add_uuid(long_df, "base")
    long_df = long_df.rename(columns={"response": "is_correct"})

    wide = aggregate_wide(
        long_df,
        correct_col="is_correct",
        id_cols=("category", "category_type", "type", "correct_res", "incorrect_res"),
    )
    # Per-round Ns make it visible which items only ran in one round.
    per_round = (
        long_df.pivot_table(
            index="prompt_uuid", columns="round", values="is_correct", aggfunc="count"
        )
        .rename(columns=lambda r: f"n_obs_round_{r}")
        .reset_index()
    )
    wide = wide.merge(per_round, on="prompt_uuid", how="left")

    long_out = long_df.drop(columns=["prompt"]).rename(columns={"prompt_norm": "prompt"})
    return long_out, wide


def build_retest(raw: str, base_prompts: set) -> tuple:
    """Re-test / justification session, restricted to base-evaluation items."""
    df = pd.read_csv(os.path.join(raw, "final_retest_data.csv"))
    df = add_uuid(df, "retest")
    df = df[df["prompt_norm"].isin(base_prompts)].copy()

    keep = [
        "subj_code", "eval", "session", "prompt_norm", "prompt_uuid", "res_A", "res_B",
        "left_response", "right_response", "final_answer", "is_correct",
        "space_rt", "choice_rt", "justification", "prompt_type", "category",
    ]
    long_df = df[[c for c in keep if c in df.columns]].copy()

    wide = aggregate_wide(
        long_df,
        correct_col="is_correct",
        measure_cols=("space_rt", "choice_rt"),
        id_cols=("category", "prompt_type"),
    )
    long_out = long_df.rename(columns={"prompt_norm": "prompt"})
    return long_out, wide


def build_minimal(raw: str, minimal_prompts: set) -> tuple:
    """Minimal-difference evaluation.

    The raw export mixes minimal-difference items with re-test items in one
    file, so it is filtered against the minimal-difference stimulus set.
    """
    df = pd.read_csv(os.path.join(raw, "minimal_diff_human_long.csv"))
    df = add_uuid(df, "minimal")
    df = df[df["prompt_norm"].isin(minimal_prompts)].copy()

    keep = [
        "subj_code", "eval", "prompt_norm", "prompt_uuid", "res_A", "res_B",
        "left_response", "right_response", "final_answer", "is_correct",
        "space_rt", "choice_rt", "justification",
    ]
    long_df = df[[c for c in keep if c in df.columns]].copy()

    wide = aggregate_wide(
        long_df, correct_col="is_correct", measure_cols=("space_rt", "choice_rt")
    )
    long_out = long_df.rename(columns={"prompt_norm": "prompt"})
    return long_out, wide


def attach_stimulus_metadata(
    wide: pd.DataFrame, stim_path: str, eval_name: str, cols: tuple
) -> pd.DataFrame:
    """Join stimulus-level fields (options, category, bin) onto a wide table.

    Columns already present in `wide` under any casing are skipped, so the
    subject-derived `category` is not shadowed by the stimulus `Category`.
    """
    stim = add_uuid(pd.read_csv(stim_path), eval_name)
    existing = {c.lower() for c in wide.columns}
    have = [c for c in cols if c in stim.columns and c.lower() not in existing]
    merged = wide.merge(
        stim[["prompt_uuid"] + have].drop_duplicates("prompt_uuid"),
        on="prompt_uuid",
        how="left",
        suffixes=("", "_stim"),
    )
    return merged


def write_stimulus_uuids(stim_path: str) -> None:
    """Add `prompt_uuid` to a stimulus CSV so it joins to the human tables."""
    stim = pd.read_csv(stim_path)
    stim["prompt_uuid"] = stim["prompt"].map(prompt_uuid)
    stim.to_csv(stim_path, index=False)
    print(f"  stimuli: {stim_path} (+prompt_uuid, {len(stim)} rows)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=RAW_DIR_DEFAULT, help="raw export directory")
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="organized output root")
    parser.add_argument("--stimuli-dir", default="stimuli")
    args = parser.parse_args(argv)

    base_stim = os.path.join(args.stimuli_dir, "general_eval.csv")
    minimal_stim = os.path.join(args.stimuli_dir, "minimal_diff_eval.csv")

    base_prompts = set(pd.read_csv(base_stim)["prompt"].map(normalize_prompt))
    minimal_prompts = set(pd.read_csv(minimal_stim)["prompt"].map(normalize_prompt))

    builders = {
        "base": lambda: build_base(args.raw_dir),
        "retest": lambda: build_retest(args.raw_dir, base_prompts),
        "minimal": lambda: build_minimal(args.raw_dir, minimal_prompts),
    }

    for name, build in builders.items():
        long_df, wide_df = build()
        if name in ("base", "retest"):
            wide_df = attach_stimulus_metadata(
                wide_df, base_stim, name if name == "base" else "retest",
                ("a", "b", "Category", "bin", "prompt_format"),
            )
        else:
            wide_df = attach_stimulus_metadata(
                wide_df, minimal_stim, "minimal", ("a", "b", "Category", "critical_trial")
            )

        out_dir = os.path.join(args.out_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        long_path = os.path.join(out_dir, f"{name}_long.csv")
        wide_path = os.path.join(out_dir, f"{name}_wide.csv")
        long_df.to_csv(long_path, index=False)
        wide_df.to_csv(wide_path, index=False)
        print(
            f"{name}: long={len(long_df)} rows -> {long_path}; "
            f"wide={len(wide_df)} items -> {wide_path}"
        )

    write_stimulus_uuids(base_stim)
    write_stimulus_uuids(minimal_stim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
