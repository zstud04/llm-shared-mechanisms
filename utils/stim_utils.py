"""Stimulus loading, prompt formatting, and category subsetting."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

CONFIG_DIR = os.getenv("WORLD_MODELS_CONFIG", "config")

with open(os.path.join(CONFIG_DIR, "instructions.json"), "r", encoding="utf-8") as f:
    INSTRUCTIONS_CFG = json.load(f)

random.seed(42)


def get_instruction(prompt_key: str) -> str:
    if prompt_key not in INSTRUCTIONS_CFG:
        raise KeyError(
            f"Unknown prompt_key {prompt_key!r}; known: {sorted(INSTRUCTIONS_CFG)}"
        )
    return INSTRUCTIONS_CFG[prompt_key]


def create_binary_choice_df(
    stim_df: pd.DataFrame,
    input_prompt: str,
    correct_opt_col: str,
    incorrect_opt_col: str,
    prompt_template: str,
    random_shuffle: bool = True,
) -> pd.DataFrame:
    """Add a `formatted_prompt` column with the instruction template applied.

    With `random_shuffle`, which option lands in slot A is randomized per row;
    interp runs pass False so option order is fixed across runs.
    """
    df = stim_df.copy()

    def _format_row(row):
        if random_shuffle and random.random() < 0.5:
            a, b = row[incorrect_opt_col], row[correct_opt_col]
        else:
            a, b = row[correct_opt_col], row[incorrect_opt_col]
        return prompt_template.format(prompt=row[input_prompt], a=a, b=b)

    df["formatted_prompt"] = df.apply(_format_row, axis=1)
    return df


def load_formatted_stimuli(
    stim_csv_path: str,
    prompt_key: str,
    subset_categories=None,
    random_shuffle: bool = False,
) -> pd.DataFrame:
    """Read a stimulus CSV, format prompts, and apply a category subset."""
    df = create_binary_choice_df(
        pd.read_csv(stim_csv_path),
        "prompt",
        "a",
        "b",
        get_instruction(prompt_key),
        random_shuffle=random_shuffle,
    )
    return apply_category_subset(df, subset_categories)


def apply_category_subset(df: pd.DataFrame, subset_categories=None) -> pd.DataFrame:
    subset_categories = normalize_subset_categories(subset_categories)
    if subset_categories is None:
        return df.copy()

    if "Category" not in df.columns:
        raise ValueError("Input dataframe does not contain a 'Category' column")

    missing = sorted(set(subset_categories) - set(df["Category"].dropna().unique()))
    if missing:
        raise ValueError(f"Requested categories not found in dataframe: {missing}")

    out_df = df[df["Category"].isin(subset_categories)].copy()
    out_df["subset_categories"] = ",".join(subset_categories)
    return out_df


def normalize_subset_categories(subset_categories) -> Optional[list]:
    """Accept None, a comma-separated string, or a collection; return a list or None."""
    if subset_categories is None:
        return None

    if isinstance(subset_categories, str):
        parts = [x.strip() for x in subset_categories.split(",") if x.strip()]
    elif isinstance(subset_categories, (list, tuple, set)):
        parts = [str(x).strip() for x in subset_categories if str(x).strip()]
    else:
        raise ValueError(
            "subset_categories must be None, a comma-separated string, or a collection"
        )
    return parts or None


def category_suffix(subset_categories) -> str:
    """Filename suffix identifying a category subset (empty when unsubsetted)."""
    normalized = normalize_subset_categories(subset_categories)
    return f"_{'_'.join(normalized)}" if normalized else ""


def add_trigram_freq(stim_csv_path: str, prompt_key: str = "prompt") -> None:
    """Write a `trigram_freq` column (mean Brown-corpus word-trigram count)."""
    import nltk
    from nltk.corpus import brown
    from nltk.util import ngrams

    try:
        brown.words()
    except LookupError:
        nltk.download("brown", quiet=True)

    # Cache the reference counts on the function; building them is slow.
    if not hasattr(add_trigram_freq, "_trigram_counts"):
        ref_tokens = [w.lower() for w in brown.words() if re.search(r"\w", w)]
        add_trigram_freq._trigram_counts = Counter(ngrams(ref_tokens, 3))
    trigram_counts = add_trigram_freq._trigram_counts

    def _mean_trigram_freq(text: str):
        toks = re.findall(r"[A-Za-z']+", str(text).lower())
        if len(toks) < 3:
            return np.nan
        return float(np.mean([trigram_counts.get(t, 0) for t in ngrams(toks, 3)]))

    df = pd.read_csv(stim_csv_path)
    if prompt_key not in df.columns:
        raise KeyError(
            f"Column '{prompt_key}' not found in {stim_csv_path}. Have: {list(df.columns)}"
        )
    df["trigram_freq"] = df[prompt_key].apply(_mean_trigram_freq)
    df.to_csv(stim_csv_path, index=False)
