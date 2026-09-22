"""Behavioral evaluations.

Runs on any inference backend: HF, TransformerLens, vLLM, or a closed API.
Chain-of-thought is captured uniformly — providers that expose reasoning
separately (OpenAI, DeepSeek, Gemini) land in `reasoning_*`, and its length is
recorded so the reasoning-length analyses work for open and closed models
alike.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.model_utils as model_utils
import utils.stim_utils as stim_utils
import utils.str_utils as str_utils
from utils.io_utils import write_output

# Reasoning models need room for a full trace; non-reasoning local models only
# need to emit the answer phrase.
REASONING_MAX_TOKENS = 25000
ANSWER_MAX_TOKENS = 10


def general_eval(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    backend: Optional[str] = None,
    in_place: bool = False,
    subset_categories=None,
    batch_size: int = 32,
):
    """Generate an answer for every stimulus and score the two options.

    Writes per-model columns: `logit_a/logit_b/logit_diff/logit_entropy`,
    `correct`, `output`, `reasoning`, `reasoning_n_words`.
    """
    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=True
    )
    lm, cfg = model_utils.load_model(model_str, backend=backend, interp=False)

    has_reasoning = bool(cfg.get("reasoning", False))
    max_toks = REASONING_MAX_TOKENS if has_reasoning else ANSWER_MAX_TOKENS
    prompts = formatted_df["formatted_prompt"].tolist()

    # vLLM's whole advantage is batching; other backends run prompt by prompt.
    if hasattr(lm, "generate_batch"):
        generations = []
        for start in tqdm(range(0, len(prompts), batch_size), desc=f"{model_str} generate"):
            generations.extend(lm.generate_batch(prompts[start:start + batch_size], max_toks))
    else:
        generations = [
            lm.generate(p, max_toks)
            for p in tqdm(prompts, desc=f"{model_str} generate")
        ]

    out = formatted_df.copy()
    out[f"output_{model_str}"] = [g.text for g in generations]
    out[f"reasoning_{model_str}"] = [g.reasoning for g in generations]
    out[f"reasoning_n_words_{model_str}"] = [
        len(g.reasoning.split()) if g.reasoning else np.nan for g in generations
    ]
    out[f"correct_{model_str}"] = [
        str_utils.is_correct_match(g.text, row["a"], row["b"])
        for g, (_, row) in zip(generations, out.iterrows())
    ]

    if lm.supports_logits:
        scores = [
            lm.score_options(row["formatted_prompt"], row["a"], row["b"])
            for _, row in tqdm(out.iterrows(), total=len(out), desc=f"{model_str} score")
        ]
        out[f"logit_a_{model_str}"] = [s.logit_a for s in scores]
        out[f"logit_b_{model_str}"] = [s.logit_b for s in scores]
        out[f"logit_diff_{model_str}"] = [s.logit_diff for s in scores]
        out[f"logit_entropy_{model_str}"] = [s.entropy for s in scores]
        # vLLM reports log-probs rather than raw logits; the difference is
        # identical either way, but flag it so analyses know.
        out[f"logit_is_logprob_{model_str}"] = scores[0].is_logprob if scores else False
    else:
        for col in ("logit_a", "logit_b", "logit_diff", "logit_entropy"):
            out[f"{col}_{model_str}"] = np.nan

    filename = (
        f"behavioral_{model_str}{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(
        out, out_dest_path, filename, in_place=in_place, source_path=stim_csv_path
    )
    print(f"Wrote {len(out)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path


def get_surprisal(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    in_place: bool = False,
    subset_categories=None,
):
    """Per-option surprisal at the BLANK position.

    Masked models (BERT) score at a [MASK]; autoregressive models score the
    next token after the prompt is cropped at BLANK. Always runs on the HF
    backend, which is the only one exposing per-position distributions.
    """
    BLANK = "BLANK"

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=False
    )
    lm, cfg = model_utils.load_model(model_str, backend="hf", interp=False)
    is_masked = bool(cfg.get("masked", "bert" in model_str.lower()))

    out = formatted_df.copy()
    surprisals = []
    for _, row in tqdm(out.iterrows(), total=len(out), desc=f"{model_str} surprisal"):
        kwargs = {"mask_str": BLANK} if is_masked else {"crop_str": BLANK}
        surprisals.append(
            lm.get_surprisal(row["formatted_prompt"], row["a"], row["b"], **kwargs)
        )

    out[f"surprisal_a_{model_str}"] = [s[0] for s in surprisals]
    out[f"surprisal_b_{model_str}"] = [s[1] for s in surprisals]

    filename = f"surprisal_{model_str}{stim_utils.category_suffix(subset_categories)}.csv"
    path = write_output(
        out, out_dest_path, filename, in_place=in_place, source_path=stim_csv_path
    )
    print(f"Wrote {len(out)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path


def get_mean_surprisal(
    model_str: str,
    stim_csv_path: str,
    prompt_key: str,
    out_dest_path: str,
    in_place: bool = False,
    subset_categories=None,
):
    """Mean per-token surprisal of the whole prompt with each option filled in."""
    BLANK = "BLANK"

    formatted_df = stim_utils.load_formatted_stimuli(
        stim_csv_path, prompt_key, subset_categories, random_shuffle=False
    )
    lm, cfg = model_utils.load_model(model_str, backend="hf", interp=False)
    if cfg.get("masked", "bert" in model_str.lower()):
        raise ValueError(f"get_mean_surprisal is autoregressive-only; '{model_str}' is masked")

    out = formatted_df.copy()
    col_a, col_b = [], []
    for _, row in tqdm(out.iterrows(), total=len(out), desc=f"{model_str} mean surprisal"):
        prompt = row["formatted_prompt"]
        col_a.append(lm.get_mean_surprisal(prompt.replace(BLANK, str(row["a"]))))
        col_b.append(lm.get_mean_surprisal(prompt.replace(BLANK, str(row["b"]))))

    out[f"mean_surprisal_a_{model_str}"] = col_a
    out[f"mean_surprisal_b_{model_str}"] = col_b

    filename = (
        f"mean_surprisal_{model_str}{stim_utils.category_suffix(subset_categories)}.csv"
    )
    path = write_output(
        out, out_dest_path, filename, in_place=in_place, source_path=stim_csv_path
    )
    print(f"Wrote {len(out)} rows to {path}")
    model_utils.mem_cleanup(lm)
    return path
