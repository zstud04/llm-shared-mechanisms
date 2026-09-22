"""Result writing: modify a CSV in place, or write out a new one.

Every eval/interp entry point takes an `in_place` flag. When set, new columns
are merged back into the CSV the run read from; otherwise a fresh CSV is
written under the output destination. Nothing overwrites an existing column,
so re-running a model never silently clobbers an earlier result.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def resolve_out_path(out_dest: str, filename: str) -> str:
    """Treat `out_dest` as a directory when it has no `.csv` suffix."""
    if out_dest.lower().endswith(".csv"):
        return out_dest
    return os.path.join(out_dest, filename)


def merge_into(df: pd.DataFrame, target_path: str, overwrite: bool = False) -> None:
    """Add `df`'s columns to the CSV at `target_path`, aligning by row order."""
    existing = pd.read_csv(target_path)
    if len(existing) != len(df):
        raise ValueError(
            f"Row count mismatch merging into {target_path}: "
            f"existing={len(existing)}, new={len(df)}"
        )
    for col in df.columns:
        if col not in existing.columns or overwrite:
            existing[col] = df[col].values
    existing.to_csv(target_path, index=False)


def write_output(
    df: pd.DataFrame,
    out_dest: str,
    filename: str,
    in_place: bool = False,
    source_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Persist `df` and return the path written.

    in_place=True  -> merge new columns into `source_path` (required)
    in_place=False -> write to `out_dest` (a directory or an explicit .csv path)
    """
    if in_place:
        if not source_path:
            raise ValueError("in_place=True requires source_path")
        merge_into(df, source_path, overwrite=overwrite)
        return source_path

    path = resolve_out_path(out_dest, filename)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.exists(path) and not overwrite:
        merge_into(df, path, overwrite=False)
    else:
        df.to_csv(path, index=False)
    return path


def as_bool(value) -> bool:
    """Parse a CLI-style boolean ('true', '1', 'yes', ...)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "t", "yes", "y")
