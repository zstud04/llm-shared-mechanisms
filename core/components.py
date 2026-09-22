"""Model components that interpretability methods can target.

Previously every interp method was hard-coded to attention heads (`hook_z`).
A `Component` names *any* intervenable site, so ablation / patching / activity
collection all share one vocabulary:

    attn_head   per-head attention output (z), indexed by head
    attn_out    whole attention block output for a layer
    mlp_pre     MLP pre-activation (before the nonlinearity)
    mlp_post    MLP post-activation (after the nonlinearity)
    mlp_out     whole MLP block output for a layer
    resid_post  residual stream after the layer

Only `attn_head` is head-indexed; the rest are whole-layer sites and carry
`index=None`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

# Kinds that are indexed by head. Everything else is whole-layer.
HEAD_INDEXED_KINDS = frozenset({"attn_head"})

# TransformerLens hook names per kind.
TL_HOOK_NAMES = {
    "attn_head": "blocks.{layer}.attn.hook_z",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_pre": "blocks.{layer}.mlp.hook_pre",
    "mlp_post": "blocks.{layer}.mlp.hook_post",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "resid_post": "blocks.{layer}.hook_resid_post",
    # Attention pattern is read-only (used by activity measures, not ablation).
    "attn_pattern": "blocks.{layer}.attn.hook_pattern",
    "attn_scores": "blocks.{layer}.attn.hook_attn_scores",
}

# Kinds that can be ablated/patched. `attn_pattern`/`attn_scores` are
# diagnostic reads only.
WRITABLE_KINDS = frozenset(
    {"attn_head", "attn_out", "mlp_pre", "mlp_post", "mlp_out", "resid_post"}
)

ALL_KINDS = frozenset(TL_HOOK_NAMES)


@dataclass(frozen=True, order=True)
class Component:
    """A single intervenable site in the model."""

    kind: str
    layer: int
    index: Optional[int] = None  # head index; None for whole-layer kinds

    def __post_init__(self):
        if self.kind not in ALL_KINDS:
            raise ValueError(
                f"Unknown component kind {self.kind!r}; expected one of {sorted(ALL_KINDS)}"
            )
        if self.kind in HEAD_INDEXED_KINDS and self.index is None:
            raise ValueError(f"{self.kind} requires a head index (got None)")
        if self.kind not in HEAD_INDEXED_KINDS and self.index is not None:
            raise ValueError(f"{self.kind} is a whole-layer site and takes no index")

    @property
    def tl_hook(self) -> str:
        return TL_HOOK_NAMES[self.kind].format(layer=self.layer)

    @property
    def is_writable(self) -> bool:
        return self.kind in WRITABLE_KINDS

    @property
    def label(self) -> str:
        """Short string used in CSV column names and result rows."""
        if self.index is not None:
            return f"L{self.layer}H{self.index}"
        return f"L{self.layer}.{self.kind}"

    def __str__(self) -> str:
        return self.label


def parse_component(spec: str, default_kind: str = "attn_head") -> Component:
    """Parse a component from a string.

    Accepted forms:
        "L38H12"            -> attn_head, layer 38, head 12
        "38,12"             -> attn_head, layer 38, head 12
        "L20.mlp_post"      -> mlp_post, layer 20
        "mlp_post:20"       -> mlp_post, layer 20
        "20"                -> `default_kind` at layer 20 (whole-layer kinds only)
    """
    s = spec.strip()
    if not s:
        raise ValueError("empty component spec")

    # "kind:layer" / "kind:layer:index"
    if ":" in s:
        parts = s.split(":")
        kind = parts[0].strip()
        layer = int(parts[1])
        index = int(parts[2]) if len(parts) > 2 else None
        return Component(kind, layer, index)

    # "L20.mlp_post"
    if "." in s and s[0].upper() == "L":
        head_part, kind = s.split(".", 1)
        return Component(kind.strip(), int(head_part[1:]))

    # "L38H12"
    up = s.upper()
    if up.startswith("L") and "H" in up:
        layer_str, head_str = up[1:].split("H", 1)
        return Component("attn_head", int(layer_str), int(head_str))

    # "38,12"
    if "," in s:
        layer_str, head_str = s.split(",", 1)
        return Component("attn_head", int(layer_str), int(head_str))

    # bare layer number
    if default_kind in HEAD_INDEXED_KINDS:
        raise ValueError(f"cannot parse {spec!r} as {default_kind} (no head index)")
    return Component(default_kind, int(s))


def parse_components(spec, default_kind: str = "attn_head") -> list[Component]:
    """Parse a list of components from a string, list, or Component iterable.

    A string is split on ';' or whitespace so shell args stay quotable:
        "L38H12;L20.mlp_post"
    """
    if spec is None:
        return []
    if isinstance(spec, Component):
        return [spec]
    if isinstance(spec, str):
        raw = [p for p in spec.replace(";", " ").split() if p]
        return [parse_component(p, default_kind) for p in raw]
    out: list[Component] = []
    for item in spec:
        out.extend(parse_components(item, default_kind))
    return out


def enumerate_components(
    kind: str,
    n_layers: int,
    n_heads: int,
    layers: Optional[Iterable[int]] = None,
) -> Iterator[Component]:
    """Yield every component of `kind` across the requested layers.

    Used by the sweep-style ablations that score the full grid.
    """
    layer_range = range(n_layers) if layers is None else layers
    for layer in layer_range:
        if kind in HEAD_INDEXED_KINDS:
            for head in range(n_heads):
                yield Component(kind, layer, head)
        else:
            yield Component(kind, layer)
