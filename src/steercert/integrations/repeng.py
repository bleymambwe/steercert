"""Interoperate with `repeng <https://github.com/vgel/repeng>`_ control vectors.

``ControlVector.directions`` maps a layer id to a vector that ``ControlModel``
adds to the output of decoder block ``layer_id`` at every position. repeng
reads its training activations from ``hidden_states[layer_id + 1]``, which is
the same tensor except for the last block, where Hugging Face applies the
final norm first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..hf import Site
from . import as_site_vectors

__all__ = ["from_control_vector", "to_control_vector"]


def from_control_vector(control_vector: Any) -> dict[Site, np.ndarray]:
    """Extract ``{Site.block(k): vector}`` from a ``repeng.ControlVector``."""
    return as_site_vectors(control_vector.directions)


def to_control_vector(vectors: Mapping[Site | int, np.ndarray], model_type: str) -> Any:
    """Build a ``repeng.ControlVector`` from block-output directions."""
    from repeng import ControlVector

    directions = {}
    for site, vector in as_site_vectors(vectors).items():
        if site.kind != "block":
            raise ValueError(f"repeng only controls block outputs, got {site}")
        directions[site.index] = vector.astype(np.float32)
    return ControlVector(model_type=model_type, directions=directions)
