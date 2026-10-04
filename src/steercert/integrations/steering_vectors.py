"""Interoperate with `steering-vectors <https://github.com/steering-vectors/steering-vectors>`_.

``SteeringVector.layer_activations`` maps a decoder-block index to a vector
that is added to that block's output at every selected token, so audit it
with ``positions="all"`` unless you restrict ``token_indices`` to the last
token.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..hf import Site
from . import as_site_vectors

__all__ = ["from_steering_vector", "to_steering_vector"]


def from_steering_vector(steering_vector: Any) -> dict[Site, np.ndarray]:
    """Extract ``{Site.block(k): vector}`` from a ``steering_vectors.SteeringVector``."""
    layer_type = getattr(steering_vector, "layer_type", "decoder_block")
    if layer_type != "decoder_block":
        raise ValueError(
            f"only decoder_block steering vectors map onto block-output sites, got {layer_type!r}"
        )
    return as_site_vectors(steering_vector.layer_activations)


def to_steering_vector(vectors: Mapping[Site | int, np.ndarray], dtype: Any = None) -> Any:
    """Build a ``SteeringVector`` from block-output directions, e.g. a dark direction."""
    import torch
    from steering_vectors import SteeringVector

    activations = {}
    for site, vector in as_site_vectors(vectors).items():
        if site.kind != "block":
            raise ValueError(f"steering-vectors only patches block outputs, got {site}")
        tensor = torch.as_tensor(vector, dtype=torch.float32)
        activations[site.index] = tensor.to(dtype) if dtype is not None else tensor
    return SteeringVector(layer_activations=activations, layer_type="decoder_block")
