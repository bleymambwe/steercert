"""Converters between steercert and other interpretability and steering libraries.

Each submodule imports its target library lazily, so ``import steercert``
never requires them. Install the extras you need, for example
``pip install steercert[steering-vectors]``.

Layer convention: every library here indexes by *decoder block output*, which
is :meth:`steercert.hf.Site.block`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..hf import Site

__all__ = ["as_site_vectors"]


def as_site_vectors(vectors: Mapping[Site | int, np.ndarray]) -> dict[Site, np.ndarray]:
    """Normalise ``{layer_index or Site: vector}`` into ``{Site: float64 array}``."""
    result: dict[Site, np.ndarray] = {}
    for key, value in vectors.items():
        site = key if isinstance(key, Site) else Site.block(int(key))
        array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        result[site] = np.asarray(array, dtype=np.float64).ravel()
    return result
