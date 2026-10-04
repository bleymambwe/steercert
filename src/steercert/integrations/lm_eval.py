"""Export to and import from lm-evaluation-harness steering configs.

lm-eval-harness evaluates a steered model with::

    lm_eval run --model steered \\
        --model_args pretrained=EleutherAI/pythia-160m,steer_path=steer.pt \\
        --tasks hellaswag

``steer.pt`` maps a hookpoint (a module name relative to
``model.base_model``) to a dict with ``steering_vector``,
``steering_coefficient``, ``action``, ``bias`` and ``head_index``. With
``action="add"`` the vector is added to that module's output at every
position, so the matching certificate uses ``positions="all"``.

Typical use: pick directions with :func:`steercert.audit` (for example a
dark direction or the Gramian-top direction), export them here, and measure
their capability cost on standard benchmarks.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import unit
from ..hf import Site, module_name
from . import as_site_vectors

__all__ = ["load_steer_config", "save_steer_config", "to_steer_config"]


def _base(model: Any) -> Any:
    return getattr(model, "base_model", model)


def to_steer_config(
    model: Any,
    vectors: Mapping[Site | int, np.ndarray],
    coefficient: float | Mapping[Site | int, float] = 1.0,
    normalise: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build an lm-eval ``steer_path`` dictionary for ``model``.

    With ``normalise=True`` each vector is scaled to unit norm, so
    ``coefficient`` is the step size in activation units. Use
    :func:`steercert.hf.residual_norm_median` to choose a comparable value.
    """
    import torch

    config: dict[str, dict[str, Any]] = {}
    per_site: dict[Site, float] | None = None
    if isinstance(coefficient, Mapping):
        per_site = {
            (k if isinstance(k, Site) else Site.block(int(k))): float(v) for k, v in coefficient.items()
        }
    for site, vector in as_site_vectors(vectors).items():
        hookpoint = module_name(model, site, root=_base(model))
        value = unit(vector) if normalise else vector
        coef = per_site[site] if per_site is not None else float(coefficient)  # type: ignore[arg-type]
        config[hookpoint] = {
            "steering_vector": torch.as_tensor(value, dtype=torch.float32),
            "steering_coefficient": coef,
            "action": "add",
            "bias": None,
            "head_index": None,
        }
    return config


def save_steer_config(
    path: str | Path,
    model: Any,
    vectors: Mapping[Site | int, np.ndarray],
    coefficient: float | Mapping[Site | int, float] = 1.0,
    normalise: bool = True,
) -> Path:
    """Write :func:`to_steer_config` to ``path`` with ``torch.save``."""
    import torch

    path = Path(path)
    torch.save(to_steer_config(model, vectors, coefficient, normalise), path)
    return path


def load_steer_config(path: str | Path, model: Any) -> dict[Site, tuple[np.ndarray, float, str]]:
    """Read a ``.pt`` steer config and map hookpoints back to sites.

    Returns ``{site: (vector, coefficient, action)}`` so existing configs,
    including ones derived from SAE features, can be audited.
    """
    import torch

    from ..hf import find_blocks, find_final_norm

    raw = torch.load(Path(path), weights_only=False)
    by_module = {id(block): Site.block(i) for i, block in enumerate(find_blocks(model))}
    by_module[id(find_final_norm(model))] = Site.final_norm()
    modules = dict(_base(model).named_modules())
    result = {}
    for hookpoint, info in raw.items():
        module = modules.get(hookpoint)
        if module is None or id(module) not in by_module:
            raise ValueError(f"hookpoint {hookpoint!r} is not a block output or final norm of this model")
        if info.get("head_index") is not None:
            raise ValueError(f"head-specific steering at {hookpoint!r} is not supported")
        vector = info["steering_vector"].detach().float().cpu().numpy().astype(np.float64)
        result[by_module[id(module)]] = (
            vector,
            float(info.get("steering_coefficient", 1.0)),
            str(info.get("action", "add")),
        )
    return result
