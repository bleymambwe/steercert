"""Collect certificate gradients through `nnsight <https://nnsight.net>`_.

With ``remote=True`` the same code runs on the National Deep Inference Fabric
(NDIF), which serves open models up to hundreds of billions of parameters. That
makes certificates on 8B-405B models possible without a local GPU.

Status: experimental. Verified locally with nnsight 0.7 on small models,
where gradients and states match :func:`steercert.hf.collect` to 1e-6; remote
execution depends on NDIF's deployed models and its policy for
gradients and custom code, which can change. Validate a remote result against
a local run on a small model first.

Example
-------
>>> from nnsight import LanguageModel   # TransformersModel on nnsight >= 0.8
>>> from steercert.hf import token_ids, token_logit_difference
>>> from steercert.integrations.nnsight import collect_nnsight
>>> model = LanguageModel("meta-llama/Llama-3.1-8B")
>>> phi = token_logit_difference(token_ids(model.tokenizer, ["hate"]), token_ids(model.tokenizer, ["love"]))
>>> data = collect_nnsight(model, prompts, layers=[8, 16, 24], functional=phi, remote=True)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..hf import Collected, Functional, Site, _batches, find_blocks

__all__ = ["collect_nnsight"]


def _block_path(model: Any) -> list[str]:
    torch_model = getattr(model, "_model", model)
    target = find_blocks(torch_model)
    for name, module in torch_model.named_modules():
        if module is target:
            return name.split(".")
    raise LookupError("cannot locate the decoder block list inside the nnsight model")


def _envoy(model: Any, path: list[str]) -> Any:
    node = model
    for attr in path:
        node = getattr(node, attr)
    return node


def collect_nnsight(
    model: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    functional: Functional,
    *,
    remote: bool = False,
    batch_size: int = 8,
    max_length: int = 128,
) -> Collected:
    """Last-token states and per-example gradients at the outputs of ``layers``.

    ``model`` is an ``nnsight.LanguageModel`` (``TransformersModel`` on
    nnsight >= 0.8) or any ``nnsight.NNsight`` wrapper with a ``tokenizer``.
    Results are keyed by :meth:`steercert.hf.Site.block` and feed straight
    into :func:`steercert.audit`.
    """
    import torch

    tokenizer = model.tokenizer
    path = _block_path(model)
    blocks = _envoy(model, path)
    order = sorted(set(int(k) for k in layers))
    states: dict[Site, list[np.ndarray]] = {Site.block(k): [] for k in order}
    grads: dict[Site, list[np.ndarray]] = {Site.block(k): [] for k in order}
    values: list[np.ndarray] = []

    for batch in _batches(prompts, batch_size):
        side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )
            last = encoded["attention_mask"].sum(dim=1) - 1
            # nnsight runs the trace body in a worker; results are written into
            # this outer container so they survive the trace on every version.
            box: dict[str, Any] = {}
            with model.trace(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], remote=remote
            ):
                hidden = {}
                for k in order:
                    output = blocks[k].output
                    tensor = output[0] if isinstance(output, tuple) else output
                    tensor.requires_grad_(True)
                    hidden[k] = tensor
                    box[f"state{k}"] = tensor.detach().save()
                logits = model.output.logits
                phi = functional(logits, last.to(logits.device))
                box["phi"] = phi.detach().save()
                with phi.sum().backward():
                    for k in reversed(order):
                        box[f"grad{k}"] = hidden[k].grad.save()
        finally:
            tokenizer.padding_side = side
        rows = torch.arange(len(batch))
        for k in order:
            site = Site.block(k)
            state = _value(box[f"state{k}"]).double().cpu()
            grad = _value(box[f"grad{k}"]).double().cpu()
            states[site].append(state[rows, last].numpy())
            grads[site].append(grad[rows, last].numpy())
        values.append(_value(box["phi"]).double().cpu().numpy())

    return Collected(
        states={s: np.concatenate(v).astype(np.float64) for s, v in states.items()},
        gradients={s: np.concatenate(v).astype(np.float64) for s, v in grads.items()},
        values=np.concatenate(values).astype(np.float64),
        positions="last",
    )


def _value(saved: Any) -> Any:
    return getattr(saved, "value", saved)
