"""Hugging Face ``transformers`` adapter.

Collects activations and per-example behavioural gradients at an intervention
site, and measures finite interventions at the same site. Both use the same
hook, so the gradient and the edit always refer to the same tensor.

Gradients are taken through an additive zero tensor inserted at the site
(``h -> h + delta`` with ``delta = 0`` requiring grad). This works with frozen
weights, half precision, and any architecture whose decoder blocks can be
located. Per-example gradients come from one backward pass per batch because
examples in a causal LM batch do not interact.

Requires ``torch`` and ``transformers`` (``pip install steercert[hf]``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .audit import EffectMeasurement
from .geometry import unit

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "Collected",
    "Site",
    "collect",
    "directional_derivatives",
    "find_blocks",
    "find_final_norm",
    "measure_effects",
    "module_name",
    "residual_norm_median",
    "token_ids",
    "token_log_probability",
    "token_logit_difference",
]

Functional = Callable[["torch.Tensor", "torch.Tensor"], "torch.Tensor"]


# --------------------------------------------------------------------------- sites


@dataclass(frozen=True)
class Site:
    """Where in the network a direction is read or written.

    ``Site.block(k)`` is the output of decoder block ``k``. This is the
    convention used by steering-vectors, repeng and lm-eval-harness.
    ``Site.embeddings()`` is the input to block 0. ``Site.final_norm()`` is
    the output of the final normalisation, which Hugging Face returns as the
    last element of ``hidden_states``.
    """

    kind: str
    index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("embeddings", "block", "final_norm"):
            raise ValueError(f"unknown site kind {self.kind!r}")
        if (self.kind == "block") != (self.index is not None):
            raise ValueError("block sites need an index; other sites must not have one")

    @classmethod
    def block(cls, index: int) -> Site:
        return cls("block", int(index))

    @classmethod
    def embeddings(cls) -> Site:
        return cls("embeddings")

    @classmethod
    def final_norm(cls) -> Site:
        return cls("final_norm")

    @classmethod
    def from_hidden_state_index(cls, index: int, n_blocks: int) -> Site:
        """Map an index into Hugging Face ``output_hidden_states`` to a site.

        ``0`` is the embeddings, ``1..n_blocks-1`` are block outputs, and
        ``n_blocks`` is the final-norm output. Negative indices count back.
        """
        if index < 0:
            index += n_blocks + 1
        if not 0 <= index <= n_blocks:
            raise IndexError(f"hidden-state index {index} outside 0..{n_blocks}")
        if index == 0:
            return cls.embeddings()
        if index == n_blocks:
            return cls.final_norm()
        return cls.block(index - 1)

    @classmethod
    def parse(cls, text: str) -> Site:
        """Parse ``"embeddings"``, ``"final_norm"``, ``"block.3"`` or ``"block:3"``."""
        text = text.strip()
        if text in ("embeddings", "final_norm"):
            return cls(text)
        for sep in (".", ":"):
            if text.startswith(f"block{sep}"):
                return cls.block(int(text.split(sep, 1)[1]))
        raise ValueError(f"cannot parse site {text!r}")

    def __str__(self) -> str:
        return f"block.{self.index}" if self.kind == "block" else self.kind


_BLOCK_PATHS = (
    ("transformer", "h"),
    ("gpt_neox", "layers"),
    ("model", "layers"),
    ("model", "decoder", "layers"),
    ("transformer", "layers"),
    ("model", "language_model", "layers"),
)
_NORM_PATHS = (
    ("transformer", "ln_f"),
    ("gpt_neox", "final_layer_norm"),
    ("model", "norm"),
    ("model", "decoder", "final_layer_norm"),
    ("transformer", "norm_f"),
    ("model", "language_model", "norm"),
)


def _walk(root: Any, path: Sequence[str]) -> Any:
    node = root
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def find_blocks(model: Any) -> Any:
    """Return the ``ModuleList`` of decoder blocks of a causal LM."""
    for path in _BLOCK_PATHS:
        blocks = _walk(model, path)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks) > 0:
            return blocks
    import torch

    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    lists = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList)]
    if n_layers is not None:
        lists = [m for m in lists if len(m) == n_layers]
    if not lists:
        raise NotImplementedError(f"cannot locate decoder blocks of {type(model).__name__}")
    return max(lists, key=len)


def find_final_norm(model: Any) -> Any:
    """Return the final normalisation module of a causal LM."""
    for path in _NORM_PATHS:
        norm = _walk(model, path)
        if norm is not None:
            return norm
    raise NotImplementedError(f"cannot locate the final norm of {type(model).__name__}")


def _site_module(model: Any, site: Site) -> Any:
    if site.kind == "final_norm":
        return find_final_norm(model)
    blocks = find_blocks(model)
    index = 0 if site.index is None else site.index
    if not -len(blocks) <= index < len(blocks):
        raise IndexError(f"{site} outside the model's {len(blocks)} blocks")
    return blocks[index]


def module_name(model: Any, site: Site, root: Any | None = None) -> str:
    """Dotted name of the module whose output is ``site``, relative to ``root``.

    ``root`` defaults to ``model``. The embeddings site is the *input* of block
    0 and has no module whose output it is, so it raises ``ValueError``.
    """
    if site.kind == "embeddings":
        raise ValueError("the embeddings site is a block input; use Site.block(k) or Site.final_norm()")
    target = _site_module(model, site)
    for name, module in (root if root is not None else model).named_modules():
        if module is target:
            return name
    raise LookupError(f"{site} is not inside the given root module")


@contextmanager
def _edit(model: Any, site: Site, fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
    module = _site_module(model, site)
    if site.kind == "embeddings":

        def pre_hook(_module, args, kwargs):
            if args:
                return (fn(args[0]),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = fn(kwargs["hidden_states"])
            return args, kwargs

        handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    else:

        def post_hook(_module, _args, output):
            if isinstance(output, tuple):
                return (fn(output[0]),) + tuple(output[1:])
            return fn(output)

        handle = module.register_forward_hook(post_hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------- functionals


def token_ids(tokenizer: Any, words: Sequence[str], leading_space: bool = True) -> list[int]:
    """Ids of the words that encode to exactly one token; others are dropped."""
    ids = []
    for word in words:
        encoded = tokenizer.encode((" " + word) if leading_space else word, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    if not ids:
        raise ValueError("none of the words is a single token for this tokenizer")
    return ids


def token_logit_difference(target_ids: Sequence[int], reference_ids: Sequence[int]) -> Functional:
    """``logmeanexp(logits[target]) - logmeanexp(logits[reference])`` at the last token.

    A smooth next-token proxy for behaviours such as toxicity or sentiment.
    It is not a generation-level measure; validate any conclusion on
    generated text.
    """
    target, reference = list(target_ids), list(reference_ids)
    log_t, log_r = math.log(len(target)), math.log(len(reference))

    def functional(logits, last_index):
        import torch

        rows = torch.arange(logits.shape[0], device=logits.device)
        final = _upcast(logits[rows, last_index])
        return (
            torch.logsumexp(final[:, target], dim=-1)
            - log_t
            - torch.logsumexp(final[:, reference], dim=-1)
            + log_r
        )

    return functional


def token_log_probability(target_ids: Sequence[int]) -> Functional:
    """Log probability that the next token is in ``target_ids``."""
    target = list(target_ids)

    def functional(logits, last_index):
        import torch

        rows = torch.arange(logits.shape[0], device=logits.device)
        log_probs = torch.log_softmax(_upcast(logits[rows, last_index]), dim=-1)
        return torch.logsumexp(log_probs[:, target], dim=-1)

    return functional


def _upcast(tensor):
    """Promote half-precision tensors to float32; keep float32/float64 as they are."""
    import torch

    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


def _numpy(tensor) -> np.ndarray:
    import torch

    return tensor.detach().to(torch.float64).cpu().numpy()


# ----------------------------------------------------------------------- batching


def _device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _batches(prompts: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(prompts), size):
        yield list(prompts[start : start + size])


def _encode(tokenizer: Any, prompts: list[str], max_length: int, device: Any) -> dict:
    side, pad = tokenizer.padding_side, tokenizer.pad_token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
    finally:
        tokenizer.padding_side = side
        tokenizer.pad_token = pad
    encoded = {k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
    encoded["last_index"] = encoded["attention_mask"].sum(dim=1) - 1
    return encoded


def _forward(model: Any, encoded: dict) -> Any:
    return model(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False
    ).logits


def _check_positions(positions: str) -> None:
    if positions not in ("last", "all"):
        raise ValueError("positions must be 'last' or 'all'")


# ------------------------------------------------------------------------ collect


@dataclass
class Collected:
    """Activations, gradients and functional values gathered by :func:`collect`.

    With ``positions="last"`` each row is the last real token. With
    ``positions="all"`` states are mean-pooled over real tokens and gradients
    are *summed* over them, which is the exact derivative for a vector added
    at every position (as steering-vectors and lm-eval-harness do).
    """

    states: dict[Site, np.ndarray]
    gradients: dict[Site, np.ndarray] | None
    values: np.ndarray
    positions: str


def collect(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    sites: Site | Sequence[Site],
    functional: Functional | None = None,
    *,
    gradients: bool = True,
    positions: str = "last",
    batch_size: int = 16,
    max_length: int = 128,
) -> Collected:
    """Collect states and per-example gradients of ``functional`` at ``sites``.

    One forward and one backward pass per batch cover every site at once.
    """
    import torch

    _check_positions(positions)
    site_list = [sites] if isinstance(sites, Site) else list(sites)
    if gradients and functional is None:
        raise ValueError("a functional is required to compute gradients")
    device = _device(model)
    states: dict[Site, list[np.ndarray]] = {s: [] for s in site_list}
    grads: dict[Site, list[np.ndarray]] = {s: [] for s in site_list}
    values: list[np.ndarray] = []

    for batch in _batches(prompts, batch_size):
        encoded = _encode(tokenizer, batch, max_length, device)
        last = encoded["last_index"]
        mask = encoded["attention_mask"].unsqueeze(-1)
        captured: dict[Site, torch.Tensor] = {}
        deltas: dict[Site, torch.Tensor] = {}

        def make(site: Site):
            def fn(hidden):
                captured[site] = hidden
                if gradients:
                    delta = torch.zeros_like(hidden, requires_grad=True)
                    deltas[site] = delta
                    return hidden + delta
                return hidden

            return fn

        with ExitStack() as stack:
            for site in site_list:
                stack.enter_context(_edit(model, site, make(site)))
            with torch.enable_grad() if gradients else torch.no_grad():
                logits = _forward(model, encoded)
                phi = functional(logits, last) if functional is not None else None
                if gradients:
                    assert phi is not None
                    found = [deltas[s] for s in site_list]
                    computed = torch.autograd.grad(phi.sum(), found, allow_unused=True)
        rows = torch.arange(len(batch), device=device)
        for index, site in enumerate(site_list):
            hidden = captured[site].detach()
            if positions == "last":
                states[site].append(_numpy(hidden[rows, last]))
            else:
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                states[site].append(_numpy(pooled))
            if gradients:
                grad = computed[index]
                if grad is None:
                    grad = torch.zeros_like(hidden)
                if positions == "last":
                    grads[site].append(_numpy(grad[rows, last]))
                else:
                    grads[site].append(_numpy((grad * mask).sum(1)))
        if phi is not None:
            values.append(_numpy(phi))

    return Collected(
        states={s: np.concatenate(v).astype(np.float64) for s, v in states.items()},
        gradients=(
            {s: np.concatenate(v).astype(np.float64) for s, v in grads.items()} if gradients else None
        ),
        values=np.concatenate(values).astype(np.float64) if values else np.empty(0),
        positions=positions,
    )


def residual_norm_median(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    site: Site,
    *,
    positions: str = "last",
    batch_size: int = 16,
    max_length: int = 128,
) -> float:
    """Median activation norm at ``site``, the scale for comparable step sizes."""
    import torch

    _check_positions(positions)
    device = _device(model)
    norms: list[np.ndarray] = []
    for batch in _batches(prompts, batch_size):
        encoded = _encode(tokenizer, batch, max_length, device)
        captured: dict[str, torch.Tensor] = {}

        def fn(hidden):
            captured["h"] = hidden
            return hidden

        with _edit(model, site, fn), torch.no_grad():
            _forward(model, encoded)
        hidden = _upcast(captured["h"])
        if positions == "last":
            rows = torch.arange(len(batch), device=device)
            norms.append(hidden[rows, encoded["last_index"]].norm(dim=-1).cpu().numpy())
        else:
            keep = encoded["attention_mask"].bool()
            norms.append(hidden.norm(dim=-1)[keep].cpu().numpy())
    return float(np.median(np.concatenate(norms)))


# ------------------------------------------------------------------------ measure


def _as_named(
    directions: Mapping[str, np.ndarray] | tuple[list[str], np.ndarray],
) -> tuple[list[str], np.ndarray]:
    if isinstance(directions, tuple):
        names, matrix = directions
        return list(names), np.stack([unit(row) for row in np.asarray(matrix)])
    names = list(directions)
    return names, np.stack([unit(directions[name]) for name in names])


def measure_effects(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    site: Site,
    directions: Mapping[str, np.ndarray] | tuple[list[str], np.ndarray],
    functional: Functional,
    *,
    alpha: float = 0.03,
    alpha_mode: str = "residual-median",
    positions: str = "last",
    batch_size: int = 16,
    max_length: int = 128,
) -> EffectMeasurement:
    """Measure symmetric finite-difference effects of each direction.

    Parameters
    ----------
    directions:
        Mapping of names to vectors, or ``report.direction_matrix()``.
    alpha:
        Step size. With ``alpha_mode="residual-median"`` (default) it is a
        fraction of the median activation norm at the site, which keeps steps
        comparable across models of different width. ``"absolute"`` uses it
        as-is.
    positions:
        ``"last"`` edits the last real token; ``"all"`` edits every real token.

    Notes
    -----
    Use prompts disjoint from those used to fit probes and compute the
    certificate, so the ranking check in :func:`steercert.audit` is held out.
    """
    import torch

    _check_positions(positions)
    if alpha_mode not in ("residual-median", "absolute"):
        raise ValueError("alpha_mode must be 'residual-median' or 'absolute'")
    names, matrix = _as_named(directions)
    scale = None
    step = alpha
    if alpha_mode == "residual-median":
        scale = residual_norm_median(
            model,
            tokenizer,
            prompts,
            site,
            positions=positions,
            batch_size=batch_size,
            max_length=max_length,
        )
        step = alpha * scale
    device = _device(model)
    slopes = np.empty((len(prompts), len(names)))
    curvature = np.empty((len(prompts), len(names)))
    offset = 0
    for batch in _batches(prompts, batch_size):
        encoded = _encode(tokenizer, batch, max_length, device)
        last = encoded["last_index"]
        rows = torch.arange(len(batch), device=device)
        mask = encoded["attention_mask"].unsqueeze(-1)

        def run(vector: torch.Tensor | None) -> np.ndarray:
            def fn(hidden):
                if vector is None:
                    return hidden
                edited = hidden.clone()
                v = vector.to(device=hidden.device, dtype=hidden.dtype)
                if positions == "last":
                    edited[rows, last] = edited[rows, last] + v
                else:
                    edited = edited + mask.to(hidden.dtype) * v
                return edited

            with _edit(model, site, fn), torch.no_grad():
                return _numpy(functional(_forward(model, encoded), last))

        base = run(None)
        for column, direction in enumerate(matrix):
            vector = torch.as_tensor(direction * step, dtype=torch.float64)
            plus, minus = run(vector), run(-vector)
            slopes[offset : offset + len(batch), column] = (plus - minus) / (2.0 * step)
            curvature[offset : offset + len(batch), column] = (plus - 2.0 * base + minus) / (step * step)
        offset += len(batch)
    return EffectMeasurement(
        slopes={name: slopes[:, i] for i, name in enumerate(names)},
        alpha=float(step),
        alpha_mode=alpha_mode,
        residual_norm_median=scale,
        curvature={name: curvature[:, i] for i, name in enumerate(names)},
    )


def _math_attention():
    """Force the reference attention kernel, which supports double backward.

    Fused scaled-dot-product kernels (flash, and on CPU the default) do not
    implement the second derivative that Hessian-vector products need.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - torch < 2.3
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


def directional_derivatives(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    site: Site,
    directions: Mapping[str, np.ndarray] | tuple[list[str], np.ndarray],
    functional: Functional,
    *,
    batch_size: int = 8,
    max_length: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Exact first and second directional derivatives at the last token.

    Uses Hessian-vector products; no dense Hessian is formed. The second-order
    term bounds the step size over which the first-order certificate is
    trustworthy (see :func:`steercert.geometry.usable_radius`).
    """
    import torch

    names, matrix = _as_named(directions)
    device = _device(model)
    first: dict[str, list[np.ndarray]] = {name: [] for name in names}
    second: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for batch in _batches(prompts, batch_size):
        encoded = _encode(tokenizer, batch, max_length, device)
        last = encoded["last_index"]
        rows = torch.arange(len(batch), device=device)
        holder: dict[str, torch.Tensor] = {}

        def fn(hidden):
            delta = torch.zeros_like(hidden, requires_grad=True)
            holder["delta"] = delta
            return hidden + delta

        with _edit(model, site, fn), _math_attention(), torch.enable_grad():
            phi = functional(_forward(model, encoded), last)
            delta = holder["delta"]
            (grad,) = torch.autograd.grad(phi.sum(), delta, create_graph=True)
            selected = grad[rows, last]
            for index, direction in enumerate(matrix):
                v = torch.as_tensor(direction, device=device, dtype=selected.dtype)
                d1 = selected @ v
                (hvp,) = torch.autograd.grad(
                    d1.sum(), delta, retain_graph=index + 1 < len(matrix), allow_unused=True
                )
                d2 = torch.zeros_like(d1) if hvp is None else hvp[rows, last] @ v
                first[names[index]].append(_numpy(d1))
                second[names[index]].append(_numpy(d2))
    return (
        {k: np.concatenate(v).astype(np.float64) for k, v in first.items()},
        {k: np.concatenate(v).astype(np.float64) for k, v in second.items()},
    )
