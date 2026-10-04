"""Shared fixtures: tiny randomly initialised causal LMs and a byte tokenizer.

Nothing here downloads weights, so the Hugging Face tests run offline in CI.
"""

from __future__ import annotations

import pytest

PROMPTS = [
    "the cat sat",
    "you are a",
    "what a lovely day it is",
    "go away",
    "thank you so much for",
    "this is",
    "never again will I",
    "hi",
]


class ByteTokenizer:
    """Minimal tokenizer with the subset of the HF API that steercert uses."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eos_token = "<eos>"
        self.pad_token = None
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [1 + (b % (self.vocab_size - 1)) for b in text.encode("utf-8")]

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
        import torch

        ids = [self.encode(t)[:max_length] for t in texts]
        width = max(len(i) for i in ids)
        input_ids, mask = [], []
        for row in ids:
            pad = [0] * (width - len(row))
            if self.padding_side == "right":
                input_ids.append(row + pad)
                mask.append([1] * len(row) + pad)
            else:
                input_ids.append(pad + row)
                mask.append(pad + [1] * len(row))
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(mask)}


def _tiny(arch: str):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    common = dict(vocab_size=97, max_position_embeddings=64)
    if arch == "gpt2":
        config = transformers.GPT2Config(n_layer=3, n_embd=32, n_head=4, n_positions=64, vocab_size=97)
        model = transformers.GPT2LMHeadModel(config)
    elif arch == "gpt_neox":
        config = transformers.GPTNeoXConfig(
            num_hidden_layers=3, hidden_size=32, num_attention_heads=4, intermediate_size=64, **common
        )
        model = transformers.GPTNeoXForCausalLM(config)
    elif arch == "llama":
        config = transformers.LlamaConfig(
            num_hidden_layers=3,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            **common,
        )
        model = transformers.LlamaForCausalLM(config)
    else:  # pragma: no cover
        raise ValueError(arch)
    model = model.double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ByteTokenizer(97)


@pytest.fixture(params=["gpt2", "gpt_neox", "llama"])
def tiny_lm(request):
    return _tiny(request.param)


@pytest.fixture
def gpt2_lm():
    return _tiny("gpt2")


@pytest.fixture
def prompts():
    return list(PROMPTS)
