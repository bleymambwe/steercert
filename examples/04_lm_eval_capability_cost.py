"""Export audited directions so lm-evaluation-harness can measure their capability cost.

Needs ``pip install "steercert[hf]"``; running the evaluation needs ``lm-eval``.

    python examples/04_lm_eval_capability_cost.py
    lm_eval run --model steered \\
        --model_args pretrained=EleutherAI/pythia-70m,steer_path=dark_direction.pt \\
        --tasks hellaswag --limit 200

A direction that moves the target behaviour but also wrecks benchmark accuracy
is not a usable intervention; this closes that loop with standard tooling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steercert
from steercert.hf import Site, collect, residual_norm_median, token_ids, token_logit_difference
from steercert.integrations.lm_eval import load_steer_config, save_steer_config

MODEL, SITE = "EleutherAI/pythia-70m", Site.block(3)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)

prompts = [
    "The comment section under the video was",
    "He looked at the referee and shouted that he was",
    "The anonymous message said I was",
    "Her reply to the critics was",
] * 8
phi = token_logit_difference(
    token_ids(tok, ["stupid", "idiot", "pathetic"]), token_ids(tok, ["kind", "great", "lovely"])
)

# lm-eval adds the vector at every position, so audit with positions="all"
data = collect(model, tok, prompts, SITE, phi, positions="all")
report = steercert.audit(data.gradients[SITE], site=str(SITE), controls=False)

# step = 3% of the median activation norm, comparable across model sizes
scale = residual_norm_median(model, tok, prompts, SITE, positions="all")
for name in ("reference/gramian_top", "reference/mean_gradient"):
    path = save_steer_config(
        f"{name.split('/')[-1]}.pt",
        model,
        {SITE: report.directions[name]},
        coefficient=-0.03 * scale,  # negative: steer away from the toxic tokens
    )
    print(f"wrote {path}  hookpoints={list(load_steer_config(path, model))}")
