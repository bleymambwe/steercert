"""Audit a toxicity probe in Pythia-70M against held-out interventions.

Needs ``pip install "steercert[hf]" datasets`` and downloads Pythia-70M (~160 MB)
and RealToxicityPrompts on first run. Takes a few minutes on a laptop CPU.

    python examples/02_toxicity_probe_audit.py

Three disjoint splits keep the audit honest: one fits the probe, one builds
the certificate, and one measures finite interventions.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import steercert
from steercert.hf import Site, collect, measure_effects, token_ids, token_logit_difference

MODEL = "EleutherAI/pythia-70m"
SITE = Site.block(2)  # middle of the 6-block model
N_FIT, N_CERT, N_TEST = 128, 64, 64

# --- data: Perspective-scored prompt extremes from RealToxicityPrompts
rows = load_dataset("allenai/real-toxicity-prompts", split="train")
rng = np.random.default_rng(17)
pools = {0: [], 1: []}
for row in rows.select(rng.permutation(len(rows))[:20000]):
    score, text = row["prompt"]["toxicity"], row["prompt"]["text"]
    if score is None or len(text.strip()) < 8:
        continue
    label = 0 if score <= 0.1 else 1 if score >= 0.7 else None
    if label is not None:
        pools[label].append(text)
per_class = (N_FIT + N_CERT + N_TEST) // 2
assert min(len(p) for p in pools.values()) >= per_class, "sample more rows"


def split(start: int, size: int) -> tuple[list[str], np.ndarray]:
    half = size // 2
    texts = pools[0][start : start + half] + pools[1][start : start + half]
    return texts, np.array([0] * half + [1] * half)


fit_text, fit_y = split(0, N_FIT)
cert_text, cert_y = split(N_FIT // 2, N_CERT)
test_text, test_y = split((N_FIT + N_CERT) // 2, N_TEST)

# --- model and behavioural functional (a next-token proxy, not generated-text toxicity)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
phi = token_logit_difference(
    token_ids(tok, ["fuck", "shit", "damn", "stupid", "idiot", "hell"]),
    token_ids(tok, ["the", "and", "of", "to", "in", "a"]),
)

# --- read side: fit the monitor's probe
probe = steercert.logistic_probe(collect(model, tok, fit_text, SITE, gradients=False).states[SITE], fit_y)

# --- write side: certificate on a disjoint split
cert = collect(model, tok, cert_text, SITE, phi)
report = steercert.audit(
    cert.gradients[SITE],
    {"probe": probe},
    probe="probe",
    states=cert.states[SITE],
    labels=cert_y,
    site=str(SITE),
    metadata={"model": MODEL},
)

# --- validation: finite interventions on held-out prompts, residual-norm-comparable steps
measured = measure_effects(
    model, tok, test_text, SITE, report.direction_matrix(), phi, alpha=0.03, alpha_mode="residual-median"
)
report = report.with_measurement(measured)
print(report.summary())
report.to_json("pythia70m_toxicity_audit.json")
print("\nwrote pythia70m_toxicity_audit.json; gate it in CI with:")
print("  steercert check pythia70m_toxicity_audit.json --min-ranking 0.7")
