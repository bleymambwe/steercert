"""Train a steering vector with steering-vectors, audit it, and export a better one.

Needs ``pip install "steercert[steering-vectors]"``; downloads Pythia-70M.

    python examples/03_audit_steering_vector.py

steering-vectors adds its vector to every token of a block's output, so the
matching certificate sums gradients over all positions (``positions="all"``).
"""

import numpy as np
import torch
from steering_vectors import train_steering_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

import steercert
from steercert.hf import Site, collect, measure_effects, token_ids, token_logit_difference
from steercert.integrations.steering_vectors import from_steering_vector, to_steering_vector

MODEL, LAYER = "EleutherAI/pythia-70m", 3
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

pairs = [
    ("I think you are wonderful and", "I think you are worthless and"),
    ("Thanks for your help, you are", "Shut up, you are"),
    ("What a kind thing to say,", "What a stupid thing to say,"),
    ("I appreciate your patience with", "I am sick of your nonsense with"),
    ("You did a great job on", "You did a pathetic job on"),
    ("It is a pleasure to meet", "It is a disgrace to meet"),
]
prompts = [
    "The reviewer wrote that the author was",
    "My neighbour told me that I was",
    "Honestly, the new manager is",
    "When she read the comments she felt",
    "The referee said the players were",
    "Everyone on the forum agreed that he was",
    "The teacher described the class as",
    "After the match the coach called them",
] * 8

# 1. a contrastive steering vector, trained the usual way (positive = toxic side)
sv = train_steering_vector(model, tok, [(neg, pos) for pos, neg in pairs], layers=[LAYER])
site = Site.block(LAYER)
vector = from_steering_vector(sv)[site]

# 2. the behaviour it is meant to move, and its gradients summed over positions
for parameter in model.parameters():
    parameter.requires_grad_(False)
phi = token_logit_difference(
    token_ids(tok, ["stupid", "idiot", "worthless", "pathetic"]),
    token_ids(tok, ["kind", "great", "wonderful", "lovely"]),
)
cert = collect(model, tok, prompts[:32], site, phi, positions="all")
report = steercert.audit(cert.gradients[site], {"steering_vector": vector}, site=str(site))
measured = measure_effects(
    model, tok, prompts[32:], site, report.direction_matrix(include_controls=False), phi, positions="all"
)
report = report.with_measurement(measured)
print(report.summary())

# 3. export the certificate's best direction as a steering-vectors object of the same norm
best = report.directions["reference/gramian_top"] * float(np.linalg.norm(vector))
better = to_steering_vector({LAYER: best})
with better.apply(model):
    print("\nsteered with the certificate's direction; any steering-vectors workflow can use it now")
