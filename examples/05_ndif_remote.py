"""Certificates on an 8B model through nnsight and NDIF, with no local GPU.

Needs ``pip install "steercert[nnsight]"``, an NDIF API key
(https://login.ndif.us) set with ``nnsight.CONFIG.set_default_api_key(...)``,
and a Hugging Face token with access to the model. Experimental: check
https://nnsight.net/status for the models NDIF currently serves.

    python examples/05_ndif_remote.py
"""

from nnsight import LanguageModel

import steercert
from steercert.hf import Site, token_ids, token_logit_difference
from steercert.integrations.nnsight import collect_nnsight

MODEL = "meta-llama/Llama-3.1-8B"
LAYERS = [8, 16, 24]

model = LanguageModel(MODEL)
phi = token_logit_difference(
    token_ids(model.tokenizer, ["stupid", "idiot", "pathetic"]),
    token_ids(model.tokenizer, ["kind", "great", "lovely"]),
)
prompts = [
    "The comment section under the video was",
    "He looked at the referee and shouted that he was",
    "The anonymous message said I was",
    "Her reply to the critics was",
] * 16

data = collect_nnsight(model, prompts, LAYERS, phi, remote=True, batch_size=8)
for layer in LAYERS:
    site = Site.block(layer)
    report = steercert.audit(data.gradients[site], site=str(site), controls=False)
    g = report.geometry
    print(f"{site}: |g_bar|={g.mean_gradient_norm:.4g}  stable rank={g.stable_rank:.2f}")
