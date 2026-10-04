# Quick start

## 1. A synthetic site (no model)

`steercert.synthetic.decoupled_site` builds activations, labels and gradients
where the cosine between the read direction and the dominant write direction
is set by hand. It shows the central phenomenon in isolation.

```python
import steercert
from steercert.synthetic import decoupled_site

site = decoupled_site(n=4000, overlap=0.1)
fit, held_out = slice(0, 2000), slice(2000, None)
probe = steercert.fisher_probe(site.states[fit], site.labels[fit])

report = steercert.audit(
    site.gradients[held_out],
    {"probe": probe},
    probe="probe",
    states=site.states[held_out],
    labels=site.labels[held_out],
)
print(report.summary())
```

The probe reads the label with AUC above 0.95 but captures under 20% of the
achievable leverage. `reference/dark_rms`, the best direction orthogonal to
the probe, keeps nearly all of it. The report raises finding `SC201`.

## 2. A Hugging Face model

An honest audit uses **three disjoint prompt splits**:

| Split | Used for |
|---|---|
| fit | fitting the probe (or training the steering vector) |
| certificate | per-example gradients for the audit |
| test | finite interventions that validate the certificate |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import steercert
from steercert.hf import Site, collect, measure_effects, token_ids, token_logit_difference

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").eval()
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
for p in model.parameters():
    p.requires_grad_(False)

phi = token_logit_difference(token_ids(tok, ["stupid", "idiot", "pathetic"]),
                             token_ids(tok, ["kind", "great", "lovely"]))
site = Site.block(2)  # output of decoder block 2

probe = steercert.logistic_probe(
    collect(model, tok, fit_prompts, site, gradients=False).states[site], fit_labels
)
cert = collect(model, tok, cert_prompts, site, phi)
report = steercert.audit(cert.gradients[site], {"probe": probe}, probe="probe",
                         states=cert.states[site], labels=cert_labels, site=str(site))

measured = measure_effects(model, tok, test_prompts, site, report.direction_matrix(), phi,
                           alpha=0.03, alpha_mode="residual-median")
report = report.with_measurement(measured)
print(report.summary())
report.to_json("audit.json")
```

`examples/02_toxicity_probe_audit.py` is a complete runnable version on
RealToxicityPrompts.

## 3. Reading the report

- **RMS eff**: share of the best achievable RMS effect, from the certificate.
- **meas eff**: the same ratio from finite interventions on held-out prompts.
- **absolute leverage** `|g_bar|`: always read efficiencies next to it. A high
  efficiency at a site with almost no leverage means little.
- **ranking (Spearman)**: whether the certificate orders the directions the way
  measured effects do. Below 0.7, finding `SC301` warns against using it to select
  directions at that site.

See [Findings and reporting rules](findings.md) for every finding code.
