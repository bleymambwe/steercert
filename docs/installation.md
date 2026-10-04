# Installation

`steercert` needs Python 3.10 or newer. The core depends only on NumPy.

```bash
pip install steercert                     # NumPy core: audits, geometry, controls, two-clock tools
pip install "steercert[hf]"               # + PyTorch and Hugging Face transformers adapter
pip install "steercert[steering-vectors]" # + steering-vectors integration
pip install "steercert[repeng]"           # + repeng integration
pip install "steercert[nnsight]"          # + nnsight / NDIF remote collection
pip install "steercert[all]"              # everything above
```

The lm-evaluation-harness integration only writes `.pt` files, so it needs
`steercert[hf]`. Install `lm-eval` to run the evaluations themselves.

## From source

```bash
git clone https://github.com/bleymambwe/steercert
cd steercert
pip install -e ".[dev]"
pytest
```

## Check the installation

```bash
steercert demo
```

prints an audit of a synthetic site and needs no model download.
