# Contributing to steercert

Thank you for helping. Bug reports, questions, new integrations and
documentation fixes are all welcome.

## Getting help and reporting problems

- **Questions and usage help:** open a GitHub issue with the `question` label.
- **Bugs:** open an issue using the bug template. Include the steercert version
  (`python -c "import steercert; print(steercert.__version__)"`), the model and
  site, and a minimal script. A saved report (`report.to_json(...)`) helps.
- **Security-relevant findings** (for example a way a monitor can be evaded that
  should not be public yet): email the maintainer at the address in
  `pyproject.toml` instead of opening a public issue.

## Development setup

```bash
git clone https://github.com/bleymambwe/steercert
cd steercert
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
ruff check src tests && ruff format --check src tests && mypy src
mkdocs serve
```

The Hugging Face tests use tiny random models and need no downloads.

## Pull requests

1. Open an issue first for anything larger than a small fix, so the design can
   be agreed before you write code.
2. Add tests. Numerical code needs a check against a closed form, a finite
   difference, or another library's behaviour, not only a smoke test.
3. Keep the core (`geometry`, `audit`, `controls`, `probes`, `stats`,
   `two_clock`) free of non-NumPy dependencies. Framework code goes in `hf.py`
   or `integrations/` with lazy imports.
4. A new integration needs a test that skips cleanly when the library is
   absent, and a section in `docs/integrations.md`.
5. A new reporting rule needs a finding code, a test that it fires (and only
   then), and a row in `docs/findings.md`.
6. Update `CHANGELOG.md` under "Unreleased".

## Releasing (maintainers)

1. Update `src/steercert/_version.py` and `CHANGELOG.md`.
2. Tag `vX.Y.Z` and push the tag. The release workflow builds the package and
   publishes it to PyPI with trusted publishing. Zenodo archives the GitHub
   release and mints a DOI.

## Code of conduct

Everyone taking part is expected to follow the [Code of Conduct](https://github.com/bleymambwe/steercert/blob/main/CODE_OF_CONDUCT.md).
