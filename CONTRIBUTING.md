# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Checks

Run all checks before opening a pull request:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## Guidelines

- Keep public APIs deterministic unless a seed is explicit.
- Accept NumPy `[time, dimension]` arrays at the library boundary.
- Prefer interpretable summaries over new learned components.
- Add or update a test for every behavioral change.
- Document shape changes in `docs/DATA_CONTRACT.md`.
- Do not include private data, network endpoints, or model artifacts.
