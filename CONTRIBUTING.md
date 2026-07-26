# Contributing to WeaveShift

Thank you for your interest in WeaveShift.

## Development setup

```bash
git clone https://github.com/sherwinlu/weaveshift.git
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install `ffmpeg` on your system before running tests or renders.

Optional Demucs support:

```bash
python -m pip install -e ".[demucs]"
```

## Running tests

```bash
python -m pytest
```

## Pull requests

1. Open an issue or comment on an existing one before large changes.
2. Keep diffs focused on one concern.
3. Add or update tests when behavior changes.
4. Run `python -m pytest` locally before opening a PR.
5. Update `README.md` or `CHANGELOG.md` when user-facing behavior changes.

## Coding style

Match the surrounding module: type hints, small focused functions, and minimal comments unless the logic is non-obvious.

## Audio content

Do **not** commit copyrighted or third-party audio masters. Tests and examples use procedurally generated sine/click WAVs created at runtime or via `examples/generate_fixtures.py`.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
