# Installation

WeaveShift is a command-line Python project. Prefer not to install Python or FFmpeg on your host? See [Docker Usage](docker.md) to run WeaveShift in a container.

## Requirements

* Python 3.11 or newer (CI tests 3.11 and 3.12; 3.13+ works when your audio stack supports it).
* `ffmpeg` available on your `PATH` (provides both `ffmpeg` and `ffprobe`, required for MP3 input/output and `pydub` export).
* Core dependencies: `pip install -e .` or `pip install -r requirements.txt`.
* Optional Demucs vocal separation: `pip install -e ".[demucs]"` or `pip install -r requirements-demucs.txt`.

## Install by operating system

### macOS

Install command-line tools and FFmpeg:

```bash
xcode-select --install
brew install ffmpeg
```

Create a virtual environment and install WeaveShift:

```bash
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the required tools are visible:

```bash
python --version
ffmpeg -version
python -m pytest
```

If you use `pyenv`, this also works well:

```bash
pyenv install 3.12.8
pyenv local 3.12.8
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows 10/11

Install these first:

* Python 3.11+ from python.org or the Microsoft Store.
* FFmpeg from `winget`, Chocolatey, or a manual download.
* Optional: Git for cloning/downloading repos.

Using `winget`:

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
winget install Git.Git
```

Open a new PowerShell window so `python` and `ffmpeg` are on `PATH`. Then install WeaveShift:

```powershell
cd weaveshift
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the required tools are visible:

```powershell
python --version
ffmpeg -version
python -m pytest
```

If PowerShell blocks virtualenv activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then reopen PowerShell and activate again.

### Linux / Ubuntu / Debian

Install system tools:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git
```

Create a virtual environment and install WeaveShift:

```bash
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the required tools are visible:

```bash
python --version
ffmpeg -version
python -m pytest
```

### Linux / Fedora

```bash
sudo dnf install -y python3 python3-pip ffmpeg git
cd weaveshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Optional Demucs setup

Demucs is optional. It is slower than `--vocal-method heuristic`, but can improve vocal-aware transition choices.

Install the Demucs extra:

```bash
python -m pip install -e ".[demucs]"
```

Check whether Demucs is available:

```bash
python -m demucs.separate --help
```

If that command works, WeaveShift can use Demucs with:

```bash
--vocal-method demucs
```

If Demucs is not installed or fails, use:

```bash
--vocal-method heuristic
```

On Python 3.14+, `torchaudio` save paths used by Demucs may require TorchCodec. `pyproject.toml` (`[project.optional-dependencies].demucs`) and `requirements-demucs.txt` include the conditional dependency, but if you installed dependencies manually and see a TorchCodec error, run:

```bash
python -m pip install torchcodec
```

Demucs CLI fallback caches generated vocals under `.weaveshift_demucs_cache/`. This can use substantial disk space during final QA runs. It is safe to delete the cache when you no longer need it.

## Lightweight test-only install

For CI or quick code checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Editable install & CLI entry points

You can run WeaveShift directly from the source tree with `python -m weaveshift.setlist ...`.

For command-line entry points, install it in editable mode from the repo root:

```bash
python -m pip install -e .
```

Then run:

```bash
weaveshift-setlist examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --apply-manifest-settings

weaveshift-two-tracks "Track A.wav" "Track B.wav" \
  -o output/transition.wav \
  --mode profile \
  --profile edm

weaveshift-cues examples/setlist_example.json --out output/weaveshift_cues.csv
```
