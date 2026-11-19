# Docker Usage

WeaveShift ships a `Dockerfile` based on `python:3.12-slim` with `ffmpeg`, `ffprobe`, and `libsndfile1` pre-installed so you can run all three CLIs without installing Python or FFmpeg on your host.

## Build the image

Core image (fast, small — supports `--vocal-method heuristic` and `--vocal-method none`):

```bash
docker build -t weaveshift .
```

Optional Demucs image (larger — adds `torch`, `torchaudio`, and `demucs` for `--vocal-method demucs` / `auto`):

```bash
docker build --build-arg INSTALL_DEMUCS=1 -t weaveshift:demucs .
```

## Run the CLIs

Mount your working directory containing your manifest and audio files to `/work`:

```bash
# Optional: generate synthetic demo fixtures if trying the shipped example
docker run --rm -v "$PWD:/work" weaveshift \
  python examples/generate_fixtures.py

# Setlist mix
docker run --rm -v "$PWD:/work" weaveshift \
  weaveshift-setlist examples/setlist_example.json \
  -o output/weaveshift_demo.wav \
  --apply-manifest-settings \
  --make-snippets

# Two-track transition
docker run --rm -v "$PWD:/work" weaveshift \
  weaveshift-two-tracks "Track A.wav" "Track B.mp3" \
  -o output/transition.wav \
  --mode profile \
  --profile edm \
  --vocal-method heuristic

# Video cue sheet
docker run --rm -v "$PWD:/work" weaveshift \
  weaveshift-cues examples/setlist_example.json \
  --out output/weaveshift_cues.csv
```

Running with no arguments prints `weaveshift-setlist --help`:

```bash
docker run --rm weaveshift
```

## File ownership on Linux

The container runs as a non-root `weaveshift` user. On Linux, pass your host UID/GID so exported audio and reports are owned by your user:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" weaveshift \
  weaveshift-setlist setlist.json -o output/mix.wav
```

All runtime caches (`NUMBA_CACHE_DIR`, `XDG_CACHE_HOME`, `TORCH_HOME`) point under `/tmp`, so arbitrary `-u uid:gid` values work cleanly.

## Demucs tips & GPU caveats

* **Persist model weights:** Mount a named volume at `/tmp/.cache/torch` so Demucs does not re-download `htdemucs` weights on every run:

  ```bash
  docker run --rm \
    -v "$PWD:/work" \
    -v weaveshift-torch-cache:/tmp/.cache/torch \
    weaveshift:demucs \
    weaveshift-setlist setlist.json \
    -o output/mix.wav \
    --transition-mode profile \
    --profile vocal_trance \
    --vocal-method demucs
  ```

* **NVIDIA GPU (Linux):** Pass `--gpus all` if the NVIDIA Container Toolkit is installed.
* **macOS Apple Silicon caveat:** Docker Desktop / Linux VMs on macOS cannot access Apple Silicon MPS. For `--vocal-method demucs` on Apple Silicon Macs, a native Python install is significantly faster than CPU Demucs inside Docker.
