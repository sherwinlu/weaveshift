# syntax=docker/dockerfile:1
FROM python:3.12-slim

ARG INSTALL_DEMUCS=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_HOME=/tmp/.cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /tmp/src/
COPY weaveshift /tmp/src/weaveshift

RUN if [ "$INSTALL_DEMUCS" = "1" ]; then \
        pip install --no-cache-dir "/tmp/src[demucs]"; \
    else \
        pip install --no-cache-dir /tmp/src; \
    fi \
    && rm -rf /tmp/src

RUN useradd --create-home --shell /usr/sbin/nologin weaveshift \
    && mkdir -p /work /tmp/.cache/torch /tmp/numba_cache \
    && chown -R weaveshift:weaveshift /work /tmp/.cache /tmp/numba_cache \
    && chmod 1777 /tmp/.cache /tmp/.cache/torch /tmp/numba_cache

USER weaveshift
WORKDIR /work

CMD ["weaveshift-setlist", "--help"]
