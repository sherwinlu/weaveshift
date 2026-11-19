#!/usr/bin/env bash
python -m weaveshift.two_tracks "EDM1.wav" "EDM2.wav" \
  -o weaveshift_two_track_demo.wav \
  --mode profile \
  --profile edm \
  --make-snippets
