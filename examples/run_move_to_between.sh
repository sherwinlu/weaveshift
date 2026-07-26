#!/usr/bin/env bash
python weaveshift_two_tracks.py "EDM1.wav" "EDM2.wav" \
  -o weaveshift_two_track_demo.wav \
  --mode profile \
  --profile edm \
  --make-snippets
