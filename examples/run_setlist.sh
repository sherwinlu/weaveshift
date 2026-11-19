#!/usr/bin/env bash
python -m weaveshift.setlist examples/setlist_example.json \
  -o weaveshift_demo.wav \
  --apply-manifest-settings \
  --make-snippets
