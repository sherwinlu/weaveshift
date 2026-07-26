#!/usr/bin/env bash
python weaveshift_setlist.py examples/setlist_example.json \
  -o weaveshift_demo.wav \
  --apply-manifest-settings \
  --make-snippets
