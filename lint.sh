#!/bin/bash

# Run flake8 with specified rules
# E203 is ignored for black compatibility (black formats slices as `x[i : j]`).
flake8 . --count --max-complexity=40 --max-line-length=120 \
    --extend-ignore=E203 \
    --exclude='./Docs, dev/*, prodock/develop/*' \
    --per-file-ignores="__init__.py:F401" \
    --statistics