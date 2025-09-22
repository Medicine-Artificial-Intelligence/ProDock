#!/bin/bash

# Run flake8 with specified rules
flake8 . --count --max-complexity=40 --max-line-length=120 \
    --exclude='./Docs, dev/*, prodock/develop/*' \
    --per-file-ignores="__init__.py:F401" \
    --statistics