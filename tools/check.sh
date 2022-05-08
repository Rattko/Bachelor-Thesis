#!/usr/bin/env zsh

# Find a file using fzf
file=$(fzf)

# Run linters and a static type checker
pylint $file
pylama $file
flake8 $file
mypy --namespace-packages --ignore-missing-imports $file
