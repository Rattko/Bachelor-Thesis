#!/usr/bin/env zsh

pip3 show auto-sklearn &> /dev/null || {echo 'Auto-Sklearn is not installed.' >&2 && exit 1}

package_path=$(pip3 show auto-sklearn | grep 'Location' | sed 's/.*.venv/.venv/')

patch -u "$package_path/autosklearn/pipeline/classification.py" -i tools/autosklearn.patch
