#!/bin/bash

set -ex

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Need to activate some virtualenv"
    exit 1
fi

pip install -U pip
pip install -r dev-requirements.txt
pip install -e .

rm -f .git/hooks/pre-commit
git-pre-commit-hook install --plugin file_size --plugin json --plugin yaml
