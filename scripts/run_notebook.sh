#!/usr/bin/env bash
# Rebuild coursework.ipynb from its text source and execute it top to bottom.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/jupytext --to ipynb coursework.py -o coursework.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 coursework.ipynb
echo "Done: coursework.ipynb executed."
