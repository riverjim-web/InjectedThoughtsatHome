#!/bin/bash
# Job 002: compare three ways of extracting concept vectors (how well each steers).
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
"$LAB/.venv/bin/python" "$LAB/src/lab.py" calib2 || exit 20
