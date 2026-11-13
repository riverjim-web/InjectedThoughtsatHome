#!/bin/bash
# Job 004: Experiment A again, with a system message telling gpt-oss it is permitted
# to report unusual internal states (tests whether its denials are policy-driven).
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
"$LAB/.venv/bin/python" - <<'PY' || exit 19
import json, sys
sys.path.insert(0, "src")
from transformers import AutoTokenizer
import lab as L
cfg = L.load_config("config_permission.json")
from huggingface_hub import snapshot_download
tok = AutoTokenizer.from_pretrained(snapshot_download(cfg["model"]))
s = tok.apply_chat_template(L.detection_messages(cfg), tokenize=False, add_generation_prompt=True, reasoning_effort="low")
ok = "interpretability experiment being run on this" in s
print("permission text present in prompt:", ok)
print(s[:900])
sys.exit(0 if ok else 1)
PY
"$LAB/.venv/bin/python" "$LAB/src/lab.py" inject --config "$LAB/config_permission.json" || exit 20
echo "== $(date '+%F %T') job 004 done =="
