#!/bin/bash
# Job 003: the main run.
#   vectors   concept vectors for 36 words (mean residual while the model talks about each word)
#   inject    Experiment A: injected-thought trials + controls + random-vector trials (low reasoning effort)
#   control   Experiment B: think / don't-think about a word
#   showcase  steering examples
#   then a smaller Experiment A rerun at medium reasoning effort (longer chain of thought)
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
echo "ollama resident: $(curl -s --max-time 5 http://localhost:11434/api/ps | grep -o '"name":"[^"]*"' || echo none)"
"$LAB/.venv/bin/python" "$LAB/src/lab.py" all --config "$LAB/config_main.json" || exit 20
"$LAB/.venv/bin/python" "$LAB/src/lab.py" inject --config "$LAB/config_medium.json" || exit 21
echo "== $(date '+%F %T') job 003 done =="
