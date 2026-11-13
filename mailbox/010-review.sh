#!/bin/bash
# Job 010: runs requested in the adversarial review (see src/extra2.py), then close the mailbox.
#   baseline  per-word activations for the 80 baseline words (for the structured null vectors)
#   logits    P("Yes") vs P("No") at the first answer token, vector on vs switched off before the
#             assistant turn; concept, random, split-noise and baseline-word vectors; plus three
#             unrelated "no" questions as a yes-bias control
#   latent    sampled detection trials with the concept vector switched off before the answer
#   null2     sampled detection trials with split-noise and baseline-word vectors (on throughout)
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
echo "ollama resident: $(curl -s --max-time 5 http://localhost:11434/api/ps | grep -o '"name":"[^"]*"' || echo none)"
rc=0
for ph in baseline logits latent null2; do
  "$PY" "$LAB/src/extra2.py" $ph --config "$LAB/config_main.json" || { echo "phase $ph failed"; rc=1; }
done
"$PY" "$LAB/src/extra2.py" summary
echo "== $(date '+%F %T') job 010 done (exit $rc) =="
echo $rc > "$LAB/mailbox/010-review.exit"
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
