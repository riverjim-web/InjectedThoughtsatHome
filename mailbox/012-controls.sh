#!/bin/bash
# Job 012: controls for the detection-specific Yes shift (see src/extra4.py): two detection
# wordings, three uncertain self-report questions, and 18 factual questions, vector switched
# off before the answer. Then close the mailbox.
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
rc=0
"$PY" "$LAB/src/extra4.py" yn2 --config "$LAB/config_yn.json" || { echo "yn2 failed"; rc=1; }
"$PY" "$LAB/src/extra4.py" summary
echo "== $(date '+%F %T') job 012 done (exit $rc) =="
echo $rc > "$LAB/mailbox/012-controls.exit"
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
