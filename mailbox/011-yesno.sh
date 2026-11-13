#!/bin/bash
# Job 011: the cleaner yes/no test (see src/extra3.py), then close the mailbox.
# The detection question gets "Answer yes or no." so it answers in the same format as the
# factual control questions; the shift in Yes-vs-No odds is compared like for like.
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
rc=0
"$PY" "$LAB/src/extra2.py" baseline --config "$LAB/config_yn.json" || { echo "baseline failed"; rc=1; }
"$PY" "$LAB/src/extra3.py" yn --config "$LAB/config_yn.json" || { echo "yn failed"; rc=1; }
"$PY" "$LAB/src/extra3.py" summary
echo "== $(date '+%F %T') job 011 done (exit $rc) =="
echo $rc > "$LAB/mailbox/011-yesno.exit"
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
