#!/bin/bash
# Job 012b: resume job 012 (killed by a hangup signal at 13:57 when the SSH session closed).
# Finished readouts are skipped. Hangups are ignored this time. Closes the mailbox at the end.
trap '' HUP
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
rc=0
nohup "$PY" "$LAB/src/extra4.py" yn2 --config "$LAB/config_yn.json" || { echo "yn2 failed"; rc=1; }
"$PY" "$LAB/src/extra4.py" summary
echo "== $(date '+%F %T') job 012b done (exit $rc) =="
echo $rc > "$LAB/mailbox/012b-resume.exit"
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
