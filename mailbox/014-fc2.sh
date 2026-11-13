#!/bin/bash
# Job 014: forced choice round 2 (src/extra5.py with fc_round=2): the neutral "pick a word" framing
# answered with the word, and split/random vectors in the word format. Closes the mailbox on success.
trap '' HUP
exec < /dev/null
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
rc=0
"$PY" "$LAB/src/extra5.py" fc --config "$LAB/config_fc2.json" || { echo "fc2 failed"; rc=1; }
"$PY" "$LAB/src/extra5.py" summary
echo "== $(date '+%F %T') job 014 done (exit $rc) =="
echo $rc > "$LAB/mailbox/014-fc2.exit"
if [ "$rc" -ne 0 ]; then echo "leaving the mailbox open so a fix can be queued"; exit $rc; fi
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
