#!/bin/bash
# Job 009: two controls for the "steering just makes it say yes" objection (see src/extra.py),
# then close the mailbox so nothing stays listening.
#   random2  random vectors, matched in size to the concept vectors, at layers 8/12/16/20 and
#            strengths 0.3/0.4/0.6 (36 per setting), asked the usual detection question
#   yesbias  the usual setup with an unrelated yes/no question instead, with and without injection
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
echo "ollama resident: $(curl -s --max-time 5 http://localhost:11434/api/ps | grep -o '"name":"[^"]*"' || echo none)"
"$PY" "$LAB/src/extra.py" random2 --config "$LAB/config_main.json"; e1=$?
"$PY" "$LAB/src/extra.py" yesbias --config "$LAB/config_main.json"; e2=$?
"$PY" "$LAB/src/extra.py" summary
echo "== $(date '+%F %T') job 009 done (random2 exit $e1, yesbias exit $e2) =="
# Closing the watcher stops it from recording this job's exit code, so record it here.
echo $(( e1 > e2 ? e1 : e2 )) > "$LAB/mailbox/009-extra.exit"
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
