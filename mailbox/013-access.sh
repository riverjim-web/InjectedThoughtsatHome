#!/bin/bash
# Job 013: does information about the injected concept reach the answer? (see src/extra5.py)
#   matcher  blind Gemma matcher for the "names it" grades (src/judge2.py); Gemma is unloaded after,
#            because Gemma and gpt-oss don't both fit in memory
#   fc       36-way forced choice, vector switched off before the answer (detect / neutral / word framings)
#   matrix   36x36 "Is the injected thought about <word>?" plus a priming twin, vector switched off
# Ignores hangups, reads stdin from /dev/null, and closes the mailbox only if everything succeeded.
trap '' HUP
exec < /dev/null
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
rc=0
echo "ollama: $(curl -s --max-time 5 http://localhost:11434/api/version || echo 'not reachable')"
"$PY" "$LAB/src/judge2.py" || { echo "matcher failed"; rc=1; }
curl -s --max-time 60 http://localhost:11434/api/generate -d '{"model":"gemma4:26b","keep_alive":0}' >/dev/null; sleep 5
echo "ollama resident after unload: $(curl -s --max-time 5 http://localhost:11434/api/ps | grep -o '"name":"[^"]*"' || echo none)"
"$PY" "$LAB/src/extra5.py" fc --config "$LAB/config_yn.json" || { echo "fc failed"; rc=1; }
"$PY" "$LAB/src/extra5.py" matrix --config "$LAB/config_yn.json" || { echo "matrix failed"; rc=1; }
"$PY" "$LAB/src/extra5.py" summary
echo "== $(date '+%F %T') job 013 done (exit $rc) =="
echo $rc > "$LAB/mailbox/013-access.exit"
if [ "$rc" -ne 0 ]; then echo "leaving the mailbox open so a fix can be queued"; exit $rc; fi
PIDS=$(ps -axo pid=,command= | awk '/zsh .*agent\/lab\/mailbox\.sh/ && !/awk/ {print $1}')
echo "closing mailbox; watcher pid(s): $PIDS"
for pid in $PIDS; do kill -TERM "$pid" && echo "sent TERM to $pid"; done
sleep 3
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "sent KILL to $pid"; done
ps -axo pid=,command= | grep -E "mailbox\.sh|caffeinate" | grep -v grep || echo "mailbox closed"
