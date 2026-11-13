#!/bin/zsh
# Claude's overnight mailbox (see README.md in this folder).
# Claude's own shell on this Mac is sandboxed: it can write files here but can't
# run anything natively (no GPU, no Ollama). This script bridges that: it watches
# ~/agent/lab/mailbox for job files (NNN-name.sh + NNN-name.go) and runs each job
# once, natively, as you. Every job script and its log stays in that folder.
# It closes itself after 9 hours. Ctrl+C closes it any time.

LAB="$HOME/agent/lab"
BOX="$LAB/mailbox"
mkdir -p "$BOX"
cd "$LAB" || exit 1
setopt NULL_GLOB

END=$(( $(date +%s) + 32400 ))   # 9 hours
echo "Mailbox open. Watching $BOX until $(date -r $END '+%H:%M'). Ctrl+C to stop."

while [ "$(date +%s)" -lt "$END" ]; do
  date '+%Y-%m-%d %H:%M:%S' > "$BOX/heartbeat"
  for go in "$BOX"/*.go; do
    [ -e "$go" ] || continue
    job="${go%.go}"
    name="$(basename "$job")"
    mv "$go" "$job.taken"
    if [ ! -f "$job.sh" ]; then
      echo "missing $job.sh" > "$job.log"; echo 127 > "$job.exit"; continue
    fi
    echo "[$(date '+%H:%M:%S')] running $name"
    date '+%Y-%m-%d %H:%M:%S' > "$job.started"
    ( cd "$LAB" && /bin/bash "$job.sh" ) > "$job.log" 2>&1
    echo $? > "$job.exit"
    echo "[$(date '+%H:%M:%S')] finished $name (exit $(cat "$job.exit"))"
  done
  sleep 15
done
echo "Mailbox closed (time limit reached)."
