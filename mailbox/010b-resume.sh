#!/bin/bash
# Job 010b: resume job 010's last phase (null2), which stopped at trial 59 when the session that
# started the mailbox dropped. Finished trials are skipped. Job 011 runs next and closes the mailbox.
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_OFFLINE=1 LAB
PY="$LAB/.venv/bin/python"
"$PY" "$LAB/src/extra2.py" null2 --config "$LAB/config_main.json" || exit 20
"$PY" "$LAB/src/extra2.py" summary
echo "== $(date '+%F %T') job 010b done =="
