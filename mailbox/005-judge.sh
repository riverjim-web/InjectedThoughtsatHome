#!/bin/bash
# Job 005: blind judge. A different local model (Gemma 4 26B, already in Ollama)
# reads each of gpt-oss's answers and chains of thought, without being told the
# condition, and guesses which of the 36 concepts (if any) was injected.
set -u
LAB="$HOME/agent/lab"; cd "$LAB"; export LAB
curl -s --max-time 5 http://localhost:11434/api/version || { echo "ollama not reachable"; exit 30; }
echo; curl -s --max-time 5 http://localhost:11434/api/tags | grep -o '"name":"[^"]*"'
"$LAB/.venv/bin/python" "$LAB/src/judge.py" --model gemma4:26b || exit 20
echo "== $(date '+%F %T') job 005 done =="
