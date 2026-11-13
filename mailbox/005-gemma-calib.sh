#!/bin/bash
# Job 005: comparison model. Download Gemma 4 26B-A4B (MLX 4-bit, ~15GB) and run
# the concept-vector steering calibration on it. Results go to results_gemma/.
set -u
LAB="$HOME/agent/lab"; cd "$LAB"
export HF_HOME="$LAB/.cache/huggingface" HF_HUB_DISABLE_PROGRESS_BARS=1 LAB
export LAB_RESULTS="$LAB/results_gemma"
MODEL=mlx-community/gemma-4-26b-a4b-it-4bit
for attempt in 1 2 3; do
  "$LAB/.venv/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" && break
  echo "download attempt $attempt failed; retrying in 30s"; sleep 30
done
du -sh "$HF_HOME"
export HF_HUB_OFFLINE=1
"$LAB/.venv/bin/python" "$LAB/src/lab.py" calib2 --config "$LAB/config_gemma_calib.json" || exit 20
echo "== $(date '+%F %T') job 005 done =="
