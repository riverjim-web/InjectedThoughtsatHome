#!/bin/bash
# Job 001: set up a self-contained Python + MLX inside ~/agent/lab, download
# gpt-oss-20b (MLX build, ~12GB), and run lab.py's smoke test.
# Nothing is installed outside ~/agent/lab.
set -u
LAB="$HOME/agent/lab"
cd "$LAB"
export UV_CACHE_DIR="$LAB/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$LAB/.cache/python"
export HF_HOME="$LAB/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export LAB
UV_VERSION=0.13.7
MODEL=mlx-community/gpt-oss-20b-MXFP4-Q8

echo "== $(date '+%F %T') job 001 =="
sw_vers 2>/dev/null; sysctl -n machdep.cpu.brand_string 2>/dev/null; sysctl -n hw.memsize 2>/dev/null
df -h "$HOME" | tail -1

echo "== ollama: unload anything resident so MLX gets the memory =="
LOADED=$(curl -s --max-time 5 http://localhost:11434/api/ps | grep -o '"name":"[^"]*"' | sed 's/"name":"//; s/"$//')
echo "loaded: ${LOADED:-none}"
for m in $LOADED; do
  curl -s --max-time 30 http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null && echo "unloaded $m"
done

echo "== uv $UV_VERSION (single binary, kept in $LAB/.bin) =="
mkdir -p "$LAB/.bin"
if [ ! -x "$LAB/.bin/uv" ]; then
  curl -LsSf "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz" \
    | tar -xz -C "$LAB/.bin" --strip-components 1 || { echo "uv download failed"; exit 10; }
fi
"$LAB/.bin/uv" --version || exit 11

echo "== python env =="
if [ ! -x "$LAB/.venv/bin/python" ]; then
  "$LAB/.bin/uv" venv --python 3.12 "$LAB/.venv" || exit 12
fi
"$LAB/.bin/uv" pip install --python "$LAB/.venv/bin/python" \
  "mlx==0.32.4" "mlx-lm==0.32.2" "transformers==5.22.2" "huggingface_hub==2.4.0" "numpy==2.5.4" || exit 13
"$LAB/.venv/bin/python" -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, 'mlx_lm', mlx_lm.__version__, mx.default_device())" || exit 14

echo "== model download: $MODEL =="
T0=$(date +%s)
for attempt in 1 2 3; do
  "$LAB/.venv/bin/python" -c "from huggingface_hub import snapshot_download; p = snapshot_download('$MODEL'); print(p)" && break
  echo "download attempt $attempt failed; retrying in 30s"; sleep 30
done
echo "download took $(( $(date +%s) - T0 ))s"
du -sh "$HF_HOME" 2>/dev/null

echo "== smoke test =="
"$LAB/.venv/bin/python" "$LAB/src/lab.py" smoke || exit 20
echo "== $(date '+%F %T') job 001 done =="
