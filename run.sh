#!/usr/bin/env bash
# One-shot: build llama.cpp, make (or use) models, measure, report.
#
#   ./run.sh <label> [note]                 # synthetic TinyLlama-1.1B shapes, 4 formats
#   MODELS="a.gguf b.gguf" ./run.sh <label> # your own GGUFs instead
#   SHAPE=llama-3.2-1b ./run.sh <label>     # a different synthetic shape
#
# Linux and macOS. On Windows, use WSL2 (see README).
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${1:?usage: ./run.sh <machine-label> [note]}
NOTE=${2:-}
SHAPE=${SHAPE:-tinyllama-1.1b}
LLAMA_DIR=${LLAMA_DIR:-third_party/llama.cpp}
LLAMA_REF=${LLAMA_REF:-master}

python3 -m pip install -q -r requirements.txt

if [ ! -x "$LLAMA_DIR/build/bin/llama-bench" ]; then
  [ -d "$LLAMA_DIR" ] || git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
  cmake --build "$LLAMA_DIR/build" -j --target llama-bench llama-quantize
fi
BIN="$LLAMA_DIR/build/bin"

if [ -z "${MODELS:-}" ]; then
  mkdir -p models
  F16="models/$SHAPE-synthetic-F16.gguf"
  [ -f "$F16" ] || python3 harness/make_synthetic_gguf.py --shape "$SHAPE" --out models
  MODELS="$F16"
  for q in Q8_0 Q4_K_M Q4_0; do
    out="models/$SHAPE-synthetic-$q.gguf"
    [ -f "$out" ] || "$BIN/llama-quantize" "$F16" "$out" "$q" >/dev/null 2>&1
    MODELS="$MODELS $out"
  done
fi

# shellcheck disable=SC2086
python3 harness/bench.py --label "$LABEL" --note "$NOTE" --llama-bench "$BIN/llama-bench" --models $MODELS
python3 harness/report.py
