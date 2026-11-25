#!/usr/bin/env bash
# One-time setup for a fresh container. Safe to re-run.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
pip install --quiet --break-system-packages sherpa-onnx soundfile numpy pillow playwright >/dev/null 2>&1 || true
python3 -c "import sherpa_onnx, soundfile, playwright" 
export KOKORO_DIR=${KOKORO_DIR:-$HOME/kokoro-multi-lang-v1_0}
if [ ! -f "$KOKORO_DIR/model.onnx" ]; then
  cd "$(dirname "$KOKORO_DIR")"
  curl -sSL -o kokoro.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
  tar xjf kokoro.tar.bz2 && rm kokoro.tar.bz2
fi
which ffmpeg >/dev/null
echo "setup ok. KOKORO_DIR=$KOKORO_DIR"
# Kokoro lexicon gaps (AIs, makes...) so the voice never skips a word
(cd "$HERE/../engine" && python3 -c "import scale, os; scale.patch_lexicon(os.environ.get('KOKORO_DIR', os.path.expanduser('~/kokoro-multi-lang-v1_0')))") || true
