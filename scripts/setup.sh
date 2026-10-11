#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: Python env, llama.cpp, model files, and test media.
# Needs: uv, curl, ffmpeg, pdftoppm (poppler). No admin rights required. ~6 GB of downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
for t in uv curl ffmpeg pdftoppm say afconvert; do command -v $t >/dev/null || { echo "missing: $t"; exit 1; }; done

echo "== Python environment"
uv venv -q --python 3.13 .venv
uv pip install -q -r requirements.txt

echo "== llama.cpp b11541 (Metal build)"
if [ ! -x llamacpp/llama-b11541/llama-server ]; then
  mkdir -p llamacpp
  curl -fsSL -o llamacpp/llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b11541/llama-b11541-bin-macos-arm64.tar.gz
  tar -xzf llamacpp/llama.tar.gz -C llamacpp && rm llamacpp/llama.tar.gz
  xattr -dr com.apple.quarantine llamacpp 2>/dev/null || true
fi

echo "== GGUF model files"
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download as d
for repo, f in [("ggml-org/embeddinggemma-2-GGUF", "embeddinggemma-2-Q8_0.gguf"),
                ("unsloth/embeddinggemma-2-GGUF", "embeddinggemma-2-UD-Q4_K_XL.gguf"),
                ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf")]:
    print(d(repo, f, local_dir="gguf"))
PY

echo "== PDF pages (3 arXiv papers)"
mkdir -p assets/pdf
for p in "attention 1706.03762v7" "bert 1810.04805v2" "resnet 1512.03385v1"; do
  set -- $p
  [ -f assets/pdf/$1.pdf ] || curl -fsSL -o assets/pdf/$1.pdf https://arxiv.org/pdf/$2
  mkdir -p assets/pdf/$1 && [ -f assets/pdf/$1/p-01.png ] || pdftoppm -r 110 -png assets/pdf/$1.pdf assets/pdf/$1/p
done

echo "== Video: Big Buck Bunny (CC BY 3.0, Blender Foundation), first 5 minutes in 5 s clips"
mkdir -p assets/video/clips assets/video/f1
[ -f assets/video/bbb.mp4 ] || curl -fsSL -o assets/video/bbb.mp4 https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4
if [ ! -f assets/video/clips/c059.mp4 ]; then
  ffmpeg -loglevel error -y -i assets/video/bbb.mp4 -t 300 -vf scale=480:-2 -an -c:v libx264 -preset veryfast -crf 28 \
    -f segment -segment_time 5 -reset_timestamps 1 -force_key_frames "expr:gte(t,n_forced*5)" assets/video/clips/c%03d.mp4
fi
for f in assets/video/clips/c*.mp4; do b=$(basename $f .mp4); mkdir -p assets/video/f1/$b
  [ -f assets/video/f1/$b/1.jpg ] || ffmpeg -loglevel error -y -i $f -vf fps=1,scale=480:-2 assets/video/f1/$b/%d.jpg; done

echo "== FLEURS real speech (en, hi, ar, ml test splits, ~1.9 GB)"
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download as d
for c in ["en_us", "hi_in", "ar_eg", "ml_in"]:
    print(d("google/fleurs", f"{c}/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet", local_dir="assets/fleurs"))
PY

echo "== Spoken and translated queries"
.venv/bin/python make_assets.py
echo "Setup done. Next: export GOOGLE_API_KEY=... && scripts/run_all.sh"
