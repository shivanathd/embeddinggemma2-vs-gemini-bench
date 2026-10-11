#!/usr/bin/env bash
# Full benchmark run (~3-5 h on an M-series Mac, ~$5 of Gemini API usage). Re-runs are cheap: embeddings are cached in cache/.
# The task order below matters: the large corpora are subsampled with one seeded RNG, and the keyword and
# extended suites reproduce the samples by replaying the same order.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY (a Gemini API key)}"
mkdir -p logs results
PY=.venv/bin/python; SRV=llamacpp/llama-b11541/llama-server; TS=$(date +%Y%m%d-%H%M%S)
log() { echo "run start $(date) load $(sysctl -n vm.loadavg)" > "logs/$1-$TS.log"; }

$SRV -m gguf/embeddinggemma-2-Q8_0.gguf --embedding -ngl 99 -c 32768 -np 4 -b 8192 -ub 8192 --port 8090 --host 127.0.0.1 >/dev/null 2>&1 &
$SRV -m gguf/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding -ngl 99 -c 32768 -np 4 -b 8192 -ub 8192 --port 8091 --host 127.0.0.1 >/dev/null 2>&1 &
trap 'pkill -f "llama-server.*--port 809[01]" || true' EXIT
until curl -s 127.0.0.1:8090/health | grep -q ok && curl -s 127.0.0.1:8091/health | grep -q ok; do sleep 2; done

log text;   $PY -u suite_text.py legal | tee -a logs/text-$TS.log
            $PY -u suite_text.py banking77 stsb tatoeba codesearch nfcorpus scifact quora stackoverflow fiqa resume | tee -a logs/text-$TS.log
log bm25;   $PY -u suite_bm25.py | tee -a logs/bm25-$TS.log
log media;  $PY -u suite_media.py photos multilingual_photos sounds speech pdf latency | tee -a logs/media-$TS.log
log speed;  $PY -u suite_speed.py | tee -a logs/speed-$TS.log
log more;   $PY -u suite_more.py hybrid clustering calibration dedupe long_docs baselines mixed_input video fleurs | tee -a logs/more-$TS.log
pkill -f "llama-server.*--port 809[01]" || true; sleep 2
log q4;     $PY -u suite_more.py q4_cpu | tee -a logs/q4-$TS.log
log ann;    OMP_NUM_THREADS=8 $PY -u ann_scale.py | tee -a logs/ann-$TS.log
            OMP_NUM_THREADS=8 $PY -u ann_real.py | tee -a logs/ann-$TS.log
$PY build_report.py report/index.html
echo "Done: open report/index.html"
