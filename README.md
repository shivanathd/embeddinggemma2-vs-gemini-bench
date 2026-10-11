# EmbeddingGemma 2 vs Gemini Embedding 2

**Can a free embedding model running on a laptop replace Google's cloud embedding API?**

This repository is a complete, reproducible benchmark of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (open weights, runs locally) against [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) (Google's paid API). It covers 19 business use cases, from code search and support routing to Malayalam voice queries and finding a moment in a video, plus extended experiments, every script, every result file and every raw log.

- **Full report:** open [`report/index.html`](report/index.html) in a browser. It has interactive charts, the decision guide, method notes, all source code and all logs.
- **Run it yourself:** `scripts/setup.sh`, then `scripts/run_all.sh` (Apple Silicon Mac, about 3–5 hours, about $5 of Gemini usage).
- **Watch it run:** an optional [Claude Code](https://claude.com/claude-code) mod in [`mod/bench-watch`](mod/bench-watch) shows progress and a live scoreboard.

---

## The short answer

| | Count |
|---|---:|
| Tests the local model wins | 4 |
| Tests the cloud model wins | 6 |
| Ties (gap under 3 points) | 8 |
| Neither model is usable | 1 |

**For English text, code, photos, PDFs and video, you don't need the cloud.** The local model wins or ties 12 of 19 tests. It answers a text query in about 13 ms (the cloud takes about 617 ms), costs nothing per document, and keeps every file on your machine.

**Pay for the cloud when:**
- you search in Urdu, Arabic, Tamil or Malayalam (the local model is 9–18 points behind on translation matching);
- you match questions to written answers in finance or developer support (finance: 0.508 local vs 0.719 cloud);
- you search sound effects, or let people search photos by speaking Hindi or Arabic.

**Also worth knowing:**
- If you choose the cloud for text only, test **Gemini Embedding 1** too. The older model beat Gemini Embedding 2 on finance, legal and five of six languages.
- **Chunk long documents.** A passage in the middle of a 2,000-token document loses more than half its search quality in both models.
- **Mixed photo + text indexes** lean towards the query's own file type. A few lines of per-type score calibration fix it (cloud text-query R@1 goes from 75% to 93%).
- **Don't use embeddings to judge resume fit.** Both models score close to a coin flip.

## All 19 tests

Scores are on a 0–1 scale. The 95% confidence interval comes from 2,000 paired bootstrap resamples over queries. A test counts as a win only if the interval excludes zero **and** the gap is at least 0.03.

| Department | Test | Metric | EmbeddingGemma 2 (local) | Gemini Embedding 2 (cloud) | Cloud − local (95% CI) | Verdict |
|---|---|---|---:|---:|---|---|
| Engineering | Code search: docstring to function | nDCG@10 | 0.956 | 0.969 | +0.013 (+0.006 to +0.020) | tie |
| Engineering | Developer Q&A: question to answer | nDCG@10 | 0.898 | 0.983 | +0.085 (+0.059 to +0.112) | **cloud** |
| Support & sales | Duplicate tickets: find the twin question | nDCG@10 | 0.991 | 0.992 | +0.002 (-0.004 to +0.007) | tie |
| Finance | Investor question to answer | nDCG@10 | 0.508 | 0.719 | +0.210 (+0.176 to +0.242) | **cloud** |
| Legal | Question to contract clause | nDCG@10 | 0.771 | 0.732 | -0.039 (-0.072 to -0.010) | **local** |
| Science & health | Claim to research evidence | nDCG@10 | 0.853 | 0.896 | +0.044 (+0.017 to +0.071) | **cloud** |
| Science & health | Health question to medical research | nDCG@10 | 0.354 | 0.427 | +0.073 (+0.056 to +0.091) | **cloud** |
| Support & sales | Route tickets to 77 intents | accuracy | 0.916 | 0.932 | +0.016 (+0.003 to +0.029) | tie |
| Support & sales | Score how similar two sentences are | Spearman | 0.875 | 0.771 | -0.104 (-0.122 to -0.085) | **local** |
| HR | Resume vs job description fit | ROC-AUC | 0.576 | 0.504 | -0.072 (-0.104 to -0.039) | neither works |
| Languages | Match a sentence to its English translation | accuracy | 0.866 | 0.946 | +0.078 (+0.069 to +0.087) | **cloud** |
| Field & media | Text to photo (1,000 photos) | R@1 | 0.868 | 0.889 | +0.022 (+0.013 to +0.030) | tie |
| Field & media | Classify everyday sounds (50 classes) | accuracy | 0.230 | 0.332 | +0.102 (+0.052 to +0.152) | **cloud** |
| Field & media | Voice note to its transcript | R@1 | 1.000 | 1.000 | +0.000 (no CI, small sample) | tie |
| Field & media | Find the PDF page from its image | R@1 | 0.778 | 0.611 | -0.167 (no CI, small sample) | **local** |
| Support & sales | Group tickets into topics (clustering) | V-measure | 0.912 | 0.926 | +0.014 (no CI, small sample) | tie |
| Support & sales | Flag duplicate support questions | avg precision | 0.993 | 0.995 | +0.002 (no CI, small sample) | tie |
| Languages | Real voices: speech to its transcript (4 languages) | R@1 | 0.992 | 0.992 | +0.000 (no CI, small sample) | tie |
| Field & media | Find a moment in a video | R@1 | 0.722 | 0.611 | -0.111 (no CI, small sample) | **local** |

## How to read the results

### Metrics

| Metric | Used for | What it means |
|---|---|---|
| **nDCG@10** | Document search | Quality of the top 10 results, rewarding relevant documents ranked higher. 1.0 = every relevant document at the top. |
| **R@1, R@5, R@10** (recall at k) | Photo, voice, video, PDF search | Share of queries whose correct item appears in the top 1, 5 or 10. |
| **MRR** | Search | Average of 1 / rank of the first correct result. |
| **Accuracy** | Ticket routing, translation matching, sound classification | Share of items assigned the right label or partner. |
| **Spearman** | Similarity scoring | How well the model's similarity order matches human ratings (−1 to 1). |
| **ROC-AUC** | Resume fit | Chance a good-fit resume scores above a no-fit one. 0.5 = coin flip. |
| **V-measure** | Clustering | Agreement between the model's clusters and the true topics (0–1). |
| **Average precision** | Duplicate detection | Precision averaged over all thresholds. |

### Reading the files

- `results/text.json`: the 11 text tests. For each test, `loc_768` / `cloud_768` hold the scores at 768 dimensions (also `_512`, `_256`, `_128`, and `cloud_3072`). `diff_768` is `[cloud − local, CI low, CI high]`. `*_index_docs_per_s` is indexing throughput.
- `results/media.json`: photos (`*_t2i_*` = text → photo, `*_i2t_*` = photo → text, `*_bias` = file-type bias), multilingual and spoken photo search, sounds, voice notes, PDF pages, latency.
- `results/more.json`: extended experiments (hybrid, clustering, calibration, duplicates, long documents, other models, mixed inputs, real voices, video, 4-bit/CPU, search at scale).
- `results/bm25.json`: the keyword-search baseline on the identical samples. `results/llamacpp.json`: text latency and throughput at concurrency 1, 4 and 16.
- `logs/*.log`: raw console output of every run, with the machine's load average at the start. Local paths are replaced with `~`.
- `report/index.html`: everything above, rendered.

### Verdict rule

`local` or `cloud` = the 95% CI excludes zero and |gap| ≥ 0.03. `tie` = otherwise. Tests with fewer than about 100 queries (voice notes, PDF pages, video) have no CI. Treat them as directional.

## What was tested

| Department | Test | Data |
|---|---|---|
| Engineering | Code search; developer Q&A | CodeSearchNet (Python), StackOverflowQA |
| Support & sales | Ticket routing; duplicate questions; similarity; topic clustering; duplicate thresholds | Banking77, Quora, STS-B, Sprint duplicate questions |
| Finance | Investor question → answer | FiQA |
| Legal | Consumer question → contract clause | LegalBench consumer contracts QA |
| HR | Resume vs job description fit | resume-job-description-fit |
| Science & health | Claim → evidence; health question → research | SciFact, NFCorpus |
| Languages | Sentence → English translation (Arabic, Hindi, Malayalam, Tamil, Urdu, French); real speech → transcript (English, Hindi, Arabic, Malayalam) | Tatoeba, FLEURS |
| Field & media | Text → photo; photo search in 6 languages, typed and spoken; sounds; voice notes; PDF pages from images; video moments | Flickr30k, ESC-50, LibriSpeech, arXiv papers, Big Buck Bunny |

Extended experiments: hybrid keyword + embedding search, long documents (answer at start, middle or end), photo + caption in one vector, file-type bias calibration, Qwen3-Embedding 0.6B and Gemini Embedding 1 baselines, a 4-bit build and CPU-only latency, and search speed at 1M vectors.

## Run it yourself

### Requirements

- An Apple Silicon Mac (tested on M5 Max, macOS 27). Spoken queries use the macOS `say` voices. The rest runs anywhere Python and llama.cpp run, with small path changes.
- [`uv`](https://docs.astral.sh/uv/), `curl`, `ffmpeg`, `pdftoppm` (poppler). No admin rights needed.
- A [Gemini API key](https://aistudio.google.com/apikey) on the paid tier (the free tier may use inputs to improve Google products).
- About 10 GB free disk: about 6 GB of downloads plus about 1 GB of cached embeddings.

### Rebuild the report from the committed results (no models needed)

```bash
git clone https://github.com/shivanathd/embeddinggemma2-vs-gemini-bench
cd embeddinggemma2-vs-gemini-bench
python3 -m venv .venv && .venv/bin/pip install jinja2 numpy
.venv/bin/python build_report.py report/index.html
open report/index.html
```

### Full re-run

```bash
scripts/setup.sh              # Python env, llama.cpp b11541, GGUF models, PDFs, video clips, FLEURS, spoken queries
export GOOGLE_API_KEY=...     # Gemini API key
scripts/run_all.sh            # all suites in the required order, then rebuilds report/index.html
```

`run_all.sh` starts two local llama.cpp servers: EmbeddingGemma 2 on port 8090 and Qwen3 on 8091. It runs every suite and stops the servers. **Keep the task order.** The large corpora are subsampled with one seeded random generator, and the keyword and extended suites reproduce those samples by replaying the same order.

### Run one piece

```bash
# start the local text server first
llamacpp/llama-b11541/llama-server -m gguf/embeddinggemma-2-Q8_0.gguf --embedding -ngl 99 \
  -c 32768 -np 4 -b 8192 -ub 8192 --port 8090 &
.venv/bin/python suite_text.py legal          # one text test
.venv/bin/python suite_media.py photos        # one media test (PyTorch on the GPU)
.venv/bin/python suite_more.py long_docs      # one extended experiment
```

Every embedding is cached in `cache/` under `<model>_<test>_<part>.npy`. Re-running a test only recomputes the scores. Delete a file to force a fresh embedding.

### Test your own documents

The text suite reads [BEIR](https://github.com/beir-cellar/beir)-format data: `corpus.jsonl` (`_id`, `title`, `text`), `queries.jsonl` (`_id`, `text`) and `qrels/test.jsonl` (`query-id`, `corpus-id`, `score`). Put yours in a Hugging Face dataset repo, or adapt `load_beir()` in `suite_text.py` to read local files, then add a line to `TASKS`:

```python
"mydocs": lambda: retrieval("mydocs", "you/your-dataset", "My company: question to policy", max_q=300),
```

Fifty to a few hundred labelled questions from your own content will tell you more than any public benchmark.

## Repository layout

```
bench.py              shared helpers: local model loader, cloud batching
suite_text.py         11 text tests + paired bootstrap
suite_media.py        photos, languages, voice, sounds, PDFs, latency
suite_more.py         extended experiments
suite_bm25.py         keyword baseline       suite_speed.py   latency and throughput
ann_scale.py          1M-vector search speed ann_real.py      HNSW recall on real vectors
make_assets.py        translations + macOS spoken queries
build_report.py       builds report/index.html from results/ and logs/
report_template.html  the report (Jinja2)
scripts/              setup.sh, run_all.sh
results/  logs/  report/
diagrams/             Archify sources (JSON) and rendered images
assets/               committed query files (video labels, translations); downloads land here too
mod/bench-watch/      Claude Code mod: live progress pane
```

## The live-progress mod (optional)

`mod/bench-watch` is a [Claude Code](https://claude.com/claude-code) mod. While a run is going, its pane shows:
- the running suites;
- the newest line of each log;
- a local-vs-cloud scoreboard that fills in as results land;
- the system load;
- the tool calls Claude makes.

It also adds a status-line summary and a `/bench` command.

```bash
export BENCH_DIR="$PWD"                 # where this repository lives
claude --plugin-dir ./mod/bench-watch   # then /bench to open the pane
claude plugin validate ./mod/bench-watch && claude plugin test ./mod/bench-watch
```

## Method and caveats

- **Same inputs:** both models embed exactly the same texts, images and audio, truncated at 16,000 characters. Each gets its documented task prompt: a text prefix for EmbeddingGemma 2, a `task_type` for Gemini.
- **Runtimes:** local text runs on llama.cpp (Q8_0 GGUF, Metal). Local images, audio and video run on PyTorch (MPS, bfloat16; the model card warns float16 silently produces NaNs). Q8_0, BF16 and PyTorch scored the same on SciFact.
- **Sampling:** for FiQA, Quora and StackOverflow, every judged document for 300 sampled queries is kept, then the corpus is filled to 8,000–10,000 documents with seeded random distractors.
- **Contamination:** these are public datasets that either model may have seen in training. Absolute scores are optimistic; the head-to-head comparison is the reliable part.
- **Synthetic inputs:** the spoken photo queries use macOS text-to-speech, and the translations came from Gemini 3.5 Flash. Real recordings (FLEURS) are covered separately.
- **Video:** the local model received frames at 1 per second, its default sampling rate, because `torchcodec` needs FFmpeg shared libraries.
- **Shared machine:** the load average went above 200 during parts of the run, so local throughput figures are a floor.
- **Gemini API pitfall:** a plain list of strings passed to `gemini-embedding-2` returns **one combined embedding** for the whole list. Wrap each item in its own `types.Content(parts=[...])` and send at most 100 per request. `bench.cloud_embed` does this.

**Not covered:** your own documents, noisy real-world audio (calls, messaging-app voice notes), fine-tuning, cloud rate limits and long-run stability, and a test on an actual phone.

## Data and model credits

| Data | Source and licence |
|---|---|
| SciFact, NFCorpus, FiQA, Quora, CodeSearchNet, StackOverflowQA, LegalBench consumer contracts, Banking77, STS-B, Sprint duplicates, Tatoeba | [MTEB](https://huggingface.co/mteb) mirrors on Hugging Face, each under its original licence |
| Resume and job description fit | `cnamuangtoun/resume-job-description-fit` |
| Flickr30k test | `lmms-lab/flickr30k`. Images stay their owners'; used only to compute embeddings |
| ESC-50 | `ashraq/esc50`, CC BY-NC 3.0 |
| LibriSpeech (sample), FLEURS | CC BY 4.0 |
| Big Buck Bunny | © Blender Foundation, CC BY 3.0, peach.blender.org |
| PDF pages | arXiv 1706.03762, 1810.04805, 1512.03385 |

Models: `google/embeddinggemma-2` (Apache 2.0) and GGUF builds from ggml-org and Unsloth; `Qwen/Qwen3-Embedding-0.6B` (Apache 2.0); Google Gemini API `gemini-embedding-2` and `gemini-embedding-001`. Diagrams were drawn with [Archify](https://github.com/tt-a1i/archify).

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Datasets, models and media keep their own licences.
