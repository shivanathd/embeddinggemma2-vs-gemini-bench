"""Text query latency + throughput at concurrency 1/4/16: llama.cpp Q8_0 vs gemini-embedding-2. -> results/llamacpp.json"""
import json, time, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from bench import cloud_embed
U = "http://127.0.0.1:8090/v1/embeddings"
def loc(texts):
    r = urllib.request.urlopen(urllib.request.Request(U, json.dumps({"input": texts}).encode(), {"content-type": "application/json"}))
    return json.load(r)
c = load_dataset("mteb/scifact", "corpus")["corpus"]
docs = [f"title: {t} | text: {x}" for t, x in zip(c["title"][:800], c["text"][:800])]
qs = ["task: search result | query: " + q for q in load_dataset("mteb/scifact", "queries")["queries"]["text"][:200]]
out = {}
for _ in range(5): loc(qs[:4])
lat = []
for q in qs[:60]:
    t = time.perf_counter(); loc([q]); lat.append(time.perf_counter() - t)
out["text_p50_loc"], out["text_p95_loc"] = float(np.median(lat) * 1000), float(np.percentile(lat, 95) * 1000)
lat = []
for q in qs[:30]:
    t = time.perf_counter(); cloud_embed([q[30:]], "RETRIEVAL_QUERY", 768); lat.append(time.perf_counter() - t)
out["text_p50_cloud"], out["text_p95_cloud"] = float(np.median(lat) * 1000), float(np.percentile(lat, 95) * 1000)
for conc in (1, 4, 16):
    t = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex: list(ex.map(lambda q: loc([q]), qs[:160]))
    out[f"qps_loc_c{conc}"] = 160 / (time.perf_counter() - t)
    t = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex: list(ex.map(lambda q: cloud_embed([q[30:]], "RETRIEVAL_QUERY", 768), qs[:80]))
    out[f"qps_cloud_c{conc}"] = 80 / (time.perf_counter() - t)
    print(conc, round(out[f"qps_loc_c{conc}"], 1), round(out[f"qps_cloud_c{conc}"], 1), flush=True)
t = time.perf_counter()
with ThreadPoolExecutor(4) as ex: list(ex.map(loc, [docs[i:i+16] for i in range(0, 800, 16)]))
out["index_docs_per_s_loc"] = 800 / (time.perf_counter() - t)
t = time.perf_counter(); cloud_embed([d[22:] for d in docs], "RETRIEVAL_DOCUMENT", 768)
out["index_docs_per_s_cloud"] = 800 / (time.perf_counter() - t)
import subprocess; out["load_avg"] = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
json.dump(out, open("results/llamacpp.json", "w"), indent=1); print(json.dumps(out, indent=1))
