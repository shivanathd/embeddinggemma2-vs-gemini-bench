"""Search speed at 1M vectors (FAISS brute force vs HNSW). Runs without torch to avoid OpenMP clashes."""
import json, os, time, numpy as np, faiss
out = {}; rng = np.random.default_rng(0)
for d in (768, 256):
    X = rng.standard_normal((1_000_000, d), dtype=np.float32); faiss.normalize_L2(X)
    q = X[rng.integers(0, len(X), 200)] + 0.05 * rng.standard_normal((200, d), dtype=np.float32); faiss.normalize_L2(q)
    flat = faiss.IndexFlatIP(d); flat.add(X)
    t = time.perf_counter(); flat.search(q[:50], 10); flat_ms = (time.perf_counter() - t) / 50 * 1000
    t = time.perf_counter(); hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 80; hn.add(X); build = time.perf_counter() - t
    hn.hnsw.efSearch = 64; _, gt = flat.search(q, 10)
    t = time.perf_counter(); _, I = hn.search(q, 10); hn_ms = (time.perf_counter() - t) / len(q) * 1000
    rec = float(np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(len(q))]))
    out[f"{d}d"] = {"vectors": 1_000_000, "memory_gb": round(X.nbytes / 2**30, 2), "brute_force_ms_per_query": round(flat_ms, 1),
                    "hnsw_build_s": round(build, 1), "hnsw_ms_per_query": round(hn_ms, 2), "hnsw_recall@10": round(rec, 3)}
    print(d, out[f"{d}d"], flush=True); del X, flat, hn
p = "results/more.json"; r = json.load(open(p)); r["ann_scale"] = out; json.dump(r, open(p, "w"), indent=1)
