"""HNSW recall on real EmbeddingGemma 2 vectors (all cached local document embeddings)."""
import glob, json, time, numpy as np, faiss
X = np.vstack([np.load(f) for f in sorted(glob.glob("cache/loc_*_docs.npy")) if np.load(f).shape[1] == 768]).astype(np.float32)
Q = np.vstack([np.load(f) for f in sorted(glob.glob("cache/loc_*_q.npy")) if np.load(f).shape[1] == 768]).astype(np.float32)[:1000]
out = {"docs": int(len(X)), "queries": int(len(Q))}
for d in (768, 256):
    x = np.ascontiguousarray(X[:, :d]); q = np.ascontiguousarray(Q[:, :d]); faiss.normalize_L2(x); faiss.normalize_L2(q)
    flat = faiss.IndexFlatIP(d); flat.add(x); _, gt = flat.search(q, 10)
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 80; t = time.perf_counter(); hn.add(x); b = time.perf_counter() - t
    hn.hnsw.efSearch = 64; t = time.perf_counter(); _, I = hn.search(q, 10); ms = (time.perf_counter() - t) / len(q) * 1000
    out[f"{d}d"] = {"hnsw_recall@10": round(float(np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(len(q))])), 3), "build_s": round(b, 1), "ms_per_query": round(ms, 2)}
print(out)
r = json.load(open("results/more.json")); r.setdefault("ann_scale", {})["real"] = out; json.dump(r, open("results/more.json", "w"), indent=1)
