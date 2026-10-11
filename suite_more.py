"""Extension suite: hybrid search, clustering, duplicate thresholds, bias calibration, mixed inputs,
long documents, extra baselines (Qwen3-Embedding-0.6B, gemini-embedding-001), 4-bit/CPU build, ANN scale,
real speech (FLEURS) and video. Results -> results/more.json.  Usage: suite_more.py [task ...]
"""
import gzip, io, json, os, random, subprocess, sys, time, glob, urllib.request
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
import suite_text as S
from bench import cloud_embed, norm, trunc, CACHE
from suite_text import boot, per_query_ndcg, embed, jsonl

H = os.path.dirname(os.path.abspath(__file__))
RES = f"{H}/results/more.json"
A = f"{H}/assets"


# ---------- reproduce suite_text retrieval samples (same RNG call order as the main run)
SPECS = [("codesearch", "csn", "mteb/CodeSearchNetRetrieval", 1000, None, "code", {"parquet_prefix": "python-"}),
         ("nfcorpus", "nfcorpus", "mteb/nfcorpus", 323, None, "query", {}),
         ("scifact", "sf2", "mteb/scifact", 300, None, "query", {}),
         ("quora", "quora", "mteb/quora", 300, 10000, "query", {}),
         ("stackoverflow", "so", "mteb/StackOverflowQA", 300, 8000, "code", {"parquet_prefix": ""}),
         ("fiqa", "fiqa", "mteb/fiqa", 300, 10000, "qa", {}),
         ("legal", "legal", "mteb/legalbench_consumer_contracts_qa", 400, None, "qa", {})]
_samples = {}


def samples():
    if _samples: return _samples
    for key, name, repo, max_q, max_docs, qkind, kw in SPECS:
        if key == "legal": S.rng.seed(7)
        corpus, queries, qrels = S.load_beir(repo, **kw)
        qids = sorted(q for q in qrels if q in queries); S.rng.shuffle(qids); qids = sorted(qids[:max_q])
        if max_docs and len(corpus) > max_docs:
            keep = {d for q in qids for d in qrels[q] if d in corpus}
            rest = [d for d in corpus if d not in keep]; S.rng.shuffle(rest); keep |= set(rest[:max_docs - len(keep)])
            corpus = {d: corpus[d] for d in sorted(keep)}
        qrels = {q: {d for d in qrels[q] if d in corpus} for q in qids}; qids = [q for q in qids if qrels[q]]
        _samples[key] = dict(name=name, corpus=corpus, queries=queries, qrels=qrels, qids=qids, dids=list(corpus), qkind=qkind)
    return _samples


def bm25_rank(sm, k=100):
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english"); r = bm25s.BM25()
    r.index(bm25s.tokenize([sm["corpus"][d][:S.MAXCH] for d in sm["dids"]], stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    res, _ = r.retrieve(bm25s.tokenize([sm["queries"][q] for q in sm["qids"]], stopwords="en", stemmer=st, show_progress=False), k=k, show_progress=False)
    return res


def ndcg_from_ranking(rank, sm):
    out = []
    for i, q in enumerate(sm["qids"]):
        rel = sm["qrels"][q]; hits = [sm["dids"][j] in rel for j in rank[i][:10]]
        dcg = sum(1 / np.log2(k + 2) for k, h in enumerate(hits) if h); idcg = sum(1 / np.log2(k + 2) for k in range(min(len(rel), 10)))
        out.append(dcg / idcg)
    return np.array(out)


def hybrid():
    out = {}
    for key, sm in samples().items():
        bm = bm25_rank(sm); res = {"bm25": float(ndcg_from_ranking(bm, sm).mean())}
        for m in ("loc", "cloud"):
            Q = trunc(np.load(f"{CACHE}/{m}_{sm['name']}_q.npy"), 768); D = trunc(np.load(f"{CACHE}/{m}_{sm['name']}_docs.npy"), 768)
            dense = np.argsort(-(Q @ D.T), axis=1)[:, :100]
            fused = []
            for i in range(len(sm["qids"])):  # reciprocal rank fusion, k=60
                sc = {}
                for r, j in enumerate(dense[i]): sc[j] = sc.get(j, 0) + 1 / (60 + r + 1)
                for r, j in enumerate(bm[i]): sc[j] = sc.get(j, 0) + 1 / (60 + r + 1)
                fused.append(sorted(sc, key=lambda j: -sc[j])[:10])
            nd_d, nd_h = ndcg_from_ranking(dense, sm), ndcg_from_ranking(fused, sm)
            res[m] = float(nd_d.mean()); res[m + "_hybrid"] = float(nd_h.mean()); res[m + "_hybrid_diff"] = boot(nd_d, nd_h)
        out[key] = res; print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not k.endswith("diff")}, flush=True)
    return out


def baselines():
    """Qwen3-Embedding-0.6B (llama.cpp, port 8091) and gemini-embedding-001 on the same samples + Tatoeba."""
    out = {}
    for key, sm in samples().items():
        res = {}
        for m in ("qwen", "g1"):
            D, _ = embed(f"{sm['name']}_docs", m, [sm["corpus"][d] for d in sm["dids"]], "doc")
            Q, _ = embed(f"{sm['name']}_q", m, [sm["queries"][q] for q in sm["qids"]], sm["qkind"])
            nd, _, _ = per_query_ndcg(trunc(Q, 768) if Q.shape[1] > 768 else Q, trunc(D, 768) if D.shape[1] > 768 else D, sm["qids"], sm["dids"], sm["qrels"])
            res[m] = float(nd.mean())
            if m == "g1":
                nd3, _, _ = per_query_ndcg(Q, D, sm["qids"], sm["dids"], sm["qrels"]); res["g1_3072"] = float(nd3.mean())
        out[key] = res; print(key, res, flush=True)
    tat = {}
    for lang in ("ara", "hin", "mal", "tam", "urd", "fra"):
        rows = jsonl("mteb/tatoeba-bitext-mining", f"test/{lang}-eng.jsonl.gz")
        src, eng = [r["sentence1"].strip() for r in rows], [r["sentence2"].strip() for r in rows]; tat[lang] = {}
        for m in ("qwen", "g1"):
            Aa, _ = embed(f"tat_{lang}", m, src, "sts"); Bb, _ = embed(f"tat_{lang}_eng", m, eng, "sts")
            tat[lang][m] = float(((Aa @ Bb.T).argmax(1) == np.arange(len(src))).mean())
        print("tatoeba", lang, tat[lang], flush=True)
    out["tatoeba"] = tat
    return out


def clustering():
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score
    te = S.pq("mteb/banking77", "data/test-00000-of-00001.parquet").sample(1000, random_state=7)
    y = te.label.values; out = {"dataset": "banking77 test, 1000 tickets, 77 intents", "metric": "V-measure (KMeans, k=77, 5 seeds)"}
    for m in ("loc", "cloud"):
        X = np.load(f"{CACHE}/{m}_b77_test.npy")
        for d in (768, 256):
            vs = [v_measure_score(y, KMeans(77, n_init=4, random_state=s).fit_predict(trunc(X, d))) for s in range(5)]
            out[f"{m}_{d}"] = {"v_measure": float(np.mean(vs)), "sd": float(np.std(vs))}
    print(out, flush=True); return out


def dedupe_threshold():
    from sklearn.metrics import average_precision_score, precision_recall_curve
    p = hf_hub_download("mteb/sprintduplicatequestions-pairclassification", "test.json.gz", repo_type="dataset")
    rec = json.load(gzip.open(p, "rt")); rec = rec[0] if isinstance(rec, list) else rec
    s1, s2, lab = rec["sent1"], rec["sent2"], rec["labels"]
    if isinstance(s1[0], list): s1, s2, lab = s1[0], s2[0], lab[0]
    lab = np.array(lab); r = np.random.default_rng(7)  # all duplicates + 4,000 random non-duplicates
    keep = np.r_[np.where(lab == 1)[0], r.choice(np.where(lab == 0)[0], 4000, replace=False)]; keep.sort()
    s1, s2, lab = [s1[i] for i in keep], [s2[i] for i in keep], lab[keep]
    out = {"dataset": f"Sprint duplicate questions ({len(lab)} pairs, {int(lab.sum())} duplicates)", "metric": "average precision; best-F1 cosine threshold"}
    for m in ("loc", "cloud"):
        Aa, _ = embed("sprint_a", m, s1, "sts"); Bb, _ = embed("sprint_b", m, s2, "sts")
        sc = (Aa * Bb).sum(1); pr, rc, th = precision_recall_curve(lab, sc)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9); i = int(np.argmax(f1[:-1]))
        out[m] = {"ap": float(average_precision_score(lab, sc)), "best_f1": float(f1[i]), "threshold": float(th[i]),
                  "precision": float(pr[i]), "recall": float(rc[i]),
                  "dup_cos_median": float(np.median(sc[lab == 1])), "nondup_cos_median": float(np.median(sc[lab == 0]))}
    print(out, flush=True); return out


def calibration():
    """Per-modality score standardisation on the mixed photo+caption index."""
    out = {}
    for m in ("loc", "cloud"):
        I = np.load(f"{CACHE}/mm_{m}_fl_img.npy"); Dc = np.load(f"{CACHE}/mm_{m}_fl_capd.npy"); Q = np.load(f"{CACHE}/mm_{m}_fl_capq.npy")
        n = len(I); cap0 = Dc[0::5]; M = np.vstack([I, cap0]); is_img = np.r_[np.ones(n, bool), np.zeros(n, bool)]
        res = {}
        for qname, QQ, drop_self in (("text_query", Q[1::5], False), ("photo_query", I, True)):
            Sraw = QQ @ M.T
            if drop_self: Sraw[np.arange(n), np.arange(n)] = np.nan
            Sz = Sraw.copy()
            for grp in (is_img, ~is_img):  # z-score each query's scores within each file type
                g = Sz[:, grp]; Sz[:, grp] = (g - np.nanmean(g, 1, keepdims=True)) / np.nanstd(g, 1, keepdims=True)
            for tag, SS in (("raw", Sraw), ("calibrated", Sz)):
                SS = np.nan_to_num(SS, nan=-9); top = np.argsort(-SS, axis=1)[:, :10]
                gold = [{n + i} if drop_self else {i, n + i} for i in range(n)]
                hit1 = np.mean([top[i, 0] in gold[i] for i in range(n)]); hit10 = np.mean([bool(set(top[i]) & gold[i]) for i in range(n)])
                res[f"{qname}_{tag}"] = {"top10_image_share": float(is_img[top].mean()), "R@1": float(hit1), "R@10": float(hit10)}
        out[m] = res; print(m, res, flush=True)
    out["note"] = "text query: either the photo or its caption counts as correct; photo query: its own caption is the target"
    return out


def mixed_input():
    from suite_media import flickr, model, jpg
    from google.genai import types
    f = flickr(); imgs, caps = f["imgs"], f["caps"]; n = len(imgs)
    out = {"dataset": "1000 Flickr30k photos; document = photo + one caption in ONE vector; query = a different caption"}
    for m in ("loc", "cloud"):
        p = f"{CACHE}/mm_{m}_fl_mixed.npy"
        if os.path.exists(p): X = np.load(p)
        elif m == "loc":
            X = model().encode([{"text": f"title: none | text: <|image|> {caps[i][0]}", "image": imgs[i]} for i in range(n)], batch_size=8, normalize_embeddings=True, show_progress_bar=False); np.save(p, X)
        else:
            def gen():
                cfg = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768); vs = []
                for i0 in range(0, n, 50):
                    r = S.cloud_embed.__globals__["gc"].models.embed_content(model="gemini-embedding-2", config=cfg,
                        contents=[types.Content(parts=[jpg(imgs[i]), types.Part(text=caps[i][0])]) for i in range(i0, min(n, i0 + 50))])
                    vs += [e.values for e in r.embeddings]
                return norm(np.array(vs, np.float32))
            X = gen(); np.save(p, X)
        Q = np.load(f"{CACHE}/mm_{m}_fl_capq.npy")[1::5]
        I = np.load(f"{CACHE}/mm_{m}_fl_img.npy"); C = np.load(f"{CACHE}/mm_{m}_fl_capd.npy")[0::5]
        def r1(D): return float(((Q @ D.T).argmax(1) == np.arange(n)).mean())
        out[m] = {"photo_only": r1(I), "caption_only": r1(C), "photo_plus_caption": r1(X)}
        print(m, out[m], flush=True)
    return out


def long_docs():
    """SciFact: hide the relevant abstract inside a ~5-abstract document at start / middle / end."""
    sm = samples()["scifact"]; rr = random.Random(3)
    rel = sorted({d for q in sm["qids"] for d in sm["qrels"][q]})
    others = [d for d in sm["dids"] if d not in set(rel)]; rr.shuffle(others)
    base = rel + others[:1000 - len(rel)]; filler = others[1000:]
    out = {"dataset": f"SciFact, {len(sm['qids'])} queries, 1000 documents of ~5 abstracts (~1,500-2,000 tokens)"}
    for pos in ("short", "start", "middle", "end"):
        docs = []
        for i, d in enumerate(base):
            fill = [sm["corpus"][filler[(i * 4 + k) % len(filler)]] for k in range(4)]
            body = {"short": [sm["corpus"][d]], "start": [sm["corpus"][d]] + fill, "middle": fill[:2] + [sm["corpus"][d]] + fill[2:], "end": fill + [sm["corpus"][d]]}[pos]
            docs.append("\n\n".join(body))
        res = {}
        for m in ("loc", "cloud"):
            D, _ = embed(f"long_{pos}_docs", m, docs, "doc"); Q = np.load(f"{CACHE}/{m}_sf2_q.npy")
            nd, _, _ = per_query_ndcg(trunc(Q, 768), trunc(D, 768), sm["qids"], base, sm["qrels"]); res[m] = float(nd.mean())
        out[pos] = res; print(pos, res, flush=True)
    return out


def q4_cpu():
    """4-bit (UD-Q4_K_XL) build on GPU for quality; Q4 on 4 CPU threads for phone-like latency."""
    srv = f"{H}/llamacpp/llama-b11541/llama-server"; out = {}
    def start(model, port, gpu, threads=None):
        args = [srv, "-m", f"{H}/gguf/{model}", "--embedding", "-ngl", "99" if gpu else "0", "-c", "32768", "-np", "4", "-b", "8192", "-ub", "8192", "--port", str(port), "--host", "127.0.0.1"]
        if threads: args += ["-t", str(threads)]
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health").read(): break
            except Exception: time.sleep(1)
        return p
    p = start("embeddinggemma-2-UD-Q4_K_XL.gguf", 8092, True)
    url = "http://127.0.0.1:8092/v1/embeddings"
    for key in ("scifact", "legal"):
        sm = samples()[key]
        D = S.local_embed([S.P_LOCAL["doc"] + sm["corpus"][d][:S.MAXCH] for d in sm["dids"]], url)
        Q = S.local_embed([S.P_LOCAL[sm["qkind"]] + sm["queries"][q] for q in sm["qids"]], url)
        nd, _, _ = per_query_ndcg(Q, D, sm["qids"], sm["dids"], sm["qrels"]); out[f"q4_{key}"] = float(nd.mean())
        print("q4", key, out[f"q4_{key}"], flush=True)
    p.terminate(); p.wait()
    sm = samples()["scifact"]; qs = [S.P_LOCAL["query"] + sm["queries"][q] for q in sm["qids"][:40]]
    for model, tag in (("embeddinggemma-2-UD-Q4_K_XL.gguf", "q4"), ("embeddinggemma-2-Q8_0.gguf", "q8")):
        p = start(model, 8093, False, 4)
        for q in qs[:5]: S.local_embed([q], "http://127.0.0.1:8093/v1/embeddings")
        lat = []
        for q in qs:
            t = time.perf_counter(); S.local_embed([q], "http://127.0.0.1:8093/v1/embeddings"); lat.append(time.perf_counter() - t)
        out[f"cpu4_{tag}_p50_ms"] = float(np.median(lat) * 1000); out[f"cpu4_{tag}_p95_ms"] = float(np.percentile(lat, 95) * 1000)
        print(tag, "cpu 4 threads", out[f"cpu4_{tag}_p50_ms"], flush=True)
        p.terminate(); p.wait()
    out["sizes_mb"] = {f: round(os.path.getsize(f"{H}/gguf/{f}") / 1e6) for f in os.listdir(f"{H}/gguf") if f.endswith(".gguf")}
    return out


def ann_scale():
    import faiss
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
    return out


def fleurs(n=150):
    """Real human read speech (FLEURS): speech -> own transcript, and speech -> English translation."""
    from suite_media import loc as mloc, cld as mcld, wav
    import soundfile as sf
    langs = {"en_us": "English", "hi_in": "Hindi", "ar_eg": "Arabic", "ml_in": "Malayalam"}
    dfs = {c: pd.read_parquet(f"{A}/fleurs/{c}/test/0000.parquet") for c in langs}
    for c in dfs: dfs[c] = dfs[c].drop_duplicates("id").set_index("id")
    common = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))[:n]
    eng = [dfs["en_us"].loc[i, "transcription"] for i in common]
    out = {"dataset": f"FLEURS test, {len(common)} sentences read aloud by real speakers in each language"}
    for c, name in langs.items():
        d = f"{A}/fleurs/wav_{c}"; os.makedirs(d, exist_ok=True); files = []
        for i in common:
            fp = f"{d}/{i}.wav"; files.append(fp)
            if not os.path.exists(fp):
                a = dfs[c].loc[i, "audio"]; data, sr = sf.read(io.BytesIO(a["bytes"]))
                tmp = fp + ".src.wav"; sf.write(tmp, data, sr)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ar", "16000", "-ac", "1", fp], check=True); os.remove(tmp)
        own = [dfs[c].loc[i, "transcription"] for i in common]; res = {}
        for m in ("loc", "cloud"):
            X = mloc(f"fl_{c}_audio", [{"audio": p} for p in files]) if m == "loc" else mcld(f"fl_{c}_audio", [wav(p) for p in files], "RETRIEVAL_QUERY")
            T = mloc(f"fl_{c}_text", own, "title: none | text: ", 64) if m == "loc" else mcld(f"fl_{c}_text", own, "RETRIEVAL_DOCUMENT")
            E = mloc("fl_eng_text", eng, "title: none | text: ", 64) if m == "loc" else mcld("fl_eng_text", eng, "RETRIEVAL_DOCUMENT")
            g = np.arange(len(common))
            res[m] = {"speech_to_own_transcript_R@1": float(((X @ T.T).argmax(1) == g).mean()),
                      "speech_to_english_R@1": float(((X @ E.T).argmax(1) == g).mean())}
        out[name] = res; print(name, res, flush=True)
    return out


VIDEO_Q = json.load(open(f"{A}/video/queries.json")) if os.path.exists(f"{A}/video/queries.json") else []


def video():
    from suite_media import loc as mloc, cld as mcld
    from google.genai import types
    clips = sorted(glob.glob(f"{A}/video/clips/c*.mp4"))
    qs = [q["q"] for q in VIDEO_Q]; gold = [set(q["clips"]) for q in VIDEO_Q]
    out = {"dataset": f"Big Buck Bunny (CC-BY Blender Foundation), first {len(clips)*5} s cut into {len(clips)} five-second clips; {len(qs)} hand-labelled queries"}
    for m in ("loc", "cloud"):
        if m == "loc":
            from PIL import Image  # 1 fps frames, the model's default video sampling (torchcodec unavailable here)
            fr = lambda p: [Image.open(x).convert("RGB") for x in sorted(glob.glob(p.replace("/clips/", "/f1/")[:-4] + "/*.jpg"), key=lambda x: int(os.path.basename(x)[:-4]))]
            V = mloc("video_clips", [{"video": fr(p)} for p in clips], None, 2); Q = mloc("video_q", qs, "task: search result | query: ", 32)
        else:
            V = mcld("video_clips", [types.Part.from_bytes(data=open(p, "rb").read(), mime_type="video/mp4") for p in clips], "RETRIEVAL_DOCUMENT")
            Q = mcld("video_q", qs, "RETRIEVAL_QUERY")
        top = np.argsort(-(Q @ V.T), axis=1)
        r1 = np.mean([top[i, 0] in gold[i] for i in range(len(qs))]); r3 = np.mean([bool(set(top[i, :3]) & gold[i]) for i in range(len(qs))])
        out[m] = {"R@1": float(r1), "R@3": float(r3)}; print(m, out[m], flush=True)
    return out


TASKS = {"hybrid": hybrid, "clustering": clustering, "dedupe": dedupe_threshold, "calibration": calibration,
         "mixed_input": mixed_input, "long_docs": long_docs, "baselines": baselines, "q4_cpu": q4_cpu,
         "ann_scale": ann_scale, "fleurs": fleurs, "video": video}

if __name__ == "__main__":
    for t in sys.argv[1:]:
        t0 = time.time(); r = TASKS[t](); r["wall_s"] = round(time.time() - t0, 1)
        res = json.load(open(RES)) if os.path.exists(RES) else {}  # re-read: other runs may have written meanwhile
        if t == "ann_scale" and "real" in res.get(t, {}): r["real"] = res[t]["real"]
        res[t] = r; json.dump(res, open(RES, "w"), indent=1, default=float)
        print(f"== {t} done in {r['wall_s']}s", flush=True)
