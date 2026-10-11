"""Text use-case suite: EmbeddingGemma 2 (llama.cpp Q8_0, local) vs gemini-embedding-2 (cloud).

Every task caches embeddings in cache/, writes metrics + paired-bootstrap CIs to results/text.json.
Usage: .venv/bin/python suite_text.py [task ...]
"""
import gzip, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from bench import cloud_embed, norm, trunc, CACHE

LLAMA = "http://127.0.0.1:8090/v1/embeddings"
MAXCH = 16000  # ~4k tokens; applied to both models
RES = os.path.join(os.path.dirname(__file__), "results", "text.json")
rng = random.Random(7)

P_LOCAL = {  # EmbeddingGemma 2 model-card prefixes
    "query": "task: search result | query: ", "code": "task: code retrieval | query: ",
    "qa": "task: question answering | query: ", "doc": "title: none | text: ",
    "cls": "task: classification | query: ", "sts": "task: sentence similarity | query: ",
}
QWEN = "http://127.0.0.1:8091/v1/embeddings"
P_QWEN = {  # Qwen3-Embedding: instruction on queries only, documents raw
    "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "code": "Instruct: Given a question, retrieve code or answers that solve it\nQuery:",
    "qa": "Instruct: Given a question, retrieve passages that answer it\nQuery:",
    "doc": "", "cls": "Instruct: Classify the intent of the customer message\nQuery:",
    "sts": "Instruct: Retrieve semantically similar text\nQuery:",
}
P_CLOUD = {"query": "RETRIEVAL_QUERY", "code": "CODE_RETRIEVAL_QUERY", "qa": "QUESTION_ANSWERING",
           "doc": "RETRIEVAL_DOCUMENT", "cls": "CLASSIFICATION", "sts": "SEMANTIC_SIMILARITY"}


def local_embed(texts, url=LLAMA):
    def call(chunk):
        body = json.dumps({"input": chunk}).encode()
        for a in range(5):
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, body, {"content-type": "application/json"}), timeout=600)
                return [d["embedding"] for d in json.load(r)["data"]]
            except Exception:
                if a == 4: raise
                time.sleep(2)
    chunks = [texts[i:i + 16] for i in range(0, len(texts), 16)]
    with ThreadPoolExecutor(4) as ex:
        out = [v for c in ex.map(call, chunks) for v in c]
    return norm(np.array(out, np.float32))


def embed(name, model, texts, kind, title=None):
    """Cached embedding. Local gets a text prefix; cloud gets a task_type."""
    p = os.path.join(CACHE, f"{model}_{name}.npy")
    if os.path.exists(p): return np.load(p), None
    texts = [t[:MAXCH] for t in texts]
    t0 = time.time()
    if model == "loc":
        v = local_embed([P_LOCAL[kind] + t for t in texts])
    elif model == "qwen":
        v = local_embed([P_QWEN[kind] + t for t in texts], QWEN)
    elif model == "g1":
        v = cloud_embed([t if t.strip() else "(empty)" for t in texts], P_CLOUD[kind], model="gemini-embedding-001")
    else:
        v = cloud_embed([t if t.strip() else "(empty)" for t in texts], P_CLOUD[kind])
    dt = time.time() - t0
    np.save(p, v)
    print(f"  embedded {len(texts)} x {model} [{kind}] in {dt:.1f}s ({len(texts)/dt:.0f}/s)", flush=True)
    return v, dt


def boot(a, b, n=2000):
    """Paired bootstrap of mean(b) - mean(a): returns (diff, lo, hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float); r = np.random.default_rng(0)
    idx = r.integers(0, len(a), (n, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def per_query_ndcg(Q, D, qids, dids, qrels, k=10):
    S = Q @ D.T; top = np.argsort(-S, axis=1)[:, :k]; nd, rr, rc = [], [], []
    for i, q in enumerate(qids):
        rel = qrels[q]; hits = [dids[j] in rel for j in top[i]]
        dcg = sum(1 / np.log2(r + 2) for r, h in enumerate(hits) if h)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), k)))
        nd.append(dcg / idcg); rr.append(next((1 / (r + 1) for r, h in enumerate(hits) if h), 0.0))
        rc.append(sum(hits) / min(len(rel), k))
    return np.array(nd), np.array(rr), np.array(rc)


def jsonl(repo, f):
    p = hf_hub_download(repo, f, repo_type="dataset"); op = gzip.open if f.endswith(".gz") else open
    with op(p, "rt") as fh: return [json.loads(l) for l in fh]


def pq(repo, f): return pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"))


def load_beir(repo, split="test", parquet_prefix=None):
    if parquet_prefix is None:
        corpus = {r["_id"]: ((r.get("title") or "") + "\n" + r["text"]).strip() for r in jsonl(repo, "corpus.jsonl")}
        queries = {r["_id"]: r["text"] for r in jsonl(repo, "queries.jsonl")}
        qr = jsonl(repo, f"qrels/{split}.jsonl")
    else:
        c = pq(repo, f"{parquet_prefix}corpus/test-00000-of-00001.parquet")
        idc = "_id" if "_id" in c else "id"
        corpus = {str(i): ((t or "") + "\n" + x).strip() for i, t, x in zip(c[idc], c["title"], c["text"])}
        qd = pq(repo, f"{parquet_prefix}queries/test-00000-of-00001.parquet")
        queries = {str(i): t for i, t in zip(qd["_id" if "_id" in qd else "id"], qd["text"])}
        qr = pq(repo, f"{parquet_prefix}qrels/test-00000-of-00001.parquet").to_dict("records")
    qrels = {}
    for r in qr:
        if float(r["score"]) > 0: qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    return corpus, queries, qrels


def retrieval(name, repo, use_case, qkind="query", max_q=300, max_docs=None, **kw):
    corpus, queries, qrels = load_beir(repo, **kw)
    qids = sorted(q for q in qrels if q in queries)
    rng.shuffle(qids); qids = sorted(qids[:max_q])
    if max_docs and len(corpus) > max_docs:  # keep every judged doc, fill with random distractors
        keep = {d for q in qids for d in qrels[q] if d in corpus}
        rest = [d for d in corpus if d not in keep]; rng.shuffle(rest)
        keep |= set(rest[:max_docs - len(keep)])
        corpus = {d: corpus[d] for d in sorted(keep)}
    qrels = {q: {d for d in qrels[q] if d in corpus} for q in qids}
    qids = [q for q in qids if qrels[q]]
    dids = list(corpus)
    print(f"[{name}] {len(dids)} docs, {len(qids)} queries", flush=True)
    out = {"use_case": use_case, "dataset": repo, "docs": len(dids), "queries": len(qids), "metric": "nDCG@10"}
    E = {}
    for m in ("loc", "cloud"):
        D, ddt = embed(f"{name}_docs", m, [corpus[d] for d in dids], "doc")
        Q, _ = embed(f"{name}_q", m, [queries[q] for q in qids], qkind)
        E[m] = (Q, D)
        if ddt: out[f"{m}_index_docs_per_s"] = round(len(dids) / ddt, 1)
    for d in (768, 512, 256, 128):
        for m in ("loc", "cloud"):
            nd, rr, rc = per_query_ndcg(trunc(E[m][0], d), trunc(E[m][1], d), qids, dids, qrels)
            out[f"{m}_{d}"] = {"ndcg10": round(nd.mean(), 4), "mrr10": round(rr.mean(), 4), "recall10": round(rc.mean(), 4)}
            if d == 768: out[f"_pq_{m}"] = nd
    nd, _, _ = per_query_ndcg(E["cloud"][0], E["cloud"][1], qids, dids, qrels)
    out["cloud_3072"] = {"ndcg10": round(nd.mean(), 4)}
    out["diff_768"] = boot(out.pop("_pq_loc"), out.pop("_pq_cloud"))
    return out


def banking77():
    tr = pq("mteb/banking77", "data/train-00000-of-00001.parquet"); te = pq("mteb/banking77", "data/test-00000-of-00001.parquet")
    tr = tr.groupby("label").sample(16, random_state=7)
    te = te.sample(1000, random_state=7)
    print(f"[banking77] train {len(tr)} (16/intent), test {len(te)}", flush=True)
    out = {"use_case": "Support ticket routing (77 intents)", "dataset": "mteb/banking77", "metric": "accuracy",
           "train": len(tr), "test": len(te)}
    acc = {}
    for m in ("loc", "cloud"):
        Xtr, _ = embed("b77_train", m, tr.text.tolist(), "cls"); Xte, _ = embed("b77_test", m, te.text.tolist(), "cls")
        for d in (768, 256):
            clf = LogisticRegression(max_iter=2000).fit(trunc(Xtr, d), tr.label)
            ok = clf.predict(trunc(Xte, d)) == te.label.values
            out[f"{m}_{d}"] = {"accuracy": round(ok.mean(), 4)}
            if d == 768: acc[m] = ok
    out["diff_768"] = boot(acc["loc"], acc["cloud"])
    return out


def sts():
    rows = jsonl("mteb/stsbenchmark-sts", "test.jsonl.gz")
    s1, s2, g = [r["sentence1"] for r in rows], [r["sentence2"] for r in rows], np.array([r["score"] for r in rows])
    print(f"[stsb] {len(rows)} pairs", flush=True)
    out = {"use_case": "Near-duplicate / similarity scoring", "dataset": "mteb/stsbenchmark-sts", "metric": "Spearman", "pairs": len(rows)}
    sims = {}
    for m in ("loc", "cloud"):
        A, _ = embed("sts_a", m, s1, "sts"); B, _ = embed("sts_b", m, s2, "sts")
        for d in (768, 256):
            s = (trunc(A, d) * trunc(B, d)).sum(1)
            out[f"{m}_{d}"] = {"spearman": round(spearmanr(s, g).statistic, 4)}
            if d == 768: sims[m] = s
    r = np.random.default_rng(0); ds = []
    for _ in range(1000):
        i = r.integers(0, len(g), len(g))
        ds.append(spearmanr(sims["cloud"][i], g[i]).statistic - spearmanr(sims["loc"][i], g[i]).statistic)
    out["diff_768"] = (out["cloud_768"]["spearman"] - out["loc_768"]["spearman"], float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))
    return out


def resume_fit():
    df = pd.read_csv(hf_hub_download("cnamuangtoun/resume-job-description-fit", "test.csv", repo_type="dataset"))
    jds = sorted(df.job_description_text.unique()); jd_i = {j: i for i, j in enumerate(jds)}
    print(f"[resume] {len(df)} resume/job pairs, {len(jds)} unique jobs", flush=True)
    out = {"use_case": "HR: resume vs job description fit", "dataset": "cnamuangtoun/resume-job-description-fit",
           "metric": "ROC-AUC (Good Fit vs No Fit)", "pairs": len(df)}
    keep = df.label.isin(["Good Fit", "No Fit"]).values; y = (df.label == "Good Fit").values[keep]
    sc = {}
    for m in ("loc", "cloud"):
        J, _ = embed("hr_jd", m, jds, "query"); R, _ = embed("hr_resume", m, df.resume_text.tolist(), "doc")
        for d in (768, 256):
            s = (trunc(R, d) * trunc(J, d)[[jd_i[j] for j in df.job_description_text]]).sum(1)
            out[f"{m}_{d}"] = {"auc": round(roc_auc_score(y, s[keep]), 4),
                               "spearman_3level": round(spearmanr(s, df.label.map({"No Fit": 0, "Potential Fit": 1, "Good Fit": 2})).statistic, 4)}
            if d == 768: sc[m] = s[keep]
    r = np.random.default_rng(0); ds = []
    for _ in range(1000):
        i = r.integers(0, len(y), len(y))
        if y[i].all() or not y[i].any(): continue
        ds.append(roc_auc_score(y[i], sc["cloud"][i]) - roc_auc_score(y[i], sc["loc"][i]))
    out["diff_768"] = (out["cloud_768"]["auc"] - out["loc_768"]["auc"], float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))
    return out


def tatoeba():
    out = {"use_case": "Multilingual: match a sentence to its English translation", "dataset": "mteb/tatoeba-bitext-mining",
           "metric": "accuracy (1000 candidates)", "langs": {}}
    allok = {"loc": [], "cloud": []}
    for lang in ("ara", "hin", "mal", "tam", "urd", "fra"):
        rows = jsonl("mteb/tatoeba-bitext-mining", f"test/{lang}-eng.jsonl.gz")
        src, eng = [r["sentence1"].strip() for r in rows], [r["sentence2"].strip() for r in rows]
        res = {"pairs": len(rows)}
        for m in ("loc", "cloud"):
            A, _ = embed(f"tat_{lang}", m, src, "sts"); B, _ = embed(f"tat_{lang}_eng", m, eng, "sts")
            ok = (A @ B.T).argmax(1) == np.arange(len(src))
            res[m] = round(ok.mean(), 4); allok[m].append(ok)
        out["langs"][lang] = res
        print(f"  tatoeba {lang}: local {res['loc']:.3f} cloud {res['cloud']:.3f}", flush=True)
    out["diff_768"] = boot(np.concatenate(allok["loc"]), np.concatenate(allok["cloud"]))
    return out


TASKS = {
    "scifact": lambda: retrieval("sf2", "mteb/scifact", "Science: claim to evidence", max_q=300),
    "nfcorpus": lambda: retrieval("nfcorpus", "mteb/nfcorpus", "Medical: health question to research", max_q=323),
    "fiqa": lambda: retrieval("fiqa", "mteb/fiqa", "Finance: investor question to answer", qkind="qa", max_q=300, max_docs=10000),
    "legal": lambda: retrieval("legal", "mteb/legalbench_consumer_contracts_qa", "Legal: question to contract clause", qkind="qa", max_q=400),
    "stackoverflow": lambda: retrieval("so", "mteb/StackOverflowQA", "Software: dev question to answer", qkind="code", max_q=300, max_docs=8000, parquet_prefix=""),
    "codesearch": lambda: retrieval("csn", "mteb/CodeSearchNetRetrieval", "Software: docstring to Python function", qkind="code", max_q=1000, parquet_prefix="python-"),
    "quora": lambda: retrieval("quora", "mteb/quora", "CRM/tickets: find the duplicate question", max_q=300, max_docs=10000),
    "banking77": banking77, "stsb": sts, "resume": resume_fit, "tatoeba": tatoeba,
}

if __name__ == "__main__":
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for t in (sys.argv[1:] or TASKS):
        t0 = time.time(); r = TASKS[t](); r["wall_s"] = round(time.time() - t0, 1)
        res[t] = r; json.dump(res, open(RES, "w"), indent=1, default=float)
        print(json.dumps({t: {k: v for k, v in r.items() if k not in ("langs",)}}, default=float), flush=True)
