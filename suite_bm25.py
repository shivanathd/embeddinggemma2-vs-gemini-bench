"""BM25 keyword-search baseline on the exact same samples as suite_text.py (reproduces its RNG order)."""
import json, os, numpy as np, bm25s, Stemmer
import suite_text as S

def run(name, repo, max_q, max_docs=None, **kw):
    corpus, queries, qrels = S.load_beir(repo, **kw)
    qids = sorted(q for q in qrels if q in queries); S.rng.shuffle(qids); qids = sorted(qids[:max_q])
    if max_docs and len(corpus) > max_docs:
        keep = {d for q in qids for d in qrels[q] if d in corpus}
        rest = [d for d in corpus if d not in keep]; S.rng.shuffle(rest); keep |= set(rest[:max_docs - len(keep)])
        corpus = {d: corpus[d] for d in sorted(keep)}
    qrels = {q: {d for d in qrels[q] if d in corpus} for q in qids}; qids = [q for q in qids if qrels[q]]
    dids = list(corpus); st = Stemmer.Stemmer("english")
    r = bm25s.BM25(); r.index(bm25s.tokenize([corpus[d][:S.MAXCH] for d in dids], stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    res, _ = r.retrieve(bm25s.tokenize([queries[q] for q in qids], stopwords="en", stemmer=st, show_progress=False), k=10, show_progress=False)
    nd = []
    for i, q in enumerate(qids):
        hits = [dids[j] in qrels[q] for j in res[i]]
        dcg = sum(1/np.log2(k+2) for k, h in enumerate(hits) if h); idcg = sum(1/np.log2(k+2) for k in range(min(len(qrels[q]), 10)))
        nd.append(dcg/idcg)
    print(name, len(dids), len(qids), round(float(np.mean(nd)), 4), flush=True)
    return round(float(np.mean(nd)), 4)

out = {}
# same call order as the main run: codesearch, nfcorpus, scifact, quora, stackoverflow, fiqa
out["codesearch"] = run("csn", "mteb/CodeSearchNetRetrieval", 1000, parquet_prefix="python-")
out["nfcorpus"] = run("nfcorpus", "mteb/nfcorpus", 323)
out["scifact"] = run("sf2", "mteb/scifact", 300)
out["quora"] = run("quora", "mteb/quora", 300, 10000)
out["stackoverflow"] = run("so", "mteb/StackOverflowQA", 300, 8000, parquet_prefix="")
out["fiqa"] = run("fiqa", "mteb/fiqa", 300, 10000)
S.rng.seed(7)  # legal ran first in its own process
out["legal"] = run("legal", "mteb/legalbench_consumer_contracts_qa", 400)
json.dump(out, open("results/bm25.json", "w"), indent=1)
