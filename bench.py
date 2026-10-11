"""EmbeddingGemma 2 (local, Apple Silicon) vs gemini-embedding-2 (cloud).

Usage: .venv/bin/python bench.py [text|image|all]
"""
import io, json, os, sys, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from datasets import load_dataset

assert os.environ.get("GOOGLE_API_KEY"), "Set GOOGLE_API_KEY (a Gemini API key) before running"
from google import genai
from google.genai import types

CACHE = os.path.join(os.path.dirname(__file__), "cache"); os.makedirs(CACHE, exist_ok=True)
DIMS = (768, 512, 256, 128)
gc = genai.Client()


def norm(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
def trunc(x, d): return norm(x[:, :d])
def ms(xs): return f"p50 {st.median(xs)*1000:.0f} ms, p95 {np.percentile(xs, 95)*1000:.0f} ms"


def cached(name, fn):
    p = os.path.join(CACHE, name + ".npy")
    if os.path.exists(p): return np.load(p), None
    t = time.time(); v = fn(); dt = time.time() - t
    np.save(p, v); return v, dt


def load_local(**cfg):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("google/embeddinggemma-2", device="mps",
                               model_kwargs={"dtype": torch.bfloat16}, config_kwargs=cfg)


def cloud_embed(parts, task=None, dim=3072, model="gemini-embedding-2"):
    """parts: list of str or types.Part. One embedding per item (each wrapped in its own Content)."""
    cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=dim)

    def call(chunk):
        for attempt in range(6):
            try:
                r = gc.models.embed_content(model=model, config=cfg,
                    contents=[types.Content(parts=[p if isinstance(p, types.Part) else types.Part(text=p)]) for p in chunk])
                return [e.values for e in r.embeddings]
            except Exception as e:
                if attempt == 5: raise
                time.sleep(2 ** attempt)
    chunks = [parts[i:i + 100] for i in range(0, len(parts), 100)]
    with ThreadPoolExecutor(8) as ex:
        out = [v for c in ex.map(call, chunks) for v in c]
    return norm(np.array(out, dtype=np.float32))


def ndcg10(Q, D, qids, dids, qrels):
    S = Q @ D.T; top = np.argsort(-S, axis=1)[:, :10]; out = []
    for i, q in enumerate(qids):
        rel = qrels[q]
        dcg = sum(1 / np.log2(r + 2) for r, j in enumerate(top[i]) if dids[j] in rel)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), 10)))
        out.append(dcg / idcg)
    return float(np.mean(out))


def text_bench():
    print("\n=== TEXT: SciFact (BEIR) ===")
    corpus = load_dataset("mteb/scifact", "corpus")["corpus"]
    queries = {r["_id"]: r["text"] for r in load_dataset("mteb/scifact", "queries")["queries"]}
    qrels = {}
    for r in load_dataset("mteb/scifact", "default")["test"]:
        if r["score"] > 0: qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    qids = sorted(qrels); dids = [str(x) for x in corpus["_id"]]
    qtext = [queries[q] for q in qids]
    print(f"{len(dids)} docs, {len(qids)} queries")

    m = load_local(vision_config=None, audio_config=None)  # text-only = 270M
    m.encode(["warmup"] * 8, prompt_name="SearchQuery")
    loc_docs = [f"title: {t or 'none'} | text: {x}" for t, x in zip(corpus["title"], corpus["text"])]
    LD, ldt = cached("loc_scifact_docs", lambda: m.encode(loc_docs, batch_size=32, normalize_embeddings=True))
    LQ, _ = cached("loc_scifact_q", lambda: m.encode(qtext, prompt_name="SearchQuery", normalize_embeddings=True))
    cloud_docs = [f"{t}\n{x}" if t else x for t, x in zip(corpus["title"], corpus["text"])]
    CD, cdt = cached("cloud_scifact_docs", lambda: cloud_embed(cloud_docs, "RETRIEVAL_DOCUMENT"))
    CQ, _ = cached("cloud_scifact_q", lambda: cloud_embed(qtext, "RETRIEVAL_QUERY"))

    print("\nnDCG@10 by dimension:")
    print(f"{'dim':>6} {'EmbeddingGemma2':>16} {'gemini-emb-2':>14}")
    for d in (3072, 1536) + DIMS:
        l = f"{ndcg10(trunc(LQ, d), trunc(LD, d), qids, dids, qrels):.4f}" if d <= 768 else "-"
        print(f"{d:>6} {l:>16} {ndcg10(trunc(CQ, d), trunc(CD, d), qids, dids, qrels):>14.4f}")

    if ldt: print(f"\nIndexing {len(dids)} docs: local {ldt:.1f}s ({len(dids)/ldt:.0f} docs/s)")
    if cdt: print(f"Indexing {len(dids)} docs: cloud {cdt:.1f}s ({len(dids)/cdt:.0f} docs/s, 8 parallel x 100/req)")
    nchar = sum(map(len, cloud_docs)); print(f"Corpus ~{nchar/4/1e6:.2f}M tokens -> cloud cost ~${nchar/4/1e6*0.20:.2f} (at $0.20/1M)")

    sample = qtext[:40]
    lat = []
    for q in sample:
        t = time.perf_counter(); m.encode(q, prompt_name="SearchQuery"); torch.mps.synchronize(); lat.append(time.perf_counter() - t)
    print(f"\nSingle-query latency local (MPS bf16): {ms(lat)}")
    mc = load_local(vision_config=None, audio_config=None).to("cpu").float()
    mc.encode("warmup"); lat = []
    for q in sample[:20]:
        t = time.perf_counter(); mc.encode(q, prompt_name="SearchQuery"); lat.append(time.perf_counter() - t)
    print(f"Single-query latency local (CPU fp32): {ms(lat)}")
    lat = []
    for q in sample[:20]:
        t = time.perf_counter(); cloud_embed([q], "RETRIEVAL_QUERY", 768); lat.append(time.perf_counter() - t)
    print(f"Single-query latency cloud:            {ms(lat)}")
    t = time.perf_counter(); s = LQ @ LD.T; np.argsort(-s, axis=1)
    print(f"Brute-force search {len(qids)}x{len(dids)} @768d: {(time.perf_counter()-t)*1000:.0f} ms total")


def image_bench(n=300):
    print(f"\n=== IMAGE: Flickr30k test, {n} images ===")
    rows = list(load_dataset("lmms-lab/flickr30k", split="test", streaming=True).take(n))
    imgs = [r["image"].convert("RGB") for r in rows]
    cap0 = [r["caption"][0] for r in rows]  # in the corpus
    cap1 = [r["caption"][1] for r in rows]  # used as query

    m = load_local(audio_config=None)  # text+vision = 440M
    m.encode([{"image": imgs[0]}])
    LI, lit = cached(f"loc_img{n}", lambda: m.encode([{"image": im} for im in imgs], batch_size=8, normalize_embeddings=True))
    LC, _ = cached(f"loc_cap0_{n}", lambda: m.encode(cap0, prompt_name="Document", normalize_embeddings=True))
    LQ, _ = cached(f"loc_cap1_{n}", lambda: m.encode(cap1, prompt_name="SearchQuery", normalize_embeddings=True))

    def jpg(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=90); return types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg")
    CI, cit = cached(f"cloud_img{n}", lambda: cloud_embed([jpg(im) for im in imgs], "RETRIEVAL_DOCUMENT", 768))
    CC, _ = cached(f"cloud_cap0_{n}", lambda: cloud_embed(cap0, "RETRIEVAL_DOCUMENT", 768))
    CQ, _ = cached(f"cloud_cap1_{n}", lambda: cloud_embed(cap1, "RETRIEVAL_QUERY", 768))

    def recall(S, k): return float(np.mean([i in np.argsort(-S[i])[:k] for i in range(n)]))
    def report(name, Q, I, C):
        print(f"\n[{name}]")
        S = Q @ I.T; print(f"  text->image  R@1 {recall(S,1):.3f}  R@5 {recall(S,5):.3f}")
        S = I @ C.T; print(f"  image->text  R@1 {recall(S,1):.3f}  R@5 {recall(S,5):.3f}")
        # Modality bias: mixed corpus of n images + n captions.
        M = np.vstack([I, C]); is_img = np.r_[np.ones(n, bool), np.zeros(n, bool)]
        top = np.argsort(-(Q @ M.T), axis=1)[:, :10]
        print(f"  text query, mixed corpus: top-10 that are images {is_img[top].mean():.0%} (fair = 50%)")
        Sx = I @ M.T; Sx[np.arange(n), np.arange(n)] = -9  # drop the query image itself
        top = np.argsort(-Sx, axis=1)[:, :10]
        own_cap_rank = [int(np.where(np.argsort(-Sx[i]) == n + i)[0][0]) + 1 for i in range(n)]
        print(f"  image query, mixed corpus: top-10 that are images {is_img[top].mean():.0%}; "
              f"its own caption ranks median #{int(np.median(own_cap_rank))}, in top-10 {np.mean(np.array(own_cap_rank)<=10):.0%}")
    report("EmbeddingGemma 2 local @768", LQ, LI, LC)
    report("gemini-embedding-2 cloud @768", CQ, CI, CC)
    if lit: print(f"\nImage indexing local: {n/lit:.1f} img/s")
    if cit: print(f"Image indexing cloud: {n/cit:.1f} img/s")
    lat = []
    for im in imgs[:15]:
        t = time.perf_counter(); m.encode([{"image": im}]); torch.mps.synchronize(); lat.append(time.perf_counter() - t)
    print(f"Single image-query latency local: {ms(lat)}")
    lat = []
    for im in imgs[:10]:
        t = time.perf_counter(); cloud_embed([jpg(im)], None, 768); lat.append(time.perf_counter() - t)
    print(f"Single image-query latency cloud: {ms(lat)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("text", "all"): text_bench()
    if which in ("image", "all"): image_bench()
