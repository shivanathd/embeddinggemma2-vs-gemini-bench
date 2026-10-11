"""Media suite: photos, multilingual + spoken photo search, sounds, speech, PDF pages.
EmbeddingGemma 2 (PyTorch, MPS bf16, local) vs gemini-embedding-2 (cloud). Results -> results/media.json
Usage: .venv/bin/python suite_media.py [task ...]
"""
import io, json, os, random, re, subprocess, sys, time, glob
import numpy as np, pandas as pd, torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from google.genai import types
from bench import cloud_embed, norm, trunc, CACHE, load_local
from suite_text import boot

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "media.json")
A = os.path.join(HERE, "assets")
_m = {}


def model():
    if "m" not in _m: _m["m"] = load_local()  # full: text + vision + audio
    return _m["m"]


def loc(name, items, prompt=None, bs=8):
    p = os.path.join(CACHE, f"mm_loc_{name}.npy")
    if os.path.exists(p): return np.load(p)
    t = time.time()
    v = model().encode(items, prompt=prompt, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    print(f"  local {name}: {len(items)} in {time.time()-t:.1f}s ({len(items)/(time.time()-t):.1f}/s)", flush=True)
    np.save(p, v); return v


def cld(name, items, task):
    p = os.path.join(CACHE, f"mm_cloud_{name}.npy")
    if os.path.exists(p): return np.load(p)
    t = time.time(); v = cloud_embed(items, task, 768)
    print(f"  cloud {name}: {len(items)} in {time.time()-t:.1f}s ({len(items)/(time.time()-t):.1f}/s)", flush=True)
    np.save(p, v); return v


QP, DP = "task: search result | query: ", "title: none | text: "


def jpg(im):
    b = io.BytesIO(); im.save(b, "JPEG", quality=90); return types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg")


def png(path): return types.Part.from_bytes(data=open(path, "rb").read(), mime_type="image/png")


def wav(path): return types.Part.from_bytes(data=open(path, "rb").read(), mime_type="audio/wav")


def ranks(S, gold):
    """1-based rank of the best gold item for each query row. gold: list of sets of column ids."""
    order = np.argsort(-S, axis=1); out = []
    for i, g in enumerate(gold):
        pos = np.where(np.isin(order[i], list(g)))[0]; out.append(int(pos[0]) + 1)
    return np.array(out)


def rk(r): return {"R@1": round(float((r <= 1).mean()), 4), "R@5": round(float((r <= 5).mean()), 4), "R@10": round(float((r <= 10).mean()), 4), "MRR": round(float((1 / r).mean()), 4)}


_flickr = {}


def flickr():
    if not _flickr:
        rows = list(load_dataset("lmms-lab/flickr30k", split="test", streaming=True).take(1000))
        _flickr["imgs"] = [r["image"].convert("RGB") for r in rows]
        _flickr["caps"] = [[c.strip() for c in r["caption"][:5]] for r in rows]
    return _flickr


def photos():
    f = flickr(); imgs, caps = f["imgs"], f["caps"]; n = len(imgs)
    flat = [c for cs in caps for c in cs]; owner = np.repeat(np.arange(n), 5)
    print(f"[photos] {n} images, {len(flat)} caption queries", flush=True)
    E = {"loc": (loc("fl_img", [{"image": im} for im in imgs]), loc("fl_capq", flat, QP, 64), loc("fl_capd", flat, DP, 64)),
         "cloud": (cld("fl_img", [jpg(im) for im in imgs], "RETRIEVAL_DOCUMENT"), cld("fl_capq", flat, "RETRIEVAL_QUERY"), cld("fl_capd", flat, "RETRIEVAL_DOCUMENT"))}
    out = {"use_case": "Photo library search (Flickr30k, 1000 photos x 5 human captions)", "images": n, "queries": len(flat)}
    r1 = {}
    for m, (I, Q, Dc) in E.items():
        for d in (768, 256, 128):
            r = ranks(trunc(Q, d) @ trunc(I, d).T, [{o} for o in owner])
            out[f"{m}_t2i_{d}"] = rk(r)
            if d == 768: r1[m] = (r <= 1)
        r = ranks(I @ Dc.T, [set(range(i * 5, i * 5 + 5)) for i in range(n)]); out[f"{m}_i2t_768"] = rk(r)
        # modality bias: 1000 images + caption[0] of each, queried by caption[1] and by image
        cap0 = Dc[0::5]; M = np.vstack([I, cap0]); is_img = np.r_[np.ones(n, bool), np.zeros(n, bool)]
        S = Q[1::5] @ M.T; top = np.argsort(-S, axis=1)[:, :10]
        ri, rc = ranks(S, [{i} for i in range(n)]), ranks(S, [{n + i} for i in range(n)])
        Sx = I @ M.T; Sx[np.arange(n), np.arange(n)] = -9; topx = np.argsort(-Sx, axis=1)[:, :10]
        rcap = ranks(Sx, [{n + i} for i in range(n)])
        out[f"{m}_bias"] = {"text_q_top10_image_share": round(float(is_img[top].mean()), 3),
                            "text_q_median_rank_image": int(np.median(ri)), "text_q_median_rank_caption": int(np.median(rc)),
                            "image_q_top10_image_share": round(float(is_img[topx].mean()), 3),
                            "image_q_median_rank_own_caption": int(np.median(rcap)), "image_q_own_caption_top10": round(float((rcap <= 10).mean()), 3)}
    out["diff_t2i_R@1"] = boot(r1["loc"], r1["cloud"])
    # text-only model (270M) querying the full-model photo index
    from bench import load_local as ll
    mt = ll(vision_config=None, audio_config=None)
    qt = mt.encode(flat[:500], prompt=QP, normalize_embeddings=True, batch_size=64)
    same = (qt * E["loc"][1][:500]).sum(1)
    r = ranks(qt @ E["loc"][0].T, [{o} for o in owner[:500]])
    out["text_only_model"] = {"cosine_vs_full_model_min": round(float(same.min()), 4), "t2i_R@1_first500": rk(r)["R@1"],
                              "full_model_R@1_first500": rk(ranks(E["loc"][1][:500] @ E["loc"][0].T, [{o} for o in owner[:500]]))["R@1"]}
    del mt
    return out


def multilingual_photos():
    f = flickr(); meta = json.load(open(os.path.join(A, "mm_queries.json"))); idx = meta["idx"]; Q = meta["queries"]
    I = {"loc": loc("fl_img", None), "cloud": cld("fl_img", None, None)}
    out = {"use_case": "Photo search in 6 languages, typed and spoken (50 queries each, 1000 photos)", "typed": {}, "spoken": {}}
    for lang in ("en", "de", "es", "hi", "ar", "ml"):
        texts = [q[lang] for q in Q]; res = {}
        for m in ("loc", "cloud"):
            E = loc(f"ml_{lang}", texts, QP, 64) if m == "loc" else cld(f"ml_{lang}", texts, "RETRIEVAL_QUERY")
            res[m] = rk(ranks(E @ I[m].T, [{i} for i in idx]))
        out["typed"][lang] = res; print(f"  typed {lang}: {res['loc']['R@1']} vs {res['cloud']['R@1']}", flush=True)
    for lang, pre in (("en", "q"), ("hi", "hi"), ("ar", "ar")):
        files = sorted(glob.glob(os.path.join(A, "speech", f"{pre}[0-9][0-9].wav"))); res = {}
        for m in ("loc", "cloud"):
            E = loc(f"sp_{lang}", [{"audio": p} for p in files], None, 8) if m == "loc" else cld(f"sp_{lang}", [wav(p) for p in files], "RETRIEVAL_QUERY")
            res[m] = rk(ranks(E @ I[m].T, [{i} for i in idx]))
        out["spoken"][lang] = res; print(f"  spoken {lang}: {res['loc']['R@1']} vs {res['cloud']['R@1']}", flush=True)
    return out


def sounds():
    df = pd.concat([pd.read_parquet(hf_hub_download("ashraq/esc50", f, repo_type="dataset")) for f in
                    ("data/train-00000-of-00002-2f1ab7b824ec751f.parquet", "data/train-00001-of-00002-27425e5c1846b494.parquet")])
    df = df.groupby("category").sample(10, random_state=7).reset_index(drop=True)
    d = os.path.join(A, "esc50"); os.makedirs(d, exist_ok=True); files = []
    for i, r in df.iterrows():
        src, dst = os.path.join(d, f"{i}.src.wav"), os.path.join(d, f"{i}.wav"); files.append(dst)
        if not os.path.exists(dst):
            open(src, "wb").write(r["audio"]["bytes"])
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", src, dst], check=True); os.remove(src)
    cats = sorted(df.category.unique()); y = np.array([cats.index(c) for c in df.category])
    labels = [f"the sound of {c.replace('_', ' ')}" for c in cats]
    print(f"[sounds] ESC-50: {len(files)} clips, {len(cats)} classes", flush=True)
    out = {"use_case": "Sound search (ESC-50: 50 everyday sounds, 10 clips each)", "clips": len(files), "classes": len(cats)}
    hit = {}
    for m in ("loc", "cloud"):
        X = loc("esc_audio", [{"audio": p} for p in files]) if m == "loc" else cld("esc_audio", [wav(p) for p in files], "RETRIEVAL_DOCUMENT")
        L = loc("esc_labels", labels, QP, 64) if m == "loc" else cld("esc_labels", labels, "RETRIEVAL_QUERY")
        S = L @ X.T; top = np.argsort(-S, axis=1)
        p1 = np.array([y[top[c, 0]] == c for c in range(len(cats))]); p10 = np.array([(y[top[c, :10]] == c).mean() for c in range(len(cats))])
        zs = (X @ L.T).argmax(1) == y
        out[m] = {"label_to_clip_top1": round(float(p1.mean()), 3), "label_to_clip_P@10": round(float(p10.mean()), 3),
                  "zero_shot_clip_classification": round(float(zs.mean()), 3)}
        hit[m] = zs
        worst = sorted(range(len(cats)), key=lambda c: p10[c])[:5]
        out[m]["weakest_classes"] = [cats[c] for c in worst]
    out["diff_zero_shot"] = boot(hit["loc"], hit["cloud"]); out["chance"] = round(1 / len(cats), 3)
    return out


def speech():
    df = pd.read_parquet(hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet", repo_type="dataset"))
    d = os.path.join(A, "libri"); os.makedirs(d, exist_ok=True); files = []
    for i, r in df.iterrows():
        src, dst = os.path.join(d, f"{i}.flac"), os.path.join(d, f"{i}.wav"); files.append(dst)
        if not os.path.exists(dst):
            open(src, "wb").write(r["audio"]["bytes"]); subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst], check=True); os.remove(src)
    text = [t.capitalize() for t in df.text]
    print(f"[speech] LibriSpeech: {len(files)} spoken passages", flush=True)
    out = {"use_case": "Voice notes: match a recording to its transcript (LibriSpeech, 73 clips)", "clips": len(files)}
    for m in ("loc", "cloud"):
        X = loc("libri_audio", [{"audio": p} for p in files]) if m == "loc" else cld("libri_audio", [wav(p) for p in files], "RETRIEVAL_DOCUMENT")
        T = loc("libri_text", text, QP, 64) if m == "loc" else cld("libri_text", text, "RETRIEVAL_QUERY")
        g = [{i} for i in range(len(files))]
        out[m] = {"text_to_audio": rk(ranks(T @ X.T, g)), "audio_to_text": rk(ranks(X @ T.T, g))}
    return out


PDF_Q = [  # (query, distinctive phrase on the target page, paper)
    ("diagram of the Transformer encoder-decoder architecture", "Figure 1: The Transformer - model architecture", "attention"),
    ("figure showing scaled dot-product attention and multi-head attention", "Figure 2: (left) Scaled Dot-Product Attention", "attention"),
    ("table comparing complexity per layer of self-attention, recurrent and convolutional layers", "Table 1: Maximum path lengths", "attention"),
    ("BLEU scores and training cost versus earlier translation models", "Table 2: The Transformer achieves better BLEU", "attention"),
    ("ablation table varying the number of attention heads and model size", "Table 3: Variations on the Transformer", "attention"),
    ("English constituency parsing results on WSJ", "Table 4: The Transformer generalizes well", "attention"),
    ("learning rate schedule with warmup steps", "warmup_steps", "attention"),
    ("BERT pre-training and fine-tuning overview diagram", "Figure 1: Overall pre-training and fine-tuning", "bert"),
    ("token, segment and position embeddings input representation", "Figure 2: BERT input representation", "bert"),
    ("GLUE benchmark results table", "Table 1: GLUE Test results", "bert"),
    ("SQuAD question answering results", "Table 2: SQuAD 1.1", "bert"),
    ("effect of model size on fine-tuning accuracy", "Table 6: Ablation over BERT model size", "bert"),
    ("residual learning building block with identity shortcut", "Figure 2. Residual learning: a building block", "resnet"),
    ("training error of 20-layer versus 56-layer plain networks on CIFAR-10", "Figure 1. Training error (left) and test error (right)", "resnet"),
    ("table of ResNet architectures for ImageNet with layer configurations", "Table 1. Architectures for ImageNet", "resnet"),
    ("VGG-19 compared with 34-layer plain and residual network diagrams", "Figure 3. Example network architectures for ImageNet", "resnet"),
    ("CIFAR-10 classification error results", "Table 6. Classification error on the CIFAR-10", "resnet"),
    ("object detection results on COCO", "COCO", "resnet"),
]


def pdf_pages():
    pages, texts = [], []
    for paper in ("attention", "bert", "resnet"):
        for pth in sorted(glob.glob(os.path.join(A, "pdf", paper, "p-*.png"))):
            n = int(re.search(r"p-(\d+)", pth).group(1)); pages.append((paper, n, pth))
            texts.append(subprocess.run(["pdftotext", "-layout", "-f", str(n), "-l", str(n), os.path.join(A, "pdf", paper + ".pdf"), "-"], capture_output=True, text=True).stdout)
    norm_t = [re.sub(r"\s+", " ", t).lower() for t in texts]
    qs, gold = [], []
    for q, phrase, paper in PDF_Q:
        g = {i for i, (p, n, _) in enumerate(pages) if p == paper and re.sub(r"\s+", " ", phrase).lower() in norm_t[i]}
        if g: qs.append(q); gold.append(g)
    print(f"[pdf] {len(pages)} page images from 3 papers, {len(qs)} queries with located answers", flush=True)
    out = {"use_case": "Find the right PDF page from page images (3 papers, 43 pages)", "pages": len(pages), "queries": len(qs)}
    for m in ("loc", "cloud"):
        P = loc("pdf_pages", [{"image": p} for _, _, p in pages], None, 4) if m == "loc" else cld("pdf_pages", [png(p) for _, _, p in pages], "RETRIEVAL_DOCUMENT")
        Q = loc("pdf_q", qs, QP, 64) if m == "loc" else cld("pdf_q", qs, "RETRIEVAL_QUERY")
        T = loc("pdf_text", [t[:12000] for t in texts], DP, 8) if m == "loc" else cld("pdf_text", [t[:12000] or "(empty)" for t in texts], "RETRIEVAL_DOCUMENT")
        out[m] = {"page_image": rk(ranks(Q @ P.T, gold)), "extracted_text": rk(ranks(Q @ T.T, gold))}
    return out


def latency():
    f = flickr(); m = model(); out = {}
    for _ in range(3): m.encode([{"image": f["imgs"][0]}])
    for name, items, mk in (("image_query", f["imgs"][:12], lambda x: {"image": x}),
                            ("audio_query", sorted(glob.glob(os.path.join(A, "speech", "q*.wav")))[:12], lambda x: {"audio": x})):
        lt, ct = [], []
        for x in items:
            t = time.perf_counter(); m.encode([mk(x)]); torch.mps.synchronize(); lt.append(time.perf_counter() - t)
        for x in items[:8]:
            part = jpg(x) if name == "image_query" else wav(x)
            t = time.perf_counter(); cloud_embed([part], None, 768); ct.append(time.perf_counter() - t)
        out[name] = {"loc_p50_ms": round(np.median(lt) * 1000), "loc_p95_ms": round(np.percentile(lt, 95) * 1000),
                     "cloud_p50_ms": round(np.median(ct) * 1000), "cloud_p95_ms": round(np.percentile(ct, 95) * 1000)}
        print(f"  {name}: {out[name]}", flush=True)
    return out


TASKS = {"photos": photos, "multilingual_photos": multilingual_photos, "sounds": sounds, "speech": speech, "pdf": pdf_pages, "latency": latency}

if __name__ == "__main__":
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for t in (sys.argv[1:] or TASKS):
        t0 = time.time(); r = TASKS[t](); r["wall_s"] = round(time.time() - t0, 1)
        res[t] = r; json.dump(res, open(RES, "w"), indent=1, default=float)
        print(json.dumps({t: r}, default=float)[:1500], flush=True)
