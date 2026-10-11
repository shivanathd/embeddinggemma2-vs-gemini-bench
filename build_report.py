"""Builds the benchmark report HTML from results/*.json and logs/*.log. Usage: build_report.py <out.html>"""
import base64, glob, html, json, os, re, sys, platform, subprocess
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(f"{H}/results/text.json"))
M = json.load(open(f"{H}/results/media.json"))
B = json.load(open(f"{H}/results/bm25.json"))
L = json.load(open(f"{H}/results/llamacpp.json")) if os.path.exists(f"{H}/results/llamacpp.json") else {}
X = json.load(open(f"{H}/results/more.json")) if os.path.exists(f"{H}/results/more.json") else {}
e = html.escape


def f3(x): return f"{x:.3f}"
def pct(x): return f"{x*100:.1f}%"


# ---------- scoreboard rows: (key, department, label, metric, local, cloud, diff, lo, hi, n, bm25)
def tat_mean(m): return float(np.mean([v[m] for v in T["tatoeba"]["langs"].values()]))


ROWS = []
def add(key, dept, label, metric, loc, cld, ci=None, n="", bm=None):
    d, lo, hi = (ci if ci else (cld - loc, None, None))
    ROWS.append(dict(key=key, dept=dept, label=label, metric=metric, loc=loc, cld=cld, d=d, lo=lo, hi=hi, n=n, bm=bm))


for k, dept, label in [("codesearch", "Engineering", "Code search: docstring to function"),
                       ("stackoverflow", "Engineering", "Developer Q&A: question to answer"),
                       ("quora", "Support & sales", "Duplicate tickets: find the twin question"),
                       ("fiqa", "Finance", "Investor question to answer"),
                       ("legal", "Legal", "Question to contract clause"),
                       ("scifact", "Science & health", "Claim to research evidence"),
                       ("nfcorpus", "Science & health", "Health question to medical research")]:
    r = T[k]; add(k, dept, label, "nDCG@10", r["loc_768"]["ndcg10"], r["cloud_768"]["ndcg10"], tuple(r["diff_768"]), f'{r["queries"]} q · {r["docs"]:,} docs', B.get(k))
r = T["banking77"]; add("banking77", "Support & sales", "Route tickets to 77 intents", "accuracy", r["loc_768"]["accuracy"], r["cloud_768"]["accuracy"], tuple(r["diff_768"]), f'{r["test"]} tickets')
r = T["stsb"]; add("stsb", "Support & sales", "Score how similar two sentences are", "Spearman", r["loc_768"]["spearman"], r["cloud_768"]["spearman"], tuple(r["diff_768"]), f'{r["pairs"]} pairs')
r = T["resume"]; add("resume", "HR", "Resume vs job description fit", "ROC-AUC", r["loc_768"]["auc"], r["cloud_768"]["auc"], tuple(r["diff_768"]), f'{r["pairs"]:,} pairs')
add("tatoeba", "Languages", "Match a sentence to its English translation", "accuracy", tat_mean("loc"), tat_mean("cloud"), tuple(T["tatoeba"]["diff_768"]), "6 langs · 1000 each")
p = M["photos"]; add("photos", "Field & media", "Text to photo (1,000 photos)", "R@1", p["loc_t2i_768"]["R@1"], p["cloud_t2i_768"]["R@1"], tuple(p["diff_t2i_R@1"]), "5,000 queries")
s = M["sounds"]; add("sounds", "Field & media", "Classify everyday sounds (50 classes)", "accuracy", s["loc"]["zero_shot_clip_classification"], s["cloud"]["zero_shot_clip_classification"], tuple(s["diff_zero_shot"]), "500 clips")
sp = M["speech"]; add("speech", "Field & media", "Voice note to its transcript", "R@1", sp["loc"]["audio_to_text"]["R@1"], sp["cloud"]["audio_to_text"]["R@1"], None, "73 clips")
pd_ = M["pdf"]; add("pdf", "Field & media", "Find the PDF page from its image", "R@1", pd_["loc"]["page_image"]["R@1"], pd_["cloud"]["page_image"]["R@1"], None, f'{pd_["queries"]} q · {pd_["pages"]} pages')

if "clustering" in X:
    c = X["clustering"]; add("clustering", "Support & sales", "Group tickets into topics (clustering)", "V-measure", c["loc_768"]["v_measure"], c["cloud_768"]["v_measure"], None, "1,000 tickets")
if "dedupe" in X:
    c = X["dedupe"]; add("dedupe", "Support & sales", "Flag duplicate support questions", "avg precision", c["loc"]["ap"], c["cloud"]["ap"], None, "5,000 pairs")
if "fleurs" in X:
    langs = [k for k in X["fleurs"] if isinstance(X["fleurs"][k], dict) and "loc" in X["fleurs"][k]]
    lv = float(np.mean([X["fleurs"][k]["loc"]["speech_to_own_transcript_R@1"] for k in langs])); cv = float(np.mean([X["fleurs"][k]["cloud"]["speech_to_own_transcript_R@1"] for k in langs]))
    add("fleurs", "Languages", "Real voices: speech to its transcript (4 languages)", "R@1", lv, cv, None, f"{len(langs)} langs · 150 each")
if "video" in X:
    v = X["video"]; add("video", "Field & media", "Find a moment in a video", "R@1", v["loc"]["R@1"], v["cloud"]["R@1"], None, "18 q · 60 clips")
for r in ROWS:
    if r["lo"] is not None:
        r["win"] = "cloud" if r["lo"] > 0 else ("local" if r["hi"] < 0 else "tie")
    else:
        r["win"] = "tie" if abs(r["d"]) < 0.05 else ("cloud" if r["d"] > 0 else "local")
for r in ROWS:
    if r["key"] == "resume": r["verdict"] = "neither"
    elif r["win"] == "tie" or abs(r["d"]) < 0.03: r["verdict"] = "tie"
    else: r["verdict"] = r["win"]
wins = {w: sum(r["verdict"] == w for r in ROWS) for w in ("local", "cloud", "tie", "neither")}


# ---------- SVG helpers
def forest():
    W, rowh, top, left, right = 760, 30, 34, 300, 40
    xmin, xmax = -0.20, 0.30
    def X(v): return left + (v - xmin) / (xmax - xmin) * (W - left - right)
    h = top + rowh * len(ROWS) + 30
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Difference between cloud and local scores per test, with 95% confidence intervals">']
    for t in (-0.2, -0.1, 0, 0.1, 0.2, 0.3):
        s.append(f'<line x1="{X(t):.1f}" y1="{top-8}" x2="{X(t):.1f}" y2="{h-26}" class="{"axis0" if t == 0 else "grid"}"/>')
        s.append(f'<text x="{X(t):.1f}" y="{h-10}" text-anchor="middle">{t:+.1f}</text>' if t else f'<text x="{X(t):.1f}" y="{h-10}" text-anchor="middle">0</text>')
    s.append(f'<text x="{X(0)-8:.1f}" y="18" text-anchor="end" class="lab-l">◀ local better</text><text x="{X(0)+8:.1f}" y="18" class="lab-c">cloud better ▶</text>')
    for i, r in enumerate(ROWS):
        y = top + i * rowh + rowh / 2
        cls = {"local": "m-loc", "cloud": "m-cld", "tie": "m-tie", "neither": "m-tie"}[r["verdict"]]
        tip = f'{r["label"]}|cloud − local = {r["d"]:+.3f}' + (f' (95% CI {r["lo"]:+.3f} to {r["hi"]:+.3f})' if r["lo"] is not None else " (no CI: small sample)")
        s.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="rowlab">{e(r["label"])}</text>')
        if r["lo"] is not None:
            s.append(f'<line x1="{X(max(r["lo"], xmin)):.1f}" y1="{y:.1f}" x2="{X(min(r["hi"], xmax)):.1f}" y2="{y:.1f}" class="ci {cls}"/>')
        s.append(f'<circle cx="{X(max(min(r["d"], xmax), xmin)):.1f}" cy="{y:.1f}" r="5.5" class="{cls}" tabindex="0" data-tip="{e(tip)}"/>')
    s.append("</svg>")
    return "".join(s)


def dumbbell():
    W, rowh, top, left, right = 760, 30, 30, 300, 30
    def X(v): return left + v * (W - left - right)
    h = top + rowh * len(ROWS) + 30
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Local and cloud scores per test on a 0 to 1 scale, with keyword search where it applies">']
    for t in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{X(t):.1f}" y1="{top-8}" x2="{X(t):.1f}" y2="{h-26}" class="grid"/><text x="{X(t):.1f}" y="{h-10}" text-anchor="middle">{t:g}</text>')
    for i, r in enumerate(ROWS):
        y = top + i * rowh + rowh / 2
        s.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="rowlab">{e(r["label"])} <tspan class="unit">{r["metric"]}</tspan></text>')
        s.append(f'<line x1="{X(min(r["loc"], r["cld"])):.1f}" y1="{y:.1f}" x2="{X(max(r["loc"], r["cld"])):.1f}" y2="{y:.1f}" class="bar-link"/>')
        if r["bm"] is not None:
            s.append(f'<rect x="{X(r["bm"])-1.5:.1f}" y="{y-8:.1f}" width="3" height="16" class="m-kw" tabindex="0" data-tip="{e(r["label"])}|keyword search (BM25) {r["bm"]:.3f}"/>')
        s.append(f'<circle cx="{X(r["loc"]):.1f}" cy="{y:.1f}" r="6" class="m-loc" tabindex="0" data-tip="{e(r["label"])}|EmbeddingGemma 2 {r["loc"]:.3f} {r["metric"]}"/>')
        s.append(f'<circle cx="{X(r["cld"]):.1f}" cy="{y:.1f}" r="6" class="m-cld" tabindex="0" data-tip="{e(r["label"])}|Gemini Embedding 2 {r["cld"]:.3f} {r["metric"]}"/>')
    s.append("</svg>")
    return "".join(s)


def mrl():
    keys = [r["key"] for r in ROWS if r["metric"] == "nDCG@10"]
    dims = [768, 512, 256, 128]
    ret = {m: [float(np.mean([T[k][f"{m}_{d}"]["ndcg10"] / T[k][f"{m}_768"]["ndcg10"] for k in keys])) for d in dims] for m in ("loc", "cloud")}
    ph = {m: [M["photos"][f"{m}_t2i_{d}"]["R@1"] / M["photos"][f"{m}_t2i_768"]["R@1"] for d in (768, 256, 128)] for m in ("loc", "cloud")}
    W, Hh, l, r_, t, b = 760, 260, 60, 150, 20, 40
    xs = {768: 0, 512: 1, 256: 2, 128: 3}
    def X(d): return l + xs[d] / 3 * (W - l - r_)
    def Y(v): return t + (1.0 - v) / 0.4 * (Hh - t - b)
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Share of full-size quality kept as vectors shrink">']
    for v in (1.0, .9, .8, .7, .6):
        s.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{W-r_}" y2="{Y(v):.1f}" class="grid"/><text x="{l-8}" y="{Y(v)+4:.1f}" text-anchor="end">{int(v*100)}%</text>')
    for d in dims: s.append(f'<text x="{X(d):.1f}" y="{Hh-14}" text-anchor="middle">{d}d</text>')
    series = [("loc", ret["loc"], dims, "m-loc", "", "EmbeddingGemma 2 · text"), ("cloud", ret["cloud"], dims, "m-cld", "", "Gemini · text"),
              ("loc", ph["loc"], [768, 256, 128], "m-loc", "dash", "EmbeddingGemma 2 · photos"), ("cloud", ph["cloud"], [768, 256, 128], "m-cld", "dash", "Gemini · photos")]
    for m, vals, ds, cls, dash, name in series:
        pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(ds, vals))
        s.append(f'<polyline points="{pts}" class="ln {cls} {dash}"/>')
        for d, v in zip(ds, vals):
            s.append(f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="4.5" class="{cls}" tabindex="0" data-tip="{e(name)}|{d}d keeps {v*100:.1f}% of 768d quality"/>')
        s.append(f'<text x="{X(ds[-1])+10:.1f}" y="{Y(vals[-1])+4:.1f}" class="endlab">{e(name)}</text>')
    s.append("</svg>")
    return "".join(s), ret, ph


def grouped(cats, a, b, ylab, maxv=1.0, labels=("EmbeddingGemma 2", "Gemini Embedding 2"), fmt=lambda v: f"{v*100:.0f}%"):
    W, Hh, l, t, bt = 760, 240, 50, 16, 46
    n = len(cats); gw = (W - l - 10) / n; bw = min(26, gw / 3)
    def Y(v): return t + (1 - v / maxv) * (Hh - t - bt)
    s = [f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="{e(ylab)}">']
    for v in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{l}" y1="{Y(v*maxv):.1f}" x2="{W-10}" y2="{Y(v*maxv):.1f}" class="grid"/><text x="{l-8}" y="{Y(v*maxv)+4:.1f}" text-anchor="end">{fmt(v*maxv)}</text>')
    for i, c in enumerate(cats):
        cx = l + gw * i + gw / 2
        for j, (vals, cls, nm) in enumerate(((a, "m-loc", labels[0]), (b, "m-cld", labels[1]))):
            x = cx - bw - 1 + j * (bw + 2); v = vals[i]
            s.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{Y(0)-Y(v):.1f}" rx="3" class="{cls}" tabindex="0" data-tip="{e(c)}|{e(nm)} {fmt(v)}"/>')
        s.append(f'<text x="{cx:.1f}" y="{Hh-26}" text-anchor="middle" class="rowlab">{e(c)}</text>')
    s.append("</svg>")
    return "".join(s)


def latency_chart():
    rows = [("Text query", L.get("text_p50_loc", 13), L.get("text_p50_cloud", 551))]
    lat = M.get("latency", {})
    if "image_query" in lat: rows.append(("Photo query", lat["image_query"]["loc_p50_ms"], lat["image_query"]["cloud_p50_ms"]))
    if "audio_query" in lat: rows.append(("Voice query", lat["audio_query"]["loc_p50_ms"], lat["audio_query"]["cloud_p50_ms"]))
    W, l, rowh = 760, 120, 48
    mx = max(max(a, b) for _, a, b in rows) * 1.1
    def X(v): return l + v / mx * (W - l - 80)
    h = rowh * len(rows) + 30
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Median time to embed one query, local versus cloud">']
    for i, (nm, a, b) in enumerate(rows):
        y = 10 + i * rowh
        s.append(f'<text x="{l-10}" y="{y+22}" text-anchor="end" class="rowlab">{nm}</text>')
        s.append(f'<rect x="{l}" y="{y+4}" width="{max(X(a)-l,3):.1f}" height="14" rx="3" class="m-loc" tabindex="0" data-tip="{nm}|EmbeddingGemma 2 {a:.0f} ms"/><text x="{max(X(a), l+3)+6:.1f}" y="{y+16}" class="val">{a:.0f} ms</text>')
        s.append(f'<rect x="{l}" y="{y+22}" width="{X(b)-l:.1f}" height="14" rx="3" class="m-cld" tabindex="0" data-tip="{nm}|Gemini Embedding 2 {b:.0f} ms"/><text x="{X(b)+6:.1f}" y="{y+34}" class="val">{b:.0f} ms</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- assets
def img_data(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode() if path and os.path.exists(path) else ""


def logs():
    out = []
    for p in sorted(glob.glob(f"{H}/logs/*.log")):
        txt = open(p, errors="replace").read()
        txt = scrub(txt)
        out.append((os.path.basename(p), txt[-60000:]))
    return out


MOD = f"{H}/mod/bench-watch"
CODE = [  # (path, shown as, what it is)
    (f"{H}/bench.py", "bench.py", "Shared helpers: local model loader, cloud batching (one Content per item, 100 per call), first benchmark"),
    (f"{H}/suite_text.py", "suite_text.py", "11 text tests with paired bootstrap intervals; Qwen3 and Gemini 1 branches"),
    (f"{H}/suite_media.py", "suite_media.py", "Photos, multilingual and spoken photo search, sounds, voice notes, PDF pages, latency"),
    (f"{H}/suite_more.py", "suite_more.py", "Extended tests: hybrid, clustering, duplicates, calibration, mixed inputs, long documents, baselines, 4-bit/CPU, real voices, video"),
    (f"{H}/suite_bm25.py", "suite_bm25.py", "Keyword (BM25) baseline on the identical samples"),
    (f"{H}/suite_speed.py", "suite_speed.py", "Latency and throughput at concurrency 1, 4 and 16"),
    (f"{H}/ann_scale.py", "ann_scale.py", "Exact and HNSW search speed at 1M vectors"),
    (f"{H}/ann_real.py", "ann_real.py", "HNSW recall on real EmbeddingGemma 2 vectors"),
    (f"{H}/make_assets.py", "make_assets.py", "Caption translations and macOS text-to-speech queries"),
    (f"{H}/build_report.py", "build_report.py", "Builds this page from results/*.json and logs/*.log"),
    (f"{H}/report_template.html", "report_template.html", "This page's template (Jinja2)"),
    (f"{H}/assets/video/queries.json", "assets/video/queries.json", "The 18 hand-labelled video queries and their correct clips"),
    (f"{H}/diagrams/bench-pipeline.json", "diagrams/bench-pipeline.json", "Archify source for the pipeline diagram"),
    (f"{H}/diagrams/which-embedding.json", "diagrams/which-embedding.json", "Archify source for the decision diagram"),
    (f"{MOD}/.claude-plugin/plugin.json", "mod/bench-watch/.claude-plugin/plugin.json", "Mod manifest"),
    (f"{MOD}/hooks/hooks.json", "mod/bench-watch/hooks/hooks.json", "Mod hooks list"),
    (f"{MOD}/types/index.d.ts", "mod/bench-watch/types/index.d.ts", "Mod state contract"),
    (f"{MOD}/hooks/register.tsx", "mod/bench-watch/hooks/register.tsx", "Mod: live pane, status line entry and /bench command"),
    (f"{MOD}/hooks/register.test.ts", "mod/bench-watch/hooks/register.test.ts", "Mod test (claude plugin test)"),
]


def scrub(txt):
    for k, v in ((H + "/", ""), (os.path.expanduser("~"), "~"), (platform.node(), "mac")):
        txt = txt.replace(k, v)
    return txt


def code_files():
    out = []
    for p, shown, desc in CODE:
        if not os.path.exists(p): continue
        txt = scrub(open(p).read())
        out.append((shown, desc, txt, len(txt.splitlines())))
    return out


def sysinfo():
    def sh(c):
        try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception: return ""
    return {"cpu": sh("sysctl -n machdep.cpu.brand_string"), "mem": f'{int(sh("sysctl -n hw.memsize"))/2**30:.0f} GB',
            "os": "macOS " + sh("sw_vers -productVersion"), "py": platform.python_version(),
            "torch": sh(f"{sys.executable} -c 'import torch;print(torch.__version__)'"),
            "st": sh(f"{sys.executable} -c 'import sentence_transformers as s;print(s.__version__)'"),
            "genai": sh(f"{sys.executable} -c 'import google.genai as g;print(g.__version__)'"),
            "llama": sh(f"{H}/llamacpp/llama-b11541/llama-server --version 2>&1 | head -1")}


if __name__ == "__main__":
    out = sys.argv[1]
    tpl = open(f"{H}/report_template.html").read()
    fsvg, mrl_ret, mrl_ph = mrl()
    tat = T["tatoeba"]["langs"]; lang_names = {"ara": "Arabic", "hin": "Hindi", "mal": "Malayalam", "tam": "Tamil", "urd": "Urdu", "fra": "French"}
    mp = M["multilingual_photos"]; ln = {"en": "English", "de": "German", "es": "Spanish", "hi": "Hindi", "ar": "Arabic", "ml": "Malayalam"}
    ctx = dict(ROWS=ROWS, wins=wins, T=T, M=M, B=B, L=L, mrl_ret=mrl_ret, mrl_ph=mrl_ph, sys=sysinfo(),
               forest=forest(), dumbbell=dumbbell(), mrl=fsvg, latency=latency_chart(),
               tat=grouped([lang_names[k] for k in tat], [v["loc"] for v in tat.values()], [v["cloud"] for v in tat.values()], "Translation matching accuracy by language"),
               mlp=grouped([ln[k] for k in mp["typed"]], [v["loc"]["R@1"] for v in mp["typed"].values()], [v["cloud"]["R@1"] for v in mp["typed"].values()], "Typed photo search by language, R@1"),
               spk=grouped([ln[k] + " (spoken)" for k in mp["spoken"]], [v["loc"]["R@1"] for v in mp["spoken"].values()], [v["cloud"]["R@1"] for v in mp["spoken"].values()], "Spoken photo search, R@1"),
               img={**{k: img_data(f"{H}/diagrams/{k}-crop.png") for k in ("pipeline-light", "pipeline-dark", "decision-light", "decision-dark")}, "mod": img_data(f"{H}/diagrams/mod-pane.png")}, code=code_files(),
               X=X, logs=logs(), results_json=json.dumps({"text": T, "media": M, "bm25": B, "llamacpp": L, "extended": X}, indent=1, default=float), e=e, f3=f3, pct=pct)
    from string import Template
    import jinja2
    html_out = jinja2.Environment(autoescape=False).from_string(tpl).render(**ctx)
    open(out, "w").write(html_out); print("wrote", out, len(html_out), "bytes")
