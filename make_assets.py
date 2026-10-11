"""Builds the multilingual and spoken photo-search queries (macOS only: uses `say` and `afconvert`).

- assets/mm_queries.json: 50 Flickr30k captions translated into hi, ar, ml, de, es. It is committed, so a
  fresh run asks exactly the same questions. Delete it to regenerate the translations with Gemini 3.5 Flash.
- assets/speech/: the English, Hindi ("Lekha") and Arabic ("Majed") captions spoken by macOS voices,
  16 kHz mono WAV.
"""
import json, os, random, subprocess

OUT = "assets/mm_queries.json"
os.makedirs("assets/speech", exist_ok=True)

if not os.path.exists(OUT):
    assert os.environ.get("GOOGLE_API_KEY"), "Set GOOGLE_API_KEY (a Gemini API key) before running"
    from datasets import load_dataset
    from google import genai
    from google.genai import types
    rows = list(load_dataset("lmms-lab/flickr30k", split="test", streaming=True).take(1000))
    idx = sorted(random.Random(11).sample(range(1000), 50))
    caps = [rows[i]["caption"][1].strip().rstrip(" .") for i in idx]
    prompt = ("Translate each English image caption into Hindi (hi), Arabic (ar), Malayalam (ml), German (de) and Spanish (es). "
              "Return JSON: a list of objects {\"en\":..., \"hi\":..., \"ar\":..., \"ml\":..., \"de\":..., \"es\":...} in the same order.\n"
              + json.dumps(caps, ensure_ascii=False))
    r = genai.Client().models.generate_content(model="gemini-3.5-flash", contents=prompt,
                                               config=types.GenerateContentConfig(response_mime_type="application/json"))
    tr = json.loads(r.text); assert len(tr) == 50
    json.dump({"idx": idx, "queries": tr}, open(OUT, "w"), ensure_ascii=False, indent=1)

queries = json.load(open(OUT))["queries"]
for prefix, lang, voice in (("q", "en", None), ("hi", "hi", "Lekha"), ("ar", "ar", "Majed")):
    for k, q in enumerate(queries):
        wav = f"assets/speech/{prefix}{k:02d}.wav"
        if os.path.exists(wav): continue
        aiff = wav[:-4] + ".aiff"
        subprocess.run(["say"] + (["-v", voice] if voice else []) + ["-o", aiff, q[lang]], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True)
        os.remove(aiff)
print(len(os.listdir("assets/speech")), "spoken queries in assets/speech")
