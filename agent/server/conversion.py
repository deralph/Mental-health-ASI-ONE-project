# convert_metta_and_embeddings_to_py.py
import os, json, sys

METTA_FILE = "mental_health_kb_ultra_clean.metta"
EMBEDDINGS_FILE = "embeddings_fixed_cleaned.json"  # change if your filename is different
OUTPUT_PY = "knowledge_base.py"
EXPECTED_DIM = 1536  # you said small model

if not os.path.exists(METTA_FILE):
    raise FileNotFoundError(f".metta not found: {METTA_FILE}")
if not os.path.exists(EMBEDDINGS_FILE):
    raise FileNotFoundError(f"embeddings file not found: {EMBEDDINGS_FILE}")

# ---------- helper: robust parser for ( = (chunk-key "id") "value") ----------
def parse_metta_key_values(metta_text, key_name):
    """
    Returns dict: chunk_id -> value for patterns like:
      (= (chunk-key "CHUNK_ID") "VALUE")
    Handles multiline and escaped quotes.
    """
    out = {}
    p = f'(=\\s*\\({key_name}\\s*"'  # search anchor
    i = 0
    L = len(metta_text)
    while True:
        idx = metta_text.find(f'(= ({key_name} "', i)
        if idx == -1:
            break
        # move pointer after the opening '(= (key "'
        j = idx + len(f'(= ({key_name} "')
        # parse chunk id until next unescaped quote
        cid_chars = []
        escaped = False
        while j < L:
            ch = metta_text[j]
            if ch == '"' and not escaped:
                j += 1
                break
            if ch == '\\' and not escaped:
                escaped = True
            else:
                cid_chars.append(ch)
                escaped = False
            j += 1
        chunk_id = "".join(cid_chars)

        # Now skip whitespace until we find the next opening double-quote for the value
        # Typically there's a ) then whitespace then "
        while j < L and metta_text[j] != '"':
            j += 1
        if j >= L:
            i = idx + 1
            continue
        # Now parse the value string, honoring escapes and allowing newlines
        j += 1  # move past opening quote
        val_chars = []
        escaped = False
        while j < L:
            ch = metta_text[j]
            if ch == '"' and not escaped:
                j += 1
                break
            if ch == '\\' and not escaped:
                escaped = True
            else:
                val_chars.append(ch)
                escaped = False
            j += 1
        value = "".join(val_chars)
        # unescape common sequences (we don't over-interpret)
        try:
            value = value.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
        out[chunk_id] = value
        i = j
    return out

# ---------- Load files ----------
with open(METTA_FILE, "r", encoding="utf-8", errors="replace") as f:
    metta_text = f.read()

print("Parsing .metta — extracting chunk-text, chunk-source, chunk-page (if present)...")
chunk_texts = parse_metta_key_values(metta_text, "chunk-text")
chunk_sources = parse_metta_key_values(metta_text, "chunk-source")
# chunk-page values often numeric; parse similarly
chunk_pages_raw = parse_metta_key_values(metta_text, "chunk-page")

# convert chunk_pages to ints (if parse failed earlier due to numbers not quoted, use fallback)
chunk_pages = {}
# attempt regex fallback for numeric pages like: (= (chunk-page "id") 12)
import re
for k,v in chunk_pages_raw.items():
    try:
        chunk_pages[k] = int(v)
    except:
        chunk_pages[k] = int(re.sub(r'\D', '', v) or 0)

# try numeric page pattern for entries where our parser didn't capture (uncommon)
num_pat = re.compile(r'\(\=\s*\(chunk-page\s*"([^"]+)"\)\s*([0-9]+)\s*\)', re.MULTILINE)
for m in num_pat.finditer(metta_text):
    cid = m.group(1)
    page = int(m.group(2))
    chunk_pages[cid] = page

print(f"Found {len(chunk_texts)} chunk-text entries, {len(chunk_sources)} chunk-source entries, {len(chunk_pages)} chunk-page entries (from .metta)")

# ---------- Load embeddings.json ----------
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    embeds = json.load(f)

# sanitize embeddings list and keep only expected dim items
filtered_embeds = []
for item in embeds:
    emb = item.get("embedding") or item.get("values") or item.get("vector") or []
    # if API object present, try to extract
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    if isinstance(emb, dict) and "embedding" in emb:
        emb = emb["embedding"]
    if isinstance(emb, (list, tuple)):
        if len(emb) == EXPECTED_DIM:
            filtered_embeds.append({
                "chunk_id": item.get("chunk_id") or item.get("id") or item.get("metadata", {}).get("chunk_id"),
                "text": item.get("text") or item.get("metadata", {}).get("text") or item.get("page_content") or "",
                "source": item.get("source") or item.get("metadata", {}).get("source") or "",
                "page": int(item.get("page") or item.get("metadata", {}).get("page", 0) or 0),
                "embedding": list(map(float, emb))
            })
        else:
            # skip non-matching dims
            pass

print(f"Kept {len(filtered_embeds)} embeddings with dimension {EXPECTED_DIM} (out of {len(embeds)})")

# ---------- Merge: for each embedding item, prefer .metta chunk-text if present ----------
merged = []
missing_text_from_metta = 0
for item in filtered_embeds:
    cid = item.get("chunk_id")
    if not cid:
        continue
    text = ""
    if cid in chunk_texts:
        text = chunk_texts[cid]
    else:
        # fallback to text inside embeddings file (if any)
        text = (item.get("text") or "").strip()
    if not text:
        missing_text_from_metta += 1
    source = chunk_sources.get(cid) or item.get("source") or ""
    page = chunk_pages.get(cid, item.get("page", 0) or 0)
    merged.append({
        "chunk_id": cid,
        "text": text,
        "source": source,
        "page": int(page),
        "embedding": item["embedding"]
    })

print(f"Merged KB items: {len(merged)} (missing text from .metta for {missing_text_from_metta} items)")

# ---------- Write Python file ----------
# We will write JSON text (valid Python) — safe because we only output strings, ints, floats, lists
with open(OUTPUT_PY, "w", encoding="utf-8") as f:
    f.write("# Auto-generated knowledge base from .metta + embeddings\n")
    f.write("# Contains KNOWLEDGE_BASE (list of dicts) and CHUNK_INDEX (mapping chunk_id -> index)\n\n")
    f.write("KNOWLEDGE_BASE = ")
    json.dump(merged, f, ensure_ascii=False, indent=2)
    f.write("\n\n")
    f.write("CHUNK_INDEX = {}\n")
    f.write("for i, item in enumerate(KNOWLEDGE_BASE):\n")
    f.write("    CHUNK_INDEX[item['chunk_id']] = i\n")

print(f"Wrote Python KB module: {OUTPUT_PY}")
print("Summary:")
print(" - total merged items:", len(merged))
print(" - missing text_from_metta count:", missing_text_from_metta)
print(" - sample chunk_id (first item):", merged[0]['chunk_id'] if merged else "NONE")
print("Check knowledge_base.py now and then run your chatbot that imports KNOWLEDGE_BASE.")
