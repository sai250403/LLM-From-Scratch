import json
import tiktoken
from collections import Counter

# === Step 1: Load jsonl and extract Q+A ===
input_file = "qa_data.jsonl"
texts = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        q = obj.get("question", "").strip()
        a = obj.get("answer", "").strip()
        if q or a:
            texts.append(q + " " + a)

# === Step 2: Load TikToken encoder (e.g. GPT-3.5 or GPT-4) ===
# You can also use "cl100k_base" which is base tokenizer for gpt-4, gpt-3.5-turbo
enc = tiktoken.get_encoding("cl100k_base")

# === Step 3: Tokenize and extract vocab ===
token_counter = Counter()

for line in texts:
    ids = enc.encode(line)
    tokens = [enc.decode_single_token_bytes(tok).decode("utf-8", errors="ignore") for tok in ids]
    token_counter.update(tokens)

# === Step 4: Save vocab to file ===
with open("tiktoken_vocab.txt", "w", encoding="utf-8") as f:
    for token, freq in token_counter.most_common():
        f.write(f"{token}\t{freq}\n")

print("✅ TikToken vocab saved to tiktoken_vocab.txt")
