import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

# === Step 1: Extract question+answer from JSONL and write to corpus.txt ===
input_jsonl = "qa_data.jsonl"
corpus_file = "bpe_corpus.txt"

with open(input_jsonl, "r", encoding="utf-8") as fin, open(corpus_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        q = data.get("question", "").strip()
        a = data.get("answer", "").strip()
        if q or a:
            fout.write(q + " " + a + "\n")

# === Step 2: Setup BPE tokenizer ===
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# === Step 3: Train tokenizer on the corpus ===
tokenizer.train([corpus_file], trainer)

# === Step 4: Save the vocab + merges ===
tokenizer.model.save(".", "vocab")

print("✅ Tokenizer saved: vocab.json & merges.txt")

# === Step 5: Test on a sample input ===
sample = "What is your current salary and expected hike?"
encoded = tokenizer.encode(sample)

print("\n📥 Input:", sample)
print("🧩 Tokens:", encoded.tokens)
print("🔢 Token IDs:", encoded.ids)
