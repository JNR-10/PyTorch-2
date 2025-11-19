import spacy
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# ============================================================
# Load raw text data
# ============================================================

# Each line of these text files should contain one sentence.
english_txt = open("Code/Seq2Seq/working_with_text/mydata/train_WMT_english.txt", encoding="utf8").read().split("\n")
spanish_txt = open("Code/Seq2Seq/working_with_text/mydata/train_WMT_spanish.txt", encoding="utf8").read().split("\n")

# Use first 100 sentences for demonstration
raw_data = {
    "English": [line for line in english_txt[1:100]],
    "Spanish": [line for line in spanish_txt[1:100]],
}

# Create a DataFrame with English–Spanish sentence pairs
df = pd.DataFrame(raw_data, columns=["English", "Spanish"])

# Split into training and testing sets (90% train, 10% test)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Optionally save splits to disk for inspection
train_df.to_json("train.json", orient="records", lines=True)
test_df.to_json("test.json", orient="records", lines=True)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# ============================================================
# Load spaCy tokenizers for English and Spanish
# ============================================================

# Download these if not installed:
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm

spacy_eng = spacy.load("en_core_web_sm")
spacy_esp = spacy.load("es_core_news_sm")

def tokenize_eng(text):
    """Tokenize an English sentence into a list of lowercase tokens."""
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

def tokenize_esp(text):
    """Tokenize a Spanish sentence into a list of lowercase tokens."""
    return [tok.text.lower() for tok in spacy_esp.tokenizer(text)]

# ============================================================
# Vocabulary class (replaces torchtext Field)
# ============================================================

class Vocab:
    def __init__(self, tokenized_texts, specials=["<unk>", "<pad>", "<sos>", "<eos>"], min_freq=2):
        """Build mappings from tokens to indices and vice versa."""
        freq = {}
        for sentence in tokenized_texts:
            for token in sentence:
                freq[token] = freq.get(token, 0) + 1

        # Start vocabulary with special tokens
        self.itos = specials.copy()
        for token, count in freq.items():
            if count >= min_freq:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_index = self.stoi["<pad>"]
        self.unk_index = self.stoi["<unk>"]

    def numericalize(self, tokens):
        """Convert tokens to indices."""
        return [self.stoi.get(token, self.unk_index) for token in tokens]

    def __len__(self):
        """Return vocabulary size."""
        return len(self.itos)

# ============================================================
# Tokenize and build vocabularies
# ============================================================

# Tokenize English and Spanish training data
tokenized_eng = [tokenize_eng(s) for s in train_df["English"]]
tokenized_esp = [tokenize_esp(s) for s in train_df["Spanish"]]

# Build vocabularies for each language
vocab_eng = Vocab(tokenized_eng, min_freq=2)
vocab_esp = Vocab(tokenized_esp, min_freq=2)

# ============================================================
# Custom Dataset class (replaces TabularDataset)
# ============================================================

class TranslationDataset(Dataset):
    """Dataset storing English–Spanish sentence pairs."""
    def __init__(self, df, vocab_src, vocab_tgt):
        self.src_sents = df["English"].values
        self.tgt_sents = df["Spanish"].values
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        """Return tokenized and numericalized sentence pair."""
        src_tokens = ["<sos>"] + tokenize_eng(self.src_sents[idx]) + ["<eos>"]
        tgt_tokens = ["<sos>"] + tokenize_esp(self.tgt_sents[idx]) + ["<eos>"]
        src_tensor = torch.tensor(self.vocab_src.numericalize(src_tokens), dtype=torch.long)
        tgt_tensor = torch.tensor(self.vocab_tgt.numericalize(tgt_tokens), dtype=torch.long)
        return src_tensor, tgt_tensor

# Create training and test datasets
train_dataset = TranslationDataset(train_df, vocab_eng, vocab_esp)
test_dataset = TranslationDataset(test_df, vocab_eng, vocab_esp)

# ============================================================
# Collate function (replaces BucketIterator)
# ============================================================

def collate_fn(batch):
    """Pad all sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=vocab_eng.pad_index)
    tgt_padded = pad_sequence(tgt_batch, padding_value=vocab_esp.pad_index)
    return src_padded, tgt_padded

# Create PyTorch DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ============================================================
# Verification
# ============================================================

print("Vocabulary sizes:")
print(f"English (source): {len(vocab_eng)} tokens")
print(f"Spanish (target): {len(vocab_esp)} tokens\n")

# Fetch a single batch
for src_batch, tgt_batch in train_loader:
    print("SRC batch shape:", src_batch.shape)
    print("TGT batch shape:", tgt_batch.shape)
    break

# Demonstrate vocabulary lookup
word = "the"
idx = vocab_eng.stoi.get(word, vocab_eng.unk_index)
print(f'Index of word "{word}": {idx}')
print(f'Word at index {idx}: {vocab_eng.itos[idx]}')
