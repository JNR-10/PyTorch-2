import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# ==================== Device setup ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Language setup ====================
# Source: English, Target: Spanish
SRC_LANG = "en"
TGT_LANG = "es"

# Download these models if not already installed:
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm

spacy_src = spacy.load(f"{SRC_LANG}_core_web_sm")
spacy_tgt = spacy.load(f"{TGT_LANG}_core_news_sm")

# ==================== Tokenization functions ====================
# Tokenize a sentence into a list of lowercase tokens
def tokenize_src(text):
    return [tok.text.lower() for tok in spacy_src.tokenizer(text)]

def tokenize_tgt(text):
    return [tok.text.lower() for tok in spacy_tgt.tokenizer(text)]

# ==================== Example parallel dataset ====================
# Replace these lists with your own English-Spanish sentence pairs

data_src = [
    "i am a student",
    "i like learning artificial intelligence",
    "the weather is nice today",
    "this is a simple example",
    "he is reading a book",
    "she is writing a letter",
    "we are playing football",
    "they are cooking dinner",
    "do you speak english",
    "my house is near the park",
]

data_tgt = [
    "yo soy estudiante",
    "me gusta aprender inteligencia artificial",
    "hace buen tiempo hoy",
    "este es un ejemplo sencillo",
    "él está leyendo un libro",
    "ella está escribiendo una carta",
    "nosotros estamos jugando al fútbol",
    "ellos están cocinando la cena",
    "hablas inglés",
    "mi casa está cerca del parque",
]

# Split data into training, validation, and test sets
train_src, test_src, train_tgt, test_tgt = train_test_split(
    data_src, data_tgt, test_size=0.25, random_state=42
)
val_src, test_src, val_tgt, test_tgt = train_test_split(
    test_src, test_tgt, test_size=0.5, random_state=42
)

# ==================== Vocabulary class ====================
# This class maps words to integer IDs and vice versa
class Vocab:
    def __init__(self, tokenized_texts, specials=["<unk>", "<pad>", "<sos>", "<eos>"], min_freq=1):
        freq = {}
        for sent in tokenized_texts:
            for tok in sent:
                freq[tok] = freq.get(tok, 0) + 1
        self.itos = specials.copy()
        for tok, count in freq.items():
            if count >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_index = self.stoi["<pad>"]
        self.unk_index = self.stoi["<unk>"]

    def numericalize(self, tokens):
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

    def __len__(self):
        # Return the number of tokens in the vocabulary
        return len(self.itos)


# Tokenize all training sentences for vocabulary building
tokenized_src = [tokenize_src(sentence) for sentence in train_src]
tokenized_tgt = [tokenize_tgt(sentence) for sentence in train_tgt]

# Build vocabularies for both languages
vocab_src = Vocab(tokenized_src)
vocab_tgt = Vocab(tokenized_tgt)

# ==================== Custom Dataset ====================
# Stores tokenized, numericalized source-target sentence pairs
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, vocab_src, vocab_tgt):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = ["<sos>"] + tokenize_src(self.src_sentences[idx]) + ["<eos>"]
        tgt_tokens = ["<sos>"] + tokenize_tgt(self.tgt_sentences[idx]) + ["<eos>"]
        src_tensor = torch.tensor(self.vocab_src.numericalize(src_tokens), dtype=torch.long)
        tgt_tensor = torch.tensor(self.vocab_tgt.numericalize(tgt_tokens), dtype=torch.long)
        return src_tensor, tgt_tensor

# Create dataset objects
train_dataset = TranslationDataset(train_src, train_tgt, vocab_src, vocab_tgt)
val_dataset = TranslationDataset(val_src, val_tgt, vocab_src, vocab_tgt)
test_dataset = TranslationDataset(test_src, test_tgt, vocab_src, vocab_tgt)

# ==================== Collate function ====================
# Pads sentences within a batch to the same length
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=vocab_src.pad_index)
    tgt_padded = pad_sequence(tgt_batch, padding_value=vocab_tgt.pad_index)
    return src_padded.to(device), tgt_padded.to(device)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

# ==================== Verification ====================

print("Vocabulary sizes:")
print(f"Source (English): {len(vocab_src)} tokens")
print(f"Target (Spanish): {len(vocab_tgt)} tokens\n")

# Inspect one batch
for src_batch, tgt_batch in train_loader:
    print("SRC batch shape:", src_batch.shape)
    print("TGT batch shape:", tgt_batch.shape)
    break

# Vocabulary lookup tests
word = "student"
index = vocab_src.stoi.get(word, vocab_src.unk_index)
print(f'Index of the word "{word}" is: {index}')
print(f'Word of the index {index} is: {vocab_src.itos[index]}')
