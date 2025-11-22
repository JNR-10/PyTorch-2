import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================
# Setup and preprocessing
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spaCy models for English and Spanish
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm
spacy_eng = spacy.load("en_core_web_sm")
spacy_esp = spacy.load("es_core_news_sm")

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

def tokenize_esp(text):
    return [tok.text.lower() for tok in spacy_esp.tokenizer(text)]

# ============================================================
# Load or create dataset
# ============================================================

# Each line in these files should correspond to one sentence
english_txt = open("train_WMT_english.txt", encoding="utf8").read().split("\n")
spanish_txt = open("train_WMT_spanish.txt", encoding="utf8").read().split("\n")

# Build dataframe for simplicity
df = pd.DataFrame({"English": english_txt[:5000], "Spanish": spanish_txt[:5000]})
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# ============================================================
# Vocabulary class
# ============================================================

class Vocab:
    def __init__(self, tokenized_texts, specials=["<unk>", "<pad>", "<sos>", "<eos>"], min_freq=2):
        freq = {}
        for sentence in tokenized_texts:
            for token in sentence:
                freq[token] = freq.get(token, 0) + 1
        self.itos = specials.copy()
        for token, count in freq.items():
            if count >= min_freq:
                self.itos.append(token)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_index = self.stoi["<pad>"]
        self.unk_index = self.stoi["<unk>"]
        self.sos_index = self.stoi["<sos>"]
        self.eos_index = self.stoi["<eos>"]

    def __len__(self):
        return len(self.itos)

    def numericalize(self, tokens):
        return [self.stoi.get(token, self.unk_index) for token in tokens]

# Build vocabularies
tokenized_eng = [tokenize_eng(s) for s in train_df["English"]]
tokenized_esp = [tokenize_esp(s) for s in train_df["Spanish"]]
vocab_eng = Vocab(tokenized_eng)
vocab_esp = Vocab(tokenized_esp)

# ============================================================
# Custom Dataset
# ============================================================

class TranslationDataset(Dataset):
    def __init__(self, df, vocab_src, vocab_tgt):
        self.src = df["English"].values
        self.tgt = df["Spanish"].values
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tokens = ["<sos>"] + tokenize_eng(self.src[idx]) + ["<eos>"]
        tgt_tokens = ["<sos>"] + tokenize_esp(self.tgt[idx]) + ["<eos>"]
        src_tensor = torch.tensor(self.vocab_src.numericalize(src_tokens), dtype=torch.long)
        tgt_tensor = torch.tensor(self.vocab_tgt.numericalize(tgt_tokens), dtype=torch.long)
        return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=vocab_eng.pad_index)
    tgt_padded = pad_sequence(tgt_batch, padding_value=vocab_esp.pad_index)
    return src_padded.to(device), tgt_padded.to(device)

train_loader = DataLoader(TranslationDataset(train_df, vocab_eng, vocab_esp),
                          batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_df, vocab_eng, vocab_esp),
                        batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(TranslationDataset(test_df, vocab_eng, vocab_esp),
                         batch_size=64, shuffle=False, collate_fn=collate_fn)

# ============================================================
# Seq2Seq model: Encoder, Decoder, Seq2Seq wrapper
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)  # (1, batch_size)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc_out(outputs.squeeze(0))
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_force_ratio=0.5):
        batch_size = src.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = len(vocab_esp)

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(device)
        hidden, cell = self.encoder(src)
        x = tgt[0]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = tgt[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# ============================================================
# Initialize model, loss, optimizer
# ============================================================

input_size_encoder = len(vocab_eng)
input_size_decoder = len(vocab_esp)
output_size = len(vocab_esp)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5
learning_rate = 0.001
num_epochs = 20

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = vocab_esp.pad_index
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# ============================================================
# Helper functions: save/load checkpoints and BLEU scoring
# ============================================================

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def translate_sentence(model, sentence, vocab_src, vocab_tgt, max_length=50):
    model.eval()
    tokens = ["<sos>"] + tokenize_eng(sentence) + ["<eos>"]
    src_indexes = [vocab_src.stoi.get(token, vocab_src.unk_index) for token in tokens]
    src_tensor = torch.tensor(src_indexes).unsqueeze(1).to(device)
    hidden, cell = model.encoder(src_tensor)
    outputs = []
    x = torch.tensor([vocab_tgt.sos_index]).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            output, hidden, cell = model.decoder(x, hidden, cell)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        x = torch.tensor([best_guess]).to(device)
        if best_guess == vocab_tgt.eos_index:
            break

    translated_tokens = [vocab_tgt.itos[idx] for idx in outputs]
    return translated_tokens[:-1]  # Remove <eos>

def compute_bleu(model, dataset):
    smooth = SmoothingFunction().method4
    references = []
    candidates = []
    for i in range(min(len(dataset), 100)):
        src = dataset.src[i]
        tgt = dataset.tgt[i]
        pred = translate_sentence(model, src, vocab_eng, vocab_esp)
        references.append([tokenize_esp(tgt)])
        candidates.append(pred)
    return sentence_bleu(references, candidates, smoothing_function=smooth)

# ============================================================
# Training loop
# ============================================================

writer = SummaryWriter("runs/loss_plot")
step = 0

for epoch in range(num_epochs):
    print(f"[Epoch {epoch + 1}/{num_epochs}]")
    model.train()
    total_loss = 0

    for batch_idx, (src, tgt) in enumerate(train_loader):
        output = model(src, tgt)
        output = output[1:].reshape(-1, output.shape[2])
        tgt = tgt[1:].reshape(-1)
        loss = criterion(output, tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")
    save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()})

# ============================================================
# Evaluation
# ============================================================

sentence = "this is a small boat on the river"
translation = translate_sentence(model, sentence, vocab_eng, vocab_esp)
print("Translated example sentence:", " ".join(translation))

bleu_score = compute_bleu(model, TranslationDataset(test_df, vocab_eng, vocab_esp))
print(f"BLEU score: {bleu_score * 100:.2f}")
