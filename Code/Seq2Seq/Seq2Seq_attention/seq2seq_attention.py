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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Setup and tokenization
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download models if not available:
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm
spacy_eng = spacy.load("en_core_web_sm")
spacy_esp = spacy.load("es_core_news_sm")

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

def tokenize_esp(text):
    return [tok.text.lower() for tok in spacy_esp.tokenizer(text)]

# ============================================================
# Load data from files or create sample data
# ============================================================

english_txt = open("train_WMT_english.txt", encoding="utf8").read().split("\n")
spanish_txt = open("train_WMT_spanish.txt", encoding="utf8").read().split("\n")

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
# Custom Dataset and DataLoader
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
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_df, vocab_eng, vocab_esp),
                        batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(TranslationDataset(test_df, vocab_eng, vocab_esp),
                         batch_size=32, shuffle=False, collate_fn=collate_fn)

# ============================================================
# Encoder–Decoder with Attention
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.rnn(embedding)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(vocab_esp)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)
        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs

# ============================================================
# Helper functions
# ============================================================

def save_checkpoint(state, filename="checkpoint_attn.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def translate_sentence(model, sentence, vocab_src, vocab_tgt, device, max_length=50):
    spacy_eng = spacy.load("en_core_web_sm")
    if isinstance(sentence, str):
        tokens = [tok.text.lower() for tok in spacy_eng.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")
    src_indexes = [vocab_src.stoi.get(tok, vocab_src.unk_index) for tok in tokens]
    src_tensor = torch.tensor(src_indexes).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(src_tensor)
    outputs = [vocab_tgt.sos_index]
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        if best_guess == vocab_tgt.eos_index:
            break
    translated_tokens = [vocab_tgt.itos[idx] for idx in outputs]
    return translated_tokens[1:]

def compute_bleu(dataset, model, vocab_src, vocab_tgt, device):
    smooth = SmoothingFunction().method4
    references, predictions = [], []
    for i in range(min(len(dataset), 100)):
        src = dataset.src[i]
        tgt = dataset.tgt[i]
        pred_tokens = translate_sentence(model, src, vocab_src, vocab_tgt, device)
        references.append([tokenize_esp(tgt)])
        predictions.append(pred_tokens[:-1])
    scores = [sentence_bleu(r, p, smoothing_function=smooth) for r, p in zip(references, predictions)]
    return sum(scores) / len(scores)

# ============================================================
# Initialize model, optimizer, criterion
# ============================================================

input_size_encoder = len(vocab_eng)
input_size_decoder = len(vocab_esp)
output_size = len(vocab_esp)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0
learning_rate = 3e-4
num_epochs = 20

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = vocab_esp.pad_index
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# ============================================================
# Training loop
# ============================================================

writer = SummaryWriter("runs/loss_plot_attn")
step = 0
load_model = False
save_model = True

if load_model and os.path.exists("checkpoint_attn.pth.tar"):
    load_checkpoint(torch.load("checkpoint_attn.pth.tar"), model, optimizer)

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1}/{num_epochs}]")
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

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
        total_loss += loss.item()

    print(f"Average Loss: {total_loss / len(train_loader):.4f}")

    if save_model:
        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()})

    # Validation example
    example_sentence = "this is a small boat on the river"
    translated = translate_sentence(model, example_sentence, vocab_eng, vocab_esp, device)
    print("Translated example sentence:", " ".join(translated))

# ============================================================
# BLEU score evaluation
# ============================================================

test_dataset = TranslationDataset(test_df, vocab_eng, vocab_esp)
bleu_score = compute_bleu(test_dataset, model, vocab_eng, vocab_esp, device)
print(f"BLEU score: {bleu_score * 100:.2f}")
