import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==================== Setup ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spaCy tokenizer
spacy_en = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# ==================== Load Dataset ====================

# Each JSON file should contain a list of objects like:
# {"quote": "some text here", "score": 1}
# You can easily adapt it for CSV by replacing pd.read_json() with pd.read_csv()
train_df = pd.read_json("Code/Seq2Seq/customdata/mydata/train.json", lines=True)
test_df = pd.read_json("Code/Seq2Seq/customdata/mydata/test.json", lines=True)

# ==================== Vocabulary ====================

class Vocab:
    def __init__(self, tokens_list, min_freq=1, specials=["<unk>", "<pad>"]):
        freq = {}
        for tokens in tokens_list:
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
        # include special tokens
        self.itos = specials.copy()
        for tok, count in freq.items():
            if count >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.pad_index = self.stoi["<pad>"]
        self.unk_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def numericalize(self, tokens):
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

# Tokenize all quotes to build vocab
tokenized_quotes = [tokenize(q) for q in train_df["quote"]]
vocab = Vocab(tokenized_quotes, min_freq=1)

# ==================== Custom Dataset ====================

class QuoteDataset(Dataset):
    def __init__(self, df, vocab):
        self.quotes = df["quote"].values
        self.scores = df["score"].values
        self.vocab = vocab

    def __len__(self):
        return len(self.quotes)

    def __getitem__(self, idx):
        tokens = tokenize(self.quotes[idx])
        indices = torch.tensor(self.vocab.numericalize(tokens), dtype=torch.long)
        label = torch.tensor(float(self.scores[idx]), dtype=torch.float)
        return indices, label

train_data = QuoteDataset(train_df, vocab)
test_data = QuoteDataset(test_df, vocab)

# ==================== Collate Function ====================

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, padding_value=vocab.pad_index)
    labels = torch.stack(labels)
    return padded_texts.to(device), labels.to(device)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=2, collate_fn=collate_fn)

# ==================== LSTM Model ====================

class RNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded, (h0, c0))
        prediction = self.fc_out(outputs[-1, :, :])
        return prediction

# ==================== Hyperparameters ====================

input_size = len(vocab)
hidden_size = 512
num_layers = 2
embedding_size = 100
learning_rate = 0.005
num_epochs = 10

# ==================== Initialize & Train ====================

model = RNN_LSTM(input_size, embedding_size, hidden_size, num_layers).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training on {device} ...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data, targets in train_loader:
        scores = model(data)
        loss = criterion(scores.squeeze(1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

print("✅ Training complete!")
