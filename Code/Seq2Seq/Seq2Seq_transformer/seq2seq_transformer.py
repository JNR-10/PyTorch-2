import os
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# 1. Setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download if missing:
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm
spacy_eng = spacy.load("en_core_web_sm")
spacy_esp = spacy.load("es_core_news_sm")

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

def tokenize_esp(text):
    return [tok.text.lower() for tok in spacy_esp.tokenizer(text)]


# ============================================================
# 2. Dataset and vocabulary
# ============================================================

class Vocab:
    def __init__(self, tokenized_texts, specials=["<unk>", "<pad>", "<sos>", "<eos>"], min_freq=2):
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
        self.sos_index = self.stoi["<sos>"]
        self.eos_index = self.stoi["<eos>"]

    def numericalize(self, tokens):
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    """English → Spanish dataset"""
    def __init__(self, df, vocab_src, vocab_tgt):
        self.src_sents = df["English"].values
        self.tgt_sents = df["Spanish"].values
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src_tokens = ["<sos>"] + tokenize_eng(self.src_sents[idx]) + ["<eos>"]
        tgt_tokens = ["<sos>"] + tokenize_esp(self.tgt_sents[idx]) + ["<eos>"]
        src_tensor = torch.tensor(self.vocab_src.numericalize(src_tokens), dtype=torch.long)
        tgt_tensor = torch.tensor(self.vocab_tgt.numericalize(tgt_tokens), dtype=torch.long)
        return src_tensor, tgt_tensor


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=vocab_eng.pad_index)
    tgt_padded = pad_sequence(tgt_batch, padding_value=vocab_esp.pad_index)
    return src_padded.to(device), tgt_padded.to(device)


# Load your English and Spanish text files (1 sentence per line)
english_txt = open("train_WMT_english.txt", encoding="utf8").read().split("\n")
spanish_txt = open("train_WMT_spanish.txt", encoding="utf8").read().split("\n")

df = pd.DataFrame({"English": english_txt[:5000], "Spanish": spanish_txt[:5000]})
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Tokenize and build vocab
tokenized_eng = [tokenize_eng(s) for s in train_df["English"]]
tokenized_esp = [tokenize_esp(s) for s in train_df["Spanish"]]
vocab_eng = Vocab(tokenized_eng)
vocab_esp = Vocab(tokenized_esp)

# DataLoaders
train_loader = DataLoader(TranslationDataset(train_df, vocab_eng, vocab_esp),
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_df, vocab_eng, vocab_esp),
                        batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(TranslationDataset(test_df, vocab_eng, vocab_esp),
                         batch_size=32, collate_fn=collate_fn)


# ============================================================
# 3. Transformer Model
# ============================================================

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super().__init__()
        self.device = device
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_pos_embedding = nn.Embedding(max_len, embedding_size)

        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=embedding_size * forward_expansion,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # shape (N, src_len)
        src_mask = (src.transpose(0, 1) == self.src_pad_idx)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape
        src_positions = torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device)
        trg_positions = torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device)

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_pos_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_pos_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


# ============================================================
# 4. Helper functions (translate + BLEU + checkpoint)
# ============================================================

def translate_sentence(model, sentence, vocab_src, vocab_tgt, device, max_length=50):
    model.eval()
    tokens = ["<sos>"] + tokenize_eng(sentence) + ["<eos>"]
    src_idx = [vocab_src.stoi.get(t, vocab_src.unk_index) for t in tokens]
    src_tensor = torch.LongTensor(src_idx).unsqueeze(1).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.src_word_embedding(src_tensor) + model.src_pos_embedding(
            torch.arange(0, src_tensor.shape[0]).unsqueeze(1).to(device)
        )
        memory = model.transformer.encoder(enc_src, src_key_padding_mask=src_mask)

    outputs = [vocab_tgt.sos_index]
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        trg_positions = torch.arange(0, trg_tensor.shape[0]).unsqueeze(1).to(device)
        embed_trg = model.trg_word_embedding(trg_tensor) + model.trg_pos_embedding(trg_positions)
        trg_mask = model.transformer.generate_square_subsequent_mask(trg_tensor.shape[0]).to(device)
        with torch.no_grad():
            out = model.transformer.decoder(embed_trg, memory, tgt_mask=trg_mask)
            pred = model.fc_out(out)
        next_token = pred.argmax(2)[-1, :].item()
        outputs.append(next_token)
        if next_token == vocab_tgt.eos_index:
            break
    translated = [vocab_tgt.itos[idx] for idx in outputs]
    return translated[1:]


def compute_bleu(dataset, model, vocab_src, vocab_tgt, device):
    smooth = SmoothingFunction().method4
    scores = []
    for i in range(min(len(dataset), 100)):
        src = dataset.src_sents[i]
        tgt = dataset.tgt_sents[i]
        pred = translate_sentence(model, src, vocab_src, vocab_tgt, device)
        ref = [tokenize_esp(tgt)]
        score = sentence_bleu(ref, pred[:-1], smoothing_function=smooth)
        scores.append(score)
    return sum(scores) / len(scores)


def save_checkpoint(state, filename="transformer_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# ============================================================
# 5. Training loop
# ============================================================

embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
forward_expansion = 4
dropout = 0.10
max_len = 100
learning_rate = 3e-4
num_epochs = 10

src_pad_idx = vocab_eng.pad_index
model = Transformer(
    embedding_size,
    len(vocab_eng),
    len(vocab_esp),
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_esp.pad_index)

writer = SummaryWriter("runs/transformer_loss_plot")
step = 0
load_model = False
save_model = True

if load_model and os.path.exists("transformer_checkpoint.pth.tar"):
    load_checkpoint(torch.load("transformer_checkpoint.pth.tar"), model, optimizer)

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1}/{num_epochs}]")
    model.train()
    total_loss = 0

    for batch_idx, (src, trg) in enumerate(train_loader):
        out = model(src, trg[:-1, :])
        out = out.reshape(-1, out.shape[2])
        trg = trg[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(out, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1

    print(f"Average loss: {total_loss/len(train_loader):.4f}")

    if save_model:
        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()})

    # Example translation
    example_sentence = "this is a small boat on the river"
    translated = translate_sentence(model, example_sentence, vocab_eng, vocab_esp, device)
    print("Translated example:", " ".join(translated))

# ============================================================
# 6. BLEU score evaluation
# ============================================================

test_dataset = TranslationDataset(test_df, vocab_eng, vocab_esp)
bleu_score = compute_bleu(test_dataset, model, vocab_eng, vocab_esp, device)
print(f"BLEU score: {bleu_score*100:.2f}")
