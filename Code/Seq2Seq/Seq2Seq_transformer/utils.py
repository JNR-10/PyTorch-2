import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================
# Translate a single English sentence → Spanish translation
# ============================================================

def translate_sentence(model, sentence, vocab_src, vocab_tgt, device, max_length=50):
    """
    Translate a given English sentence using the trained Transformer model.

    Args:
        model: trained Transformer model
        sentence: English sentence (string or list of tokens)
        vocab_src: source vocabulary (English)
        vocab_tgt: target vocabulary (Spanish)
        device: torch.device
        max_length: maximum number of tokens to generate
    Returns:
        List of translated Spanish tokens
    """

    # Load English spaCy tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Tokenize and lowercase the sentence
    if isinstance(sentence, str):
        tokens = [tok.text.lower() for tok in spacy_eng.tokenizer(sentence)]
    else:
        tokens = [tok.lower() for tok in sentence]

    # Add <sos> and <eos>
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    # Convert tokens to indices
    src_indices = [vocab_src.stoi.get(token, vocab_src.unk_index) for token in tokens]

    # Convert to tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)

    # Build source mask and encoder output
    model.eval()
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        src_seq_len = src_tensor.shape[0]
        src_positions = torch.arange(0, src_seq_len).unsqueeze(1).to(device)
        embed_src = model.src_word_embedding(src_tensor) + model.src_pos_embedding(src_positions)
        memory = model.transformer.encoder(embed_src, src_key_padding_mask=src_mask)

    # Begin decoding with <sos> token
    outputs = [vocab_tgt.sos_index]

    for _ in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        trg_positions = torch.arange(0, trg_tensor.shape[0]).unsqueeze(1).to(device)
        embed_trg = model.trg_word_embedding(trg_tensor) + model.trg_pos_embedding(trg_positions)
        trg_mask = model.transformer.generate_square_subsequent_mask(trg_tensor.shape[0]).to(device)

        with torch.no_grad():
            out = model.transformer.decoder(embed_trg, memory, tgt_mask=trg_mask)
            prediction = model.fc_out(out)
            best_guess = prediction.argmax(2)[-1, :].item()

        outputs.append(best_guess)
        if best_guess == vocab_tgt.eos_index:
            break

    # Convert indices back to words
    translated_sentence = [vocab_tgt.itos[idx] for idx in outputs]
    return translated_sentence[1:]


# ============================================================
# Compute BLEU score (no torchtext)
# ============================================================

def compute_bleu(dataset, model, vocab_src, vocab_tgt, device):
    """
    Compute average BLEU score on a test dataset.

    Args:
        dataset: custom TranslationDataset (English–Spanish pairs)
        model: trained Transformer model
        vocab_src: English vocabulary
        vocab_tgt: Spanish vocabulary
        device: torch.device
    Returns:
        float: average BLEU score
    """
    smooth = SmoothingFunction().method4
    scores = []

    model.eval()
    with torch.no_grad():
        for i in range(min(len(dataset), 100)):
            src_sentence = dataset.src_sents[i]
            tgt_sentence = dataset.tgt_sents[i]

            # Generate translation
            predicted_tokens = translate_sentence(model, src_sentence, vocab_src, vocab_tgt, device)
            predicted_tokens = predicted_tokens[:-1]  # remove <eos>

            # Reference tokens
            spacy_esp = spacy.load("es_core_news_sm")
            reference = [tok.text.lower() for tok in spacy_esp.tokenizer(tgt_sentence)]

            # Compute BLEU
            score = sentence_bleu([reference], predicted_tokens, smoothing_function=smooth)
            scores.append(score)

    return sum(scores) / len(scores)


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(state, filename="transformer_checkpoint.pth.tar"):
    """Save model and optimizer states to a file."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """Load model and optimizer states from a checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
