import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================
# Sentence translation without torchtext
# ============================================================

def translate_sentence(model, sentence, vocab_src, vocab_tgt, device, max_length=50):
    """
    Translate a given sentence (string or list of tokens)
    using the trained Seq2Seq model.

    Args:
        model: Trained Seq2Seq model
        sentence: Input sentence (English string or token list)
        vocab_src: Source vocabulary (English)
        vocab_tgt: Target vocabulary (Spanish)
        device: torch.device
        max_length: Maximum translation length
    Returns:
        List of translated tokens
    """

    # Load English tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Tokenize input sentence (string or already-tokenized list)
    if isinstance(sentence, str):
        tokens = [tok.text.lower() for tok in spacy_eng.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add start and end tokens
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    # Convert tokens to indices
    src_indices = [vocab_src.stoi.get(token, vocab_src.unk_index) for token in tokens]

    # Create tensor: shape (sequence_length, 1)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)

    # Encode the input
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    # Start decoding with <sos> token
    outputs = [vocab_tgt.sos_index]

    for _ in range(max_length):
        prev_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(prev_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Stop if <eos> is generated
        if best_guess == vocab_tgt.eos_index:
            break

    translated_tokens = [vocab_tgt.itos[idx] for idx in outputs]

    # Remove initial <sos>
    return translated_tokens[1:]


# ============================================================
# BLEU score calculation without torchtext
# ============================================================

def compute_bleu_score(dataset, model, vocab_src, vocab_tgt, device):
    """
    Compute BLEU score on a dataset of (English, Spanish) sentence pairs.

    Args:
        dataset: Custom Dataset object (contains English, Spanish pairs)
        model: Trained Seq2Seq model
        vocab_src: Source (English) vocabulary
        vocab_tgt: Target (Spanish) vocabulary
        device: torch.device
    Returns:
        Average BLEU score (float)
    """

    smooth = SmoothingFunction().method4
    references = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            src = dataset.src[i]
            tgt = dataset.tgt[i]
            pred_tokens = translate_sentence(model, src, vocab_src, vocab_tgt, device)
            references.append([tgt.lower().split()])
            predictions.append(pred_tokens[:-1])  # exclude <eos>

    # Compute mean BLEU across samples
    bleu_scores = [
        sentence_bleu(ref, pred, smoothing_function=smooth) for ref, pred in zip(references, predictions)
    ]
    return sum(bleu_scores) / len(bleu_scores)


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save training state (model + optimizer)."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """Load training state (model + optimizer)."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
