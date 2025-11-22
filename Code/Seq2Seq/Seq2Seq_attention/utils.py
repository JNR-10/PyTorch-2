import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================
# Translate a single sentence (English → Spanish)
# ============================================================

def translate_sentence(model, sentence, vocab_src, vocab_tgt, device, max_length=50):
    """
    Translate a given English sentence using the trained Seq2Seq model.

    Args:
        model: Trained Seq2Seq model
        sentence: Input sentence (string or list of tokens)
        vocab_src: Source vocabulary (English)
        vocab_tgt: Target vocabulary (Spanish)
        device: torch.device
        max_length: Maximum translation length
    Returns:
        List of translated tokens (excluding <sos>)
    """

    # Load English tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Tokenize sentence if needed
    if isinstance(sentence, str):
        tokens = [tok.text.lower() for tok in spacy_eng.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <sos> and <eos> tokens
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    # Convert tokens to indices using source vocab
    src_indices = [vocab_src.stoi.get(tok, vocab_src.unk_index) for tok in tokens]

    # Convert to tensor and move to device
    sentence_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)

    # Encode the sentence
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(sentence_tensor)

    # Start decoding with <sos> token
    outputs = [vocab_tgt.sos_index]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_outputs, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Stop if <eos> predicted
        if best_guess == vocab_tgt.eos_index:
            break

    # Convert predicted indices back to tokens
    translated_tokens = [vocab_tgt.itos[idx] for idx in outputs]

    # Remove initial <sos>
    return translated_tokens[1:]


# ============================================================
# BLEU score evaluation without torchtext
# ============================================================

def compute_bleu(dataset, model, vocab_src, vocab_tgt, device):
    """
    Compute BLEU score for model translations on a dataset.

    Args:
        dataset: TranslationDataset containing English–Spanish pairs
        model: Trained Seq2Seq model
        vocab_src: Source (English) vocabulary
        vocab_tgt: Target (Spanish) vocabulary
        device: torch.device
    Returns:
        Average BLEU score (float)
    """

    smooth = SmoothingFunction().method4
    references, predictions = [], []

    model.eval()
    with torch.no_grad():
        for i in range(min(len(dataset), 100)):
            src_sentence = dataset.src[i]
            tgt_sentence = dataset.tgt[i]

            # Predict translation
            pred_tokens = translate_sentence(model, src_sentence, vocab_src, vocab_tgt, device)
            predictions.append(pred_tokens[:-1])  # exclude <eos>

            # Reference (tokenized Spanish sentence)
            reference = [tok.text.lower() for tok in spacy.load("es_core_news_sm")(tgt_sentence)]
            references.append([reference])

    bleu_scores = [
        sentence_bleu(ref, pred, smoothing_function=smooth)
        for ref, pred in zip(references, predictions)
    ]
    return sum(bleu_scores) / len(bleu_scores)


# ============================================================
# Checkpoint utilities (save/load)
# ============================================================

def save_checkpoint(state, filename="checkpoint_attention.pth.tar"):
    """Save model and optimizer states."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """Load model and optimizer states."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
