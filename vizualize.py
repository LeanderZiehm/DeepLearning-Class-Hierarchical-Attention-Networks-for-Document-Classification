import torch
import matplotlib.pyplot as plt
from src.han_model import create_padding_for_hierarchical_sequences, HierarchicalAttentionNetwork
from sklearn.preprocessing._label import LabelEncoder
import re
import pickle

def main():

    # Example text (single document with multiple sentences)
    text = "Hello world! This is an example sentence. Another sentence follows."
    
    model = load_model()
    prediction = predict(text,model)
    print(f"Prediction: {prediction}")

def predict(text, model):
    tokenized, numericalized = text2numbers(text)  # Fix: Unpack both values
    word_attn, sent_attn = calc_attention(numericalized, model)  # Fix: Pass numericalized instead of text
    # Do something meaningful here — this is just a placeholder
    predicted_class = torch.argmax(torch.tensor(sent_attn)).item()
    return predicted_class


def load_model():
    # Reconstruct the model and load weights (as previously done)
    model = HierarchicalAttentionNetwork(
        vocabulary_size=50000,
        embedding_dimmentions=200,
        word_gru_hidden_units=50,
        word_gru_layers=1,
        word_attention_dimmentions=100,
        sentence_gru_hidden_units=50,
        sentence_gru_layers=1,
        sentence_attention_dimmention=100,
        number_of_classes=2
    )
    torch.serialization.add_safe_globals([LabelEncoder])
    ckpt     = torch.load("best_han_model.pth", map_location="cpu", weights_only=False)
    raw_sd   = ckpt["model_state_dict"]
    fixed_sd = {}
    for k, v in raw_sd.items():
        new_k = (
            k
            .replace("sentence_attention.sent_gru",            "sentence_attention.sentence_gru")
            .replace("sentence_attention.sent_attention",      "sentence_attention.sentence_attention")
            .replace("sentence_attention.sent_context_vector", "sentence_attention.sentence_context_vector")
        )
        fixed_sd[new_k] = v
    model.load_state_dict(fixed_sd)
    model.eval()
    return model


def text2numbers(text):

    word2idx = load_word2idx()
    # 1. Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 2. Tokenize words in each sentence
    tokenized = [s.split() for s in sentences]
    
    # 3. Convert tokens to indices, use <UNK> if word not found
    numericalized = [
        [word2idx.get(w.lower(), word2idx.get("<UNK>", 0)) for w in sent]
        for sent in tokenized
    ]
    
    return tokenized, numericalized



def load_word2idx(vocab_path="vocabulary.pkl"):
    with open("vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)
        word2idx = vocab['word_to_idx']  # Ensure you're accessing the correct dictionary
    return word2idx


def calc_attention(numericalized,model):
    # Pad
    docs, word_lens, sent_lens = create_padding_for_hierarchical_sequences([numericalized])
    docs      = torch.LongTensor(docs)
    word_lens = torch.LongTensor(word_lens)
    sent_lens = torch.LongTensor(sent_lens)

    # Forward pass
    with torch.no_grad():
        logits, word_attn, sent_attn = model(docs, word_lens, sent_lens)

    word_attn = word_attn.squeeze(0).cpu().numpy()
    sent_attn = sent_attn.squeeze(0).cpu().numpy()

    return word_attn,sent_attn


if __name__ == "__main__":
    main()