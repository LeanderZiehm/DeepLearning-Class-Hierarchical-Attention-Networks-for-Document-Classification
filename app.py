import torch
import re
import pickle
from flask import Flask, render_template, request, jsonify
from han_model import (
    create_padding_for_hierarchical_sequences,
    HierarchicalAttentionNetwork,
)


app = Flask(__name__)

def load_model_and_vocab():
    global model, word2idx
    model = HierarchicalAttentionNetwork(
        vocabulary_size=50000,
        embedding_dimmentions=200,
        word_gru_hidden_units=50,
        word_gru_layers=1,
        word_attention_dimmentions=100,
        sentence_gru_hidden_units=50,
        sentence_gru_layers=1,
        sentence_attention_dimmention=100,
        number_of_classes=2,
    )
    ckpt = torch.load("files/best_han_model.pth", map_location="cpu", weights_only=False)
    fixed_sd = {}
    for k, v in ckpt["model_state_dict"].items():
        new_k = (
            k.replace("sentence_attention.sent_gru", "sentence_attention.sentence_gru")
            .replace("sentence_attention.sent_attention", "sentence_attention.sentence_attention")
            .replace("sentence_attention.sent_context_vector", "sentence_attention.sentence_context_vector")
        )
        fixed_sd[new_k] = v
    model.load_state_dict(fixed_sd)
    model.eval()

    with open("files/vocabulary.pkl", "rb") as f:
        word2idx = pickle.load(f)["word_to_idx"]

    return model, word2idx


def preprocess_text(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    tokenized = [s.split() for s in sentences]
    numericalized = [
        [word2idx.get(w.lower(), word2idx.get("<UNK>", 0)) for w in sent]
        for sent in tokenized
    ]
    return tokenized, numericalized


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    tokenized, numericalized = preprocess_text(text)

    docs, word_lens, sent_lens = create_padding_for_hierarchical_sequences([numericalized])
    docs = torch.LongTensor(docs)
    word_lens = torch.LongTensor(word_lens)
    sent_lens = torch.LongTensor(sent_lens)

    with torch.no_grad():
        logits, word_attn, sent_attn = model(docs, word_lens, sent_lens)

    prediction = torch.argmax(logits, dim=1).item()

    word_attn = word_attn.squeeze(0).cpu().tolist()
    sent_attn = sent_attn.squeeze(0).cpu().tolist()

    data = {
        "prediction": prediction,
        "sentence_attention": sent_attn,
        "sentences": []
    }

    for i, (sentence, attention) in enumerate(zip(tokenized, word_attn)):
        word_data = [{"word": w, "attention": a} for w, a in zip(sentence, attention)]
        data["sentences"].append({"words": word_data})

    return jsonify(data)




if __name__ == "__main__":
    # Load once at startup
    model, word2idx = load_model_and_vocab()
    app.run(debug=True)
