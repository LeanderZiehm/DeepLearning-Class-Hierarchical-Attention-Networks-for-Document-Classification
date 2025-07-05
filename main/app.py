import os
import pickle
import torch
from flask import Flask, request, jsonify, render_template
from nltk.tokenize import sent_tokenize, word_tokenize
from pipeline_all import TextPreprocessor, HierarchicalAttentionNetwork
from pipeline_all import create_padding_for_hierarchical_sequences

app = Flask(__name__)

MODEL_SAVE_PATH = "files/han_model2025-07-06_01-13-47.pth"
VOCAB_PATH = "files/vocabulary_2025-07-06_01-13-47.pkl"

# Load model & vocab
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'),weights_only=False)
model_config = checkpoint['model_config']
model = HierarchicalAttentionNetwork(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)

word_to_idx = vocab_data['word_to_idx']
preprocessor = TextPreprocessor()
preprocessor.word_to_idx = word_to_idx

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")

    cleaned_text = preprocessor.clean_text(input_text)
    sentences = sent_tokenize(cleaned_text)[:20]

    hierarchical_doc = []
    original_words = []

    for sentence in sentences:
        words = word_tokenize(sentence)[:50]
        original_words.append(words)
        word_indices = [preprocessor.word_to_idx.get(word, 1) for word in words]
        hierarchical_doc.append(word_indices)

    padded_docs, word_lengths, sentence_lengths = create_padding_for_hierarchical_sequences([hierarchical_doc])

    with torch.no_grad():
        docs_tensor = torch.tensor(padded_docs, dtype=torch.long)
        word_lengths_tensor = torch.tensor(word_lengths, dtype=torch.long)
        sentence_lengths_tensor = torch.tensor(sentence_lengths, dtype=torch.long)
        logits, word_attentions, sentence_attentions = model(docs_tensor, word_lengths_tensor, sentence_lengths_tensor)

    prediction = torch.argmax(logits, dim=1).item()

    # Normalize attentions
    sentence_attentions = sentence_attentions[0].tolist()
    word_attentions = [att.tolist() for att in word_attentions[0]]  # list of word-level attentions

    response = {
        "prediction": prediction,
        "sentences": original_words,
        "sentence_attentions": sentence_attentions,
        "word_attentions": word_attentions
    }

    return jsonify(response)


import json

@app.route("/samples")
def get_sample_texts():
    with open("files/sample_texts.json", "r") as f:
        samples = json.load(f)
    return jsonify(samples)

@app.route("/tag", methods=["POST"])
def tag_text():
    data = request.get_json()
    text_id = data.get("id")
    tag = data.get("tag")

    # Optional: Save to a local file
    with open("tagged_data.json", "a") as f:
        f.write(json.dumps({"id": text_id, "tag": tag}) + "\n")

    return jsonify({"message": "Tag saved", "id": text_id, "tag": tag})


if __name__ == "__main__":
    app.run(debug=True)
