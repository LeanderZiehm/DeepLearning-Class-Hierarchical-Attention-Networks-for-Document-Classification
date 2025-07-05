import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from main.han_model import HierarchicalAttentionNetwork
from load_model import load_trained_model, VOCAB_PATH, MODEL_SAVE_PATH
import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize

# ----------------------------
# Custom Dataset
# ----------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, vocab, max_sentences=15, max_words=20):
        self.texts = texts
        self.vocab = vocab
        self.max_sentences = max_sentences
        self.max_words = max_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        article = self.texts[idx]
        encoded_article = []

        # Sentence tokenization
        sentences = sent_tokenize(article.lower())[:self.max_sentences]
        for sent in sentences:
            words = word_tokenize(sent)[:self.max_words]
            word_ids = [self.vocab.get(w, self.vocab.get('<UNK>', 1)) for w in words]
            # pad word ids
            word_ids += [0] * (self.max_words - len(word_ids))
            encoded_article.append(word_ids)

        # pad sentence
        while len(encoded_article) < self.max_sentences:
            encoded_article.append([0] * self.max_words)

        return torch.tensor(encoded_article)

# ----------------------------
# Load vocab and model
# ----------------------------
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

model, _, _ = load_trained_model(MODEL_SAVE_PATH)

# ----------------------------
# Load your CSV files
# ----------------------------
df_true = pd.read_csv("main/files/True.csv")
df_fake = pd.read_csv("main/files/Fake.csv")

# Assign labels
df_true["label"] = 1
df_fake["label"] = 0

df = pd.concat([df_true, df_fake], ignore_index=True)
df["text"] = df["title"] + " " + df["text"]  # optional

# ----------------------------
# Prepare Dataset and DataLoader
# ----------------------------
dataset = NewsDataset(df["text"].tolist(), vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ----------------------------
# Run Inference
# ----------------------------
model.eval()
predictions = []
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(next(model.parameters()).device)  # Move to GPU if available
        output = model(batch)
        preds = torch.argmax(output, dim=1)
        predictions.extend(preds.cpu().numpy())

# ----------------------------
# Evaluate (if true labels available)
# ----------------------------
from sklearn.metrics import accuracy_score, classification_report

true_labels = df["label"].tolist()
print("Accuracy:", accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions, target_names=["Fake", "True"]))

