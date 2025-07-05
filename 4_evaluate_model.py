import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score

from main.han_model import HierarchicalAttentionNetwork

nltk.download("punkt")

# ---- Paths ----
MODEL_SAVE_PATH = "data/new_best_han_model.pth"
VOCAB_PATH = "data/new_vocabulary2.pkl"
TESTSET_PATH = "data/testset.csv"

# ---- Load model ----
def load_trained_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = HierarchicalAttentionNetwork(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Best validation accuracy: {checkpoint.get('highest_validation_accuracy', 'N/A')}")
    return model, checkpoint["model_config"], checkpoint, device

# ---- Load vocabulary ----
def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# ---- Preprocess function ----
def preprocess_text(document, vocab, max_sent_len=30, max_sent_num=10):
    PAD_IDX = vocab.get("<PAD>", 0)
    UNK_IDX = vocab.get("<UNK>", 1)

    sentences = sent_tokenize(document.lower())[:max_sent_num]
    tokenized = []
    word_lengths = []

    for sent in sentences:
        words = word_tokenize(sent)
        if not words:
            words = ["<PAD>"]  # fallback if tokenization fails

        indexed = [vocab.get(word, UNK_IDX) for word in words[:max_sent_len]]
        word_len = len(indexed)

        # Pad sentence to max_sent_len
        if word_len < max_sent_len:
            indexed += [PAD_IDX] * (max_sent_len - word_len)

        tokenized.append(indexed)
        word_lengths.append(max(1, word_len))  # Make sure word_len ≥ 1

    # sentence_length = len(tokenized)
    sentence_length = max(1, len(tokenized))  # Ensure at least one sentence


    # Pad the document with dummy sentences if fewer than max_sent_num
    while len(tokenized) < max_sent_num:
        tokenized.append([PAD_IDX] * max_sent_len)
        word_lengths.append(1)  # dummy sentence has length 1

    return (
        torch.tensor(tokenized, dtype=torch.long),        # (max_sent_num, max_sent_len)
        torch.tensor(word_lengths, dtype=torch.long),     # (max_sent_num,)
        torch.tensor(min(sentence_length, max_sent_num), dtype=torch.long)  # scalar
    )


# ---- Dataset class ----
class HANTestDataset(Dataset):
    def __init__(self, dataframe, vocab, max_sent_len=30, max_sent_num=10):
        self.data = dataframe
        self.vocab = vocab
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.label2idx = {label: idx for idx, label in enumerate(sorted(dataframe["label"].unique()))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        label = self.label2idx[row["label"]]
        
        # if any(w == 0 for w in word_lengths):
            # print("⚠️ Zero word length detected:", row["text"][:100])


        inputs, word_lengths, sentence_length = preprocess_text(
            text, self.vocab, self.max_sent_len, self.max_sent_num
        )
        return inputs, word_lengths, sentence_length, label

# ---- Collate function ----
def collate_fn(batch):
    inputs, word_lengths, sentence_lengths, labels = zip(*batch)

    inputs = torch.stack(inputs)                     # (batch, max_sent, max_word)
    word_lengths = torch.stack(word_lengths)         # (batch, max_sent)
    sentence_lengths = torch.stack(sentence_lengths) # (batch,)
    labels = torch.tensor(labels, dtype=torch.long)

    return inputs, word_lengths, sentence_lengths, labels

# ---- Evaluation ----
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, word_lengths, sentence_lengths, labels in dataloader:
            inputs = inputs.to(device)
            word_lengths = word_lengths.to(device)
            sentence_lengths = sentence_lengths.to(device)
            labels = labels.to(device)

            outputs = model(inputs, word_lengths, sentence_lengths)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# ---- Main ----
if __name__ == "__main__":
    # Load model and vocab
    model, config, checkpoint, device = load_trained_model(MODEL_SAVE_PATH)
    vocab = load_vocab(VOCAB_PATH)

    # Load test data
    test_df = pd.read_csv(TESTSET_PATH)

    # Dataset and DataLoader
    test_dataset = HANTestDataset(test_df, vocab)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Evaluate
    evaluate(model, test_loader, device)

