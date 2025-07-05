from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    """Dataset class for fake news classification."""

    def __init__(self, texts, labels, preprocessor, max_sentences=20, max_words_per_sentence=50):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence

        # Convert texts to hierarchical format
        self.hierarchical_docs = []
        for text in tqdm(texts, desc="Converting text to hierarchical format"):
            doc = preprocessor.text_to_hierarchical_indices(
                text, max_sentences, max_words_per_sentence
            )
            self.hierarchical_docs.append(doc)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.hierarchical_docs[idx], self.labels[idx]