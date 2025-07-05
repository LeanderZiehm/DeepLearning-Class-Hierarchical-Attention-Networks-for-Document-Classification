from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

class TextPreprocessor:
    def __init__(self, maximum_vocabulary_size=50000, minimum_word_frequency=2):
        self.maximum_vocabulary_size = maximum_vocabulary_size
        self.minimum_word_frequency = minimum_word_frequency
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocabulary_size = 0

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:]", " ", text)
        return text.strip()

    def build_vocabulary(self, texts):
        word_counts = Counter()
        for text in texts:
            cleaned_text = self.clean_text(text)
            sentences = sent_tokenize(cleaned_text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                word_counts.update(words)

        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}

        valid_words = [
            word for word, count in word_counts.items() if count >= self.minimum_word_frequency
        ]
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[:self.maximum_vocabulary_size - 2]

        for idx, word in enumerate(valid_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocabulary_size = len(self.word_to_idx)

    def text_to_hierarchical_indices(self, text, max_sentences=20, max_words_per_sentence=50):
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)
        sentences = sentences[:max_sentences]

        hierarchical_doc = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = words[:max_words_per_sentence]
            word_indices = [self.word_to_idx.get(word, 1) for word in words]

            if word_indices:
                hierarchical_doc.append(word_indices)

        return hierarchical_doc if hierarchical_doc else [[1]]

    def save_vocabulary(self, filepath):
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "vocabulary_size": self.vocabulary_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(vocab_data, f)