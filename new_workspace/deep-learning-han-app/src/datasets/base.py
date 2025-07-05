class BaseDataset:
    """Base class for datasets, providing common functionality for loading and processing datasets."""

    def __init__(self, texts, labels, preprocessor, max_sentences=20, max_words_per_sentence=50):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess_text(self, text):
        """Preprocess the text using the provided preprocessor."""
        return self.preprocessor.text_to_hierarchical_indices(text, self.max_sentences, self.max_words_per_sentence)