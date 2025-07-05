import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import datetime


# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")



current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# CONFIGURATION CONSTANTS
DATA_PATH = "data/"
MODEL_SAVE_PATH = f"files/han_model{current_date}.pth"
VOCAB_PATH = f"files/vocabulary_{current_date}.pkl"
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16
TEST_SIZE = 0.2
MAX_SENTENCES = 20
MAX_WORDS_PER_SENTENCE = 50
MAX_VOCAB_SIZE = 50000
MIN_WORD_FREQ = 2
PATIENCE = 3
RANDOM_STATE = 42



class WordAttention(nn.Module):
    """
    Word-level attention mechanism as described in the HAN paper.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dimmentions,
        word_gru_hidden_units,
        word_gru_layers,
        word_attention_dimmentions,
    ):
        super(WordAttention, self).__init__()

        # Word embedding layer
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimmentions, padding_idx=0
        )

        # Bidirectional GRU for word-level encoding
        self.word_gru = nn.GRU(
            embedding_dimmentions,
            word_gru_hidden_units,
            num_layers=word_gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if word_gru_layers > 1 else 0,
        )

        # Word attention mechanism
        self.word_attention = nn.Linear(
            2 * word_gru_hidden_units, word_attention_dimmentions
        )
        self.word_context_vector = nn.Linear(word_attention_dimmentions, 1, bias=False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, sentences, word_lengths):

        batch_size, max_sentence_length, max_word_length = sentences.size()

        # Reshape to process all sentences together
        sentences = sentences.view(-1, max_word_length)
        word_lengths = word_lengths.view(-1)

        # Remove sentences with zero length
        valid_sentences = word_lengths > 0
        if not valid_sentences.any():
            # Handle case where all sentences are empty
            zero_output = torch.zeros(
                batch_size,
                max_sentence_length,
                2 * self.word_gru.hidden_size,
                device=sentences.device,
            )
            zero_weights = torch.zeros(
                batch_size,
                max_sentence_length,
                max_word_length,
                device=sentences.device,
            )
            return zero_output, zero_weights

        # Process only valid sentences
        valid_sentences_data = sentences[valid_sentences]
        valid_word_lengths = word_lengths[valid_sentences]

        # Word embeddings
        embedded = self.embedding(valid_sentences_data)
        embedded = self.dropout(embedded)

        # Pack sequences for efficient RNN processing
        packed_embedded = pack_padded_sequence(
            embedded, valid_word_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Bidirectional GRU
        packed_output, _ = self.word_gru(packed_embedded)
        word_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Word attention mechanism
        attention_weights = torch.tanh(self.word_attention(word_outputs))
        attention_weights = self.word_context_vector(attention_weights).squeeze(-1)
        # Create attention mask for padding tokens
        word_mask = (
            torch.arange(word_outputs.size(1), device=word_outputs.device)[None, :]
            < valid_word_lengths[:, None]
        )
        attention_weights = attention_weights.masked_fill(~word_mask, -float("inf"))
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights
        sentence_vectors = torch.sum(
            word_outputs * attention_weights.unsqueeze(-1), dim=1
        )

        # Reconstruct full batch including zero vectors for invalid sentences
        full_sentence_vectors = torch.zeros(
            batch_size * max_sentence_length,
            sentence_vectors.size(-1),
            device=sentences.device,
        )
        full_attention_weights = torch.zeros(
            batch_size * max_sentence_length, max_word_length, device=sentences.device
        )

        full_sentence_vectors[valid_sentences] = sentence_vectors
        full_attention_weights[valid_sentences] = attention_weights

        # Reshape back to original dimensions
        sentence_vectors = full_sentence_vectors.view(
            batch_size, max_sentence_length, -1
        )
        word_attention_weights = full_attention_weights.view(
            batch_size, max_sentence_length, max_word_length
        )

        return sentence_vectors, word_attention_weights


class SentenceAttention(nn.Module):
    """
    Sentence-level attention mechanism as described in the HAN paper.
    """

    def __init__(
        self,
        word_gru_hidden_units,
        sentence_gru_hidden_units,
        sentence_gru_layers,
        sentence_attention_dimmention,
        number_of_classes,
    ):
        super(SentenceAttention, self).__init__()

        # Bidirectional GRU for sentence-level encoding
        self.sentence_gru = nn.GRU(
            2 * word_gru_hidden_units,
            sentence_gru_hidden_units,
            num_layers=sentence_gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if sentence_gru_layers > 1 else 0,
        )

        # Sentence attention mechanism
        self.sentence_attention = nn.Linear(
            2 * sentence_gru_hidden_units, sentence_attention_dimmention
        )
        self.sentence_context_vector = nn.Linear(
            sentence_attention_dimmention, 1, bias=False
        )

        # Final classification layer
        self.classifier = nn.Linear(2 * sentence_gru_hidden_units, number_of_classes)

        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_vectors, sentence_lengths):

        sentence_vectors = self.dropout(sentence_vectors)

        # Pack sequences for efficient RNN processing
        packed_sentences = pack_padded_sequence(
            sentence_vectors,
            sentence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Bidirectional GRU
        packed_output, _ = self.sentence_gru(packed_sentences)
        sentence_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Sentence attention mechanism
        attention_weights = torch.tanh(self.sentence_attention(sentence_outputs))
        attention_weights = self.sentence_context_vector(attention_weights).squeeze(-1)

        # Create attention mask for padding sentences
        sentence_mask = (
            torch.arange(sentence_outputs.size(1), device=sentence_outputs.device)[
                None, :
            ]
            < sentence_lengths[:, None]
        )
        attention_weights = attention_weights.masked_fill(~sentence_mask, -float("inf"))
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights to get document representation
        doc_vectors = torch.sum(
            sentence_outputs * attention_weights.unsqueeze(-1), dim=1
        )

        # Final classification
        logits = self.classifier(doc_vectors)

        return logits, attention_weights


class HierarchicalAttentionNetwork(nn.Module):
    """
    Complete Hierarchical Attention Network implementation.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dimmentions,
        word_gru_hidden_units,
        word_gru_layers,
        word_attention_dimmentions,
        sentence_gru_hidden_units,
        sentence_gru_layers,
        sentence_attention_dimmention,
        number_of_classes,
        pretrained_embeddings=None,
    ):
        super(HierarchicalAttentionNetwork, self).__init__()

        self.word_attention = WordAttention(
            vocabulary_size,
            embedding_dimmentions,
            word_gru_hidden_units,
            word_gru_layers,
            word_attention_dimmentions,
        )

        self.sentence_attention = SentenceAttention(
            word_gru_hidden_units,
            sentence_gru_hidden_units,
            sentence_gru_layers,
            sentence_attention_dimmention,
            number_of_classes,
        )

        # Initialize embeddings with pre-trained vectors if provided
        if pretrained_embeddings is not None:
            self.word_attention.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )
            # Freeze embedding layer if desired
            # self.word_attention.embedding.weight.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "embedding" not in name:  # Don't reinitialize embeddings
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

    def forward(self, documents, word_lengths, sentence_lengths):
        # Word-level attention
        sentence_vectors, word_attention_weights = self.word_attention(
            documents, word_lengths
        )

        # Sentence-level attention
        logits, sentence_attention_weights = self.sentence_attention(
            sentence_vectors, sentence_lengths
        )

        return logits, word_attention_weights, sentence_attention_weights


# Example usage and training utilities
class HANTrainer:
    """Training utilities for Hierarchical Attention Network."""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def trainset_step(self, batch, optimizer):
        """Single training step."""
        self.model.train()
        documents, word_lengths, sentence_lengths, labels = batch

        # Move to device
        documents = documents.to(self.device)
        word_lengths = word_lengths.to(self.device)
        sentence_lengths = sentence_lengths.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()

        # Forward pass
        logits, _, _ = self.model(documents, word_lengths, sentence_lengths)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        optimizer.step()

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()

        return loss.item(), accuracy.item()

    def evaluate(self, dataloader):
        """Evaluate model on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                documents, word_lengths, sentence_lengths, labels = batch

                # Move to device
                documents = documents.to(self.device)
                word_lengths = word_lengths.to(self.device)
                sentence_lengths = sentence_lengths.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits, _, _ = self.model(documents, word_lengths, sentence_lengths)

                # Compute loss and accuracy
                loss = self.criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        return total_loss / num_batches, total_accuracy / num_batches


# Creates the Hierarchical Attention Network
def create_han_model(
    vocabulary_size=10000, embedding_dimmentions=200, number_of_classes=5
):

    model = HierarchicalAttentionNetwork(
        vocabulary_size=vocabulary_size,
        embedding_dimmentions=embedding_dimmentions,
        word_gru_hidden_units=50,
        word_gru_layers=1,
        word_attention_dimmentions=100,
        sentence_gru_hidden_units=50,
        sentence_gru_layers=1,
        sentence_attention_dimmention=100,
        number_of_classes=number_of_classes,
    )
    return model



if __name__ == "__main__":

    # Model parameters
    vocabulary_size = 10000
    embedding_dimmentions = 200
    number_of_classes = 5

    # Create model
    model = create_han_model(vocabulary_size, embedding_dimmentions, number_of_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    batch_size = 2
    max_sentence_length = 3
    max_word_length = 5

    # Create dummy data
    documents = torch.randint(
        1, vocabulary_size, (batch_size, max_sentence_length, max_word_length)
    )
    word_lengths = torch.randint(
        1, max_word_length + 1, (batch_size, max_sentence_length)
    )
    sentence_lengths = torch.randint(1, max_sentence_length + 1, (batch_size,))

    print("\nTesting forward pass")
    with torch.no_grad():
        logits, word_att, sentence_att = model(
            documents, word_lengths, sentence_lengths
        )

    print(f"Output logits shape: {logits.shape}")
    print(f"Word attention weights shape: {word_att.shape}")
    print(f"Sentence attention weights shape: {sentence_att.shape}")
    print("\nHierarchical Attention Network implementation complete")


class TextPreprocessor:
    """Text preprocessing utilities for hierarchical document structure."""

    def __init__(self, maximum_vocabulary_size=50000, minimum_word_frequency=2):
        self.maximum_vocabulary_size = maximum_vocabulary_size
        self.minimum_word_frequency = minimum_word_frequency
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocabulary_size = 0

    def clean_text(self, text):
        """Clean and normalize text."""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r"[^\w\s.,!?;:]", " ", text)

        return text.strip()

    def build_vocabulary(self, texts):
        """Build vocabulary from training texts."""
        print("Building vocabulary")
        word_counts = Counter()

        for text in tqdm(texts, desc="Processing texts"):
            cleaned_text = self.clean_text(text)
            sentences = sent_tokenize(cleaned_text)

            for sentence in sentences:
                words = word_tokenize(sentence)
                word_counts.update(words)

        # Create word-to-index mapping
        # Reserve indices: 0=padding, 1=unknown
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}

        # Add words that meet frequency threshold
        valid_words = [
            word
            for word, count in word_counts.items()
            if count >= self.minimum_word_frequency
        ]

        # Sort by frequency and take top words
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[
            : self.maximum_vocabulary_size - 2
        ]  # -2 for PAD and UNK

        for idx, word in enumerate(valid_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocabulary_size = len(self.word_to_idx)
        print(f"Vocabulary size: {self.vocabulary_size}")

    def text_to_hierarchical_indices(
        self, text, max_sentences=20, max_words_per_sentence=50
    ):
        """Convert text to hierarchical structure of word indices."""
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)

        # Limit number of sentences
        sentences = sentences[:max_sentences]

        hierarchical_doc = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            # Limit words per sentence
            words = words[:max_words_per_sentence]

            # Convert words to indices
            word_indices = []
            for word in words:
                idx = self.word_to_idx.get(word, 1)  # 1 is <UNK>
                word_indices.append(idx)

            if word_indices:  # Only add non-empty sentences
                hierarchical_doc.append(word_indices)

        return hierarchical_doc if hierarchical_doc else [[1]]  # Return <UNK> if empty


    def save_vocabulary(self, filepath):
        """Save vocabulary to file."""
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "vocabulary_size": self.vocabulary_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(vocab_data, f)


class FakeNewsDataset(Dataset):
    """Dataset class for fake news classification."""

    def __init__(
        self, texts, labels, preprocessor: TextPreprocessor, max_sentences=20, max_words_per_sentence=50
    ):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence

        # Convert texts to hierarchical format
        self.hierarchical_docs = []
        for text in tqdm(texts, desc="Convertng text to hierarchical format"):
            doc = preprocessor.text_to_hierarchical_indices(
                text, max_sentences, max_words_per_sentence
            )
            self.hierarchical_docs.append(doc)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.hierarchical_docs[idx], self.labels[idx]


def create_padding_for_hierarchical_sequences(
    documents, max_sentence_length=None, max_word_length=None, pad_token=0
):

    if max_sentence_length is None:
        max_sentence_length = max(len(doc) for doc in documents)
    if max_word_length is None:
        max_word_length = max(len(sent) for doc in documents for sent in doc)

    num_docs = len(documents)
    padded_docs = np.full(
        (num_docs, max_sentence_length, max_word_length), pad_token, dtype=np.int64
    )
    word_lengths = np.zeros((num_docs, max_sentence_length), dtype=np.int64)
    sentence_lengths = np.zeros(num_docs, dtype=np.int64)

    for i, doc in enumerate(documents):
        sentence_lengths[i] = min(len(doc), max_sentence_length)
        for j, sent in enumerate(doc[:max_sentence_length]):
            word_len = min(len(sent), max_word_length)
            word_lengths[i, j] = word_len
            padded_docs[i, j, :word_len] = sent[:word_len]

    return padded_docs, word_lengths, sentence_lengths


def collate_fn(batch):
    """Custom collate function for hierarchical data."""
    docs, labels = zip(*batch)

    # Pad hierarchical sequences
    padded_docs, word_lengths, sentence_lengths = create_padding_for_hierarchical_sequences(list(docs))

    return (
        torch.tensor(padded_docs, dtype=torch.long),
        torch.tensor(word_lengths, dtype=torch.long),
        torch.tensor(sentence_lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )




class ModelTrainer:
    """Complete training pipeline for HAN model."""

    def __init__(self, vocabulary_size, number_of_classes, label_encoder, device=None):
        self.vocabulary_size = vocabulary_size
        self.number_of_classes = number_of_classes
        self.label_encoder = label_encoder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model hyperparameters
        self.model_config = {
            "vocabulary_size": vocabulary_size,
            "embedding_dimmentions": 200,
            "word_gru_hidden_units": 50,
            "word_gru_layers": 1,
            "word_attention_dimmentions": 100,
            "sentence_gru_hidden_units": 50,
            "sentence_gru_layers": 1,
            "sentence_attention_dimmention": 100,
            "number_of_classes": number_of_classes,
        }

        # Create model
        self.model = HierarchicalAttentionNetwork(**self.model_config)
        self.trainer = HANTrainer(self.model, self.device)

        print(
            f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        print(f"Using device: {self.device}")

    def train(
        self,
        trainset_loader,
        validationset_loader,
        number_of_epochs=10,
        learning_rate=0.001,
        save_path="best_han_model.pth",
        patience=3,
    ):
        # Training parameters
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

        # Tracking best validation accuracy
        highest_validation_accuracy = 0.0
        patience_counter = 0
        trainingset_history = []
        validationset_history = []

        print("Training started")
        print(f"Training samples:   {len(trainset_loader.dataset)}")
        print(f"Validation samples: {len(validationset_loader.dataset)}")

        for epoch in range(number_of_epochs):
            # 1) Training phase
            self.model.train()
            trainset_loss = 0.0
            trainset_acc = 0.0
            num_batches = 0

            progress_bar = tqdm(trainset_loader, desc=f"Epoch {epoch+1}/{number_of_epochs}")
            for batch in progress_bar:
                loss, acc = self.trainer.trainset_step(batch, optimizer)
                trainset_loss += loss
                trainset_acc += acc
                num_batches += 1

                progress_bar.set_postfix({"Loss": f"{loss:.4f}", "Acc": f"{acc:.4f}"})

            avg_trainset_loss = trainset_loss / num_batches
            avg_trainset_acc = trainset_acc / num_batches

            # 2) Validation phase
            validationset_loss, validationset_accuracy = self.trainer.evaluate(validationset_loader)

            # 3) Scheduler step
            scheduler.step()

            # 4) Record history
            trainingset_history.append(
                {"loss": avg_trainset_loss, "accuracy": avg_trainset_acc}
            )
            validationset_history.append({"loss": validationset_loss, "accuracy": validationset_accuracy})

            # 5) Print epoch metrics
            print(f"\nEpoch {epoch+1}/{number_of_epochs}:")
            print(f"Train Loss: {avg_trainset_loss:.4f}, Train Accuracy: {avg_trainset_acc:.4f}")
            print(f"Validation Loss: {validationset_loss:.4f}, Validation   Accuracy: {validationset_accuracy:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # 6) Check for improvement
            if validationset_accuracy > highest_validation_accuracy:
                highest_validation_accuracy = validationset_accuracy
                patience_counter = 0

                # Save model checkpoint only if a valid path is provided
                if save_path is not None:
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "model_config": self.model_config,
                        "highest_validation_accuracy": highest_validation_accuracy,
                        "epoch": epoch + 1,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "label_encoder": self.label_encoder,  # ensure this exists on self
                    }
                    torch.save(checkpoint, save_path)
                    print(f"New best model saved Validation Accuracy: {validationset_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            # 7) Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            print()

        print(
            f"Training completed with the best validation accuracy being: {highest_validation_accuracy:.4f}"
        )

        # 8) Load best model only if save_path is not None
        if save_path is not None:
            self.load_model(save_path)

        return {
            "highest_validation_accuracy": highest_validation_accuracy,
            "trainingset_history": trainingset_history,
            "validationset_history": validationset_history,
            "total_epochs": epoch + 1,
        }


    def load_model(self, model_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {model_path}")
        return checkpoint
    
class FakeNewsAnalyzer:
    """Comprehensive fake news detection analysis pipeline."""

    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.trainset_data = None
        self.validationset_data = None
        self.test_data = None
        self.model = None
        self.training_results = None

    #################
    # Generatl utils
    #################

    def load_and_prepare_data(self):
        print("Loading data")
        fake = pd.read_csv(os.path.join(DATA_PATH, "Fake.csv"))
        true = pd.read_csv(os.path.join(DATA_PATH, "True.csv"))
        fake["label"] = "fake"
        true["label"] = "true"
        data = pd.concat([fake, true], ignore_index=True)
        trainset_data, temp_data = train_test_split(
            data, test_size=TEST_SIZE, stratify=data["label"], random_state=RANDOM_STATE
        )
        validationset_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            stratify=temp_data["label"],
            random_state=RANDOM_STATE,
        )
        self.trainset_data, self.validationset_data, self.test_data = trainset_data, validationset_data, test_data
        self._print_data_info()
        return trainset_data, validationset_data, test_data

    def _print_data_info(self):
        """Print dataset information."""
        print(f"Dataset splits:")
        print(f"Train: {len(self.trainset_data)} samples")
        print(f"Validation: {len(self.validationset_data)} samples")
        print(f"Test: {len(self.test_data)} samples")
        print("\nLabel distribution:")
        for name, data in [
            ("Train", self.trainset_data),
            ("Validation", self.validationset_data),
            ("Test", self.test_data),
        ]:
            print(f"{name} set:")
            print(data["label"].value_counts())

    def prepare_preprocessing(self):
        print("Building vocabulary")
        self.preprocessor = TextPreprocessor(
            maximum_vocabulary_size=MAX_VOCAB_SIZE, minimum_word_frequency=MIN_WORD_FREQ
        )
        self.preprocessor.build_vocabulary(self.trainset_data["text"].tolist())
        self.preprocessor.save_vocabulary(VOCAB_PATH)
        print("Encoding labels")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.trainset_data["label"])
   
        print(f"Vocabulary size: {self.preprocessor.vocabulary_size}")
        print(
            f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}"
        )

    def create_datasets_and_loaders(self, data_splits=None):
        print("Creating datasets and data loaders")
        if data_splits is None:
            data_splits = (self.trainset_data, self.validationset_data, self.test_data)
        trainset_data, validationset_data, test_data = data_splits
        # Create dataset objects
        datasets = []
        for data in [trainset_data, validationset_data, test_data]:
            texts = data["text"].tolist()
            labels = self.label_encoder.transform(data["label"])
            dataset = FakeNewsDataset(
                texts, labels, self.preprocessor, MAX_SENTENCES, MAX_WORDS_PER_SENTENCE
            )
            datasets.append(dataset)
        # Create data loaders
        trainset_loader = DataLoader(
            datasets[0],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        validationset_loader = DataLoader(
            datasets[1],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
        test_loader = DataLoader(
            datasets[2],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
        return datasets, (trainset_loader, validationset_loader, test_loader)

    def trainset_model(self, trainset_loader, validationset_loader):
        print("Training HAN model")
        trainer = ModelTrainer(
            vocabulary_size=self.preprocessor.vocabulary_size,
            number_of_classes=len(self.label_encoder.classes_),
            label_encoder=self.label_encoder,
        )
        self.training_results = trainer.train(
            trainset_loader=trainset_loader,
            validationset_loader=validationset_loader,
            number_of_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_path=MODEL_SAVE_PATH,
            patience=PATIENCE,
        )
        # Save additional metadata
        self._save_model_metadata(trainer)
        self.model = trainer.model
        return trainer

    def _save_model_metadata(self, trainer):
        additional_data = {
            "label_encoder": self.label_encoder,
            "word_to_idx": self.preprocessor.word_to_idx,  # Store the actual vocabulary dict
            "vocabulary_size": self.preprocessor.vocabulary_size,
            "number_of_classes": len(self.label_encoder.classes_),
            "training_results": self.training_results,
            "model_config": trainer.model_config,  # <-- Required to reinit model
            "MAX_WORDS_PER_SENTENCE": MAX_WORDS_PER_SENTENCE,
            "MAX_SENTENCES": MAX_SENTENCES
        }

        checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
        checkpoint.update(additional_data)
        torch.save(checkpoint, MODEL_SAVE_PATH)



    ##################
    # Actual pipeline
    ##################

    def run_complete_analysis(self):
        print("Starting Fake News Detection Analysis Pipeline")
        # 1. Data preparation
        self.load_and_prepare_data()
        self.prepare_preprocessing()
        # 2. Data loaders
        datasets, (trainset_loader, validationset_loader, test_loader) = (self.create_datasets_and_loaders())
        trainer = self.trainset_model(trainset_loader, validationset_loader)


# Initialize analyzer
analyzer = FakeNewsAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis()

print("\nRESULTS\n")
print(results)