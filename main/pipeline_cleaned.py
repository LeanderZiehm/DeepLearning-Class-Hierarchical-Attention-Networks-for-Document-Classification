import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from src.data_preprocessor import TextPreprocessor, FakeNewsDataset, collate_fn
from src.han_model import HierarchicalAttentionNetwork
from src.model_trainer import ModelTrainer



# CONTROL CONSTANTS
ENABLE_TRAINING = True
ENABLE_VALIDATION = True
ENABLE_KFOLD_CV = True
ENABLE_BOOTSTRAP_VALIDATION = True
ENABLE_LEARNING_CURVE = False
SAVE_PLOTS = True


# CONFIGURATION CONSTANTS
DATA_PATH = "data/"
MODEL_SAVE_PATH = "files/new_best_han_model_clean.pth"
VOCAB_PATH = "files/vocabulary_clean.pkl"
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
K_FOLDS = 5
N_BOOTSTRAP = 100
MAX_SENTENCES = 20
MAX_WORDS_PER_SENTENCE = 50
MAX_VOCAB_SIZE = 50000
MIN_WORD_FREQ = 2
PATIENCE = 3
RANDOM_STATE = 42


def get_model_predictions(model, data_loader, label_encoder, device=None):
    """
    Get predictions for the whole dataset.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Getting predictions"):
            documents, word_lengths, sentence_lengths, labels = batch

            # Move to device
            documents = documents.to(device)
            word_lengths = word_lengths.to(device)
            sentence_lengths = sentence_lengths.to(device)

            # Forward pass
            logits, _, _ = model(documents, word_lengths, sentence_lengths)

            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to label names
    predicted_labels = label_encoder.inverse_transform(all_predictions)
    true_labels = label_encoder.inverse_transform(all_true_labels)

    return {
        "predictions": predicted_labels,
        "true_labels": true_labels,
        "probabilities": all_probabilities,
        "prediction_indices": all_predictions,
        "true_label_indices": all_true_labels,
    }
    
    
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
            "vocabulary_size": self.preprocessor.vocabulary_size,
            "number_of_classes": len(self.label_encoder.classes_),
            "training_results": self.training_results,
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