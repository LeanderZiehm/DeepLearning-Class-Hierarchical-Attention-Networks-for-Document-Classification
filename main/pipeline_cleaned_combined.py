import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from src.data_preprocessor import TextPreprocessor, FakeNewsDataset, collate_fn
from src.han_model import HierarchicalAttentionNetwork, HANTrainer

# CONFIGURATION CONSTANTS
DATA_PATH = "data/"
MODEL_SAVE_PATH = "files/new_best_han_model_clean.pth"
VOCAB_PATH = "files/vocabulary_clean.pkl"
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
            "vocab": self.preprocessor.vocab,  # Store the actual vocabulary dict
            "vocabulary_size": self.preprocessor.vocabulary_size,
            "number_of_classes": len(self.label_encoder.classes_),
            "training_results": self.training_results,
            "model_config": trainer.model_config,  # <-- Required to reinit model
            "max_sent_len": self.preprocessor.max_sent_len,
            "max_sent_num": self.preprocessor.max_sent_num,
            "highest_validation_accuracy": self.training_results["best_val_accuracy"],  # Optional
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