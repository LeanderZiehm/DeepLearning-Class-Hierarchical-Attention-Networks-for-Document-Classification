from datasets.fake_news import FakeNewsDataset
from preprocessing.text import TextPreprocessor
from models.han import HierarchicalAttentionNetwork
from training.trainer import ModelTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class Pipeline:
    def __init__(self, data_path, model_save_path, vocab_save_path):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.vocab_save_path = vocab_save_path
        self.preprocessor = None
        self.model = None
        self.training_results = None

    def load_data(self):
        fake = pd.read_csv(os.path.join(self.data_path, "Fake.csv"))
        true = pd.read_csv(os.path.join(self.data_path, "True.csv"))
        fake["label"] = "fake"
        true["label"] = "true"
        data = pd.concat([fake, true], ignore_index=True)
        trainset_data, temp_data = train_test_split(
            data, test_size=0.2, stratify=data["label"], random_state=42
        )
        validationset_data, test_data = train_test_split(
            temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42
        )
        return trainset_data, validationset_data, test_data

    def preprocess_data(self, trainset_data):
        self.preprocessor = TextPreprocessor(maximum_vocabulary_size=50000, minimum_word_frequency=2)
        self.preprocessor.build_vocabulary(trainset_data["text"].tolist())
        self.preprocessor.save_vocabulary(self.vocab_save_path)

    def train_model(self, trainset_loader, validationset_loader):
        trainer = ModelTrainer(
            vocabulary_size=self.preprocessor.vocabulary_size,
            number_of_classes=2,  # Fake and True
            label_encoder=None  # Implement label encoding if needed
        )
        self.training_results = trainer.train(
            trainset_loader=trainset_loader,
            validationset_loader=validationset_loader,
            number_of_epochs=10,
            learning_rate=0.001,
            save_path=self.model_save_path,
            patience=3
        )
        self.model = trainer.model

    def run(self):
        trainset_data, validationset_data, test_data = self.load_data()
        self.preprocess_data(trainset_data)
        # Create datasets and loaders here
        # Call self.train_model with appropriate loaders

if __name__ == "__main__":
    pipeline = Pipeline(data_path='data/', model_save_path='files/best_model.pth', vocab_save_path='files/vocabulary.pkl')
    pipeline.run()