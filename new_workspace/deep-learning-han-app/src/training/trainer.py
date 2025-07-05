from models.han import HierarchicalAttentionNetwork
from datasets.fake_news import FakeNewsDataset
from torch.utils.data import DataLoader
import torch

class ModelTrainer:
    """Complete training pipeline for any model."""

    def __init__(self, model_class, model_config, device=None):
        self.model_class = model_class
        self.model_config = model_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_class(**self.model_config).to(self.device)

    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001, save_path=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                documents, word_lengths, sentence_lengths, labels = batch
                documents, word_lengths, sentence_lengths, labels = (
                    documents.to(self.device),
                    word_lengths.to(self.device),
                    sentence_lengths.to(self.device),
                    labels.to(self.device),
                )

                optimizer.zero_grad()
                logits, _, _ = self.model(documents, word_lengths, sentence_lengths)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            self.evaluate(val_loader)

            if save_path:
                torch.save(self.model.state_dict(), save_path)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                documents, word_lengths, sentence_lengths, labels = batch
                documents, word_lengths, sentence_lengths, labels = (
                    documents.to(self.device),
                    word_lengths.to(self.device),
                    sentence_lengths.to(self.device),
                    labels.to(self.device),
                )

                logits, _, _ = self.model(documents, word_lengths, sentence_lengths)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                total_accuracy += (predictions == labels).float().mean().item()

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")

    def create_data_loader(self, dataset, batch_size=16, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)