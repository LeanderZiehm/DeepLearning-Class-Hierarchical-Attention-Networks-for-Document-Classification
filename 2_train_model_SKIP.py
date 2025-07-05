import torch
from main.src.model_trainer import ModelTrainer
import pandas as pd
import os

import pickle
from sklearn.preprocessing import LabelEncoder
from main.src.data_preprocessor import TextPreprocessor  # adjust import path

def get_preprocessor(trainset_data):
    print("Building vocabulary")
    preprocessor = TextPreprocessor(
        maximum_vocabulary_size=MAX_VOCAB_SIZE,
        minimum_word_frequency=MIN_WORD_FREQ
    )
    preprocessor.build_vocabulary(trainset_data["text"].tolist())
    preprocessor.save_vocabulary(VOCAB_PATH)  # Optional if you want to save it

    print("Encoding labels")
    label_encoder = LabelEncoder()
    label_encoder.fit(trainset_data["label"])

    # Optionally save the label encoder for reuse
    with open("data/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Vocabulary size: {preprocessor.vocabulary_size}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    return preprocessor, label_encoder



DATA_PATH = "data/"
MODEL_SAVE_PATH = "data/new_best_han_model.pth"
VOCAB_PATH = "data/new_vocabulary2.pkl"
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

trainset = pd.read_csv(os.path.join(DATA_PATH, "trainset.csv"))


# def get_preprocessor(trainset_data):
#     print("Building vocabulary")
#     preprocessor = TextPreprocessor(
#         maximum_vocabulary_size=MAX_VOCAB_SIZE, minimum_word_frequency=MIN_WORD_FREQ
#     )
#     preprocessor.build_vocabulary(trainset_data["text"].tolist())
#     preprocessor.save_vocabulary(VOCAB_PATH)
#     print("Encoding labels")
#     label_encoder = LabelEncoder()
#     label_encoder.fit(trainset_data["label"])

#     print(f"Vocabulary size: {preprocessor.vocabulary_size}")
#     print(
#         f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
#     )
#     return preprocessor


def trainset_model(trainset_loader, validationset_loader):
    
    vocabulary_size = ?
    
    preprocessor = get_preprocessor()
    
    print("Training HAN model")
    trainer = ModelTrainer(
        vocabulary_size=vocabulary_size,
        number_of_classes=len(label_encoder.classes_),
        label_encoder=label_encoder,
    )
    training_results = trainer.train(
        trainset_loader=trainset_loader,
        validationset_loader=validationset_loader,
        number_of_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_path=MODEL_SAVE_PATH,
        patience=PATIENCE,
    )
    # Save additional metadata
    additional_data = {
        "label_encoder": label_encoder,
        "vocabulary_size": preprocessor.vocabulary_size,
        "number_of_classes": len(label_encoder.classes_),
        "training_results": training_results,
    }
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    checkpoint.update(additional_data)
    torch.save(checkpoint, MODEL_SAVE_PATH)
    model = trainer.model
    return trainer


   