
from main.src.data_preprocessor import TextPreprocessor, FakeNewsDataset, collate_fn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# import nltk
# nltk.download('punkt_tab')

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
    
def create_datasets_and_loaders():
    print("Creating datasets and data loaders")
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
    trainset_data, validationset_data, test_data = trainset_data, validationset_data, test_data
    # _print_data_info()
    # return trainset_data, validationset_data, test_data

    print("Building vocabulary")
    preprocessor = TextPreprocessor(
        maximum_vocabulary_size=MAX_VOCAB_SIZE, minimum_word_frequency=MIN_WORD_FREQ
    )
    preprocessor.build_vocabulary(trainset_data["text"].tolist())
    preprocessor.save_vocabulary(VOCAB_PATH)
    print("Encoding labels")
    global label_encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(trainset_data["label"])

    print(f"Vocabulary size: {preprocessor.vocabulary_size}")
    print(
        f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
    )
    # if data_splits is None:
        # data_splits = (trainset_data, validationset_data, test_data)
    # trainset_data, validationset_data, test_data = data_splits
    # Create dataset objects
    datasets = []
    for data in [trainset_data, validationset_data, test_data]:
        texts = data["text"].tolist()
        labels = label_encoder.transform(data["label"])
        dataset = FakeNewsDataset(
            texts, labels, preprocessor, MAX_SENTENCES, MAX_WORDS_PER_SENTENCE
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
    
    trainset_path = os.path.join(DATA_PATH, "trainset.csv")
    validationset_path = os.path.join(DATA_PATH, "validationset.csv")
    testset_path = os.path.join(DATA_PATH, "testset.csv")
    print(f"Trainset saved to {trainset_path}")
    print(f"Validationset saved to {validationset_path}")
    print(f"Testset saved to {testset_path}")
    
    # save them as seperate csvs 
    trainset_data.to_csv(os.path.join(DATA_PATH, "trainset.csv"), index=False)
    validationset_data.to_csv(os.path.join(DATA_PATH, "validationset.csv"), index=False)
    test_data.to_csv(os.path.join(DATA_PATH, "testset.csv"), index=False)
    # print
    
    # return datasets, (trainset_loader, validationset_loader, test_loader)

create_datasets_and_loaders()

