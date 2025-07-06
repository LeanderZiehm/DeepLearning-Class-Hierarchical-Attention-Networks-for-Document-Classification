import os
import pickle
import torch
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from main.pipeline_all import TextPreprocessor, HierarchicalAttentionNetwork
from main.pipeline_all import create_padding_for_hierarchical_sequences
from tqdm import tqdm

# Paths
MODEL_SAVE_PATH = "main/files/han_model2025-07-06_01-13-47.pth"
VOCAB_PATH = "main/files/vocabulary_2025-07-06_01-13-47.pkl"
# TESTSET_PATH = "data/testset.csv"

dataset_name = "validationset"  # Change to "testset" if needed

TESTSET_PATH = f"data/{dataset_name}.csv"



# Load model and vocab
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'), weights_only=False)
model_config = checkpoint['model_config']
model = HierarchicalAttentionNetwork(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)

word_to_idx = vocab_data['word_to_idx']
preprocessor = TextPreprocessor()
preprocessor.word_to_idx = word_to_idx

# Load test set
df = pd.read_csv(TESTSET_PATH)

#remove all rows text or less the n 10 character
df = df[df['text'].str.len() > 10]  

predictions = []
true_labels = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    label = row['label']  # ground truth

    cleaned_text = preprocessor.clean_text(text)
    sentences = sent_tokenize(cleaned_text)[:20]

    hierarchical_doc = []
    for sentence in sentences:
        words = word_tokenize(sentence)[:50]
        word_indices = [word_to_idx.get(word, 1) for word in words]  # 1 = UNK
        hierarchical_doc.append(word_indices)

    padded_docs, word_lengths, sentence_lengths = create_padding_for_hierarchical_sequences([hierarchical_doc])

    with torch.no_grad():
        docs_tensor = torch.tensor(padded_docs, dtype=torch.long)
        word_lengths_tensor = torch.tensor(word_lengths, dtype=torch.long)
        sentence_lengths_tensor = torch.tensor(sentence_lengths, dtype=torch.long)

        logits, _, _ = model(docs_tensor, word_lengths_tensor, sentence_lengths_tensor)
        pred = torch.argmax(logits, dim=1).item()

    predictions.append(pred)
    true_labels.append(label)

# Save results
output_df = pd.DataFrame({
    "text": df['text'],
    "true_label": true_labels,
    "predicted_label": predictions
})



#count ho

output_df.to_csv(f"{dataset_name}_output_predictions.csv", index=False)
print("Inference complete. Results saved to 'output_predictions.csv'.")

# Map true_label from string to int for comparison
reverse_mapping = {'fake': 0, 'true': 1}
output_df['true_label_mapped'] = output_df['true_label'].map(reverse_mapping)

# Get predicted and true labels as integer
preds = output_df['predicted_label']
trues = output_df['true_label_mapped']

# Find rows where prediction was incorrect
incorrect_df = output_df[preds != trues]

# Save incorrect predictions to CSV
incorrect_df.to_csv(f"{dataset_name}_incorrect_predictions.csv", index=False)


