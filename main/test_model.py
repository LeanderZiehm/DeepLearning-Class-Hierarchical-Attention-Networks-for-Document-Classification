import os
import pickle
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
from pipeline_all import TextPreprocessor, HierarchicalAttentionNetwork
from pipeline_all import create_padding_for_hierarchical_sequences


MODEL_SAVE_PATH = "files/han_model2025-07-06_01-13-47.pth"  # Update with the correct model path
VOCAB_PATH = "files/vocabulary_2025-07-06_01-13-47.pkl"  # Update with the correct vocabulary path

def load_model_and_vocab():
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'), weights_only=False)
    model_config = checkpoint['model_config']
    
    model = HierarchicalAttentionNetwork(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)
    
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = vocab_data['idx_to_word']
    
    return model, word_to_idx

def preprocess_input(text, preprocessor, max_sentences=20, max_words_per_sentence=50):
    cleaned_text = preprocessor.clean_text(text)
    sentences = sent_tokenize(cleaned_text)
    sentences = sentences[:max_sentences]
    
    hierarchical_doc = []
    for sentence in sentences:
        words = word_tokenize(sentence)[:max_words_per_sentence]
        word_indices = [preprocessor.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        if word_indices:
            hierarchical_doc.append(word_indices)
    
    return hierarchical_doc if hierarchical_doc else [[1]]  # Return <UNK> if empty

def main():
    model, word_to_idx = load_model_and_vocab()
    preprocessor = TextPreprocessor()
    preprocessor.word_to_idx = word_to_idx  # Set the loaded vocabulary

    while True:
        input_text = input("Enter a news article (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
        
        hierarchical_doc = preprocess_input(input_text, preprocessor)
        padded_docs, word_lengths, sentence_lengths = create_padding_for_hierarchical_sequences([hierarchical_doc])
        
        with torch.no_grad():
            docs_tensor = torch.tensor(padded_docs, dtype=torch.long)
            word_lengths_tensor = torch.tensor(word_lengths, dtype=torch.long)
            sentence_lengths_tensor = torch.tensor(sentence_lengths, dtype=torch.long)
            logits, _, _ = model(docs_tensor, word_lengths_tensor, sentence_lengths_tensor)
        
        predictions = torch.argmax(logits, dim=1)
        print(f"Predicted label: {predictions.item()}")

if __name__ == "__main__":
    main()