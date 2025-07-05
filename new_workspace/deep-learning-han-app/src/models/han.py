from torch import nn
from .base import BaseModel

class WordAttention(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_units, num_layers, attention_dim):
        super(WordAttention, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_units, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_units, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sentences, word_lengths):
        embedded = self.embedding(sentences)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, word_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_embedded)
        word_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        attention_weights = self.attention(word_outputs)
        attention_weights = self.context_vector(attention_weights).squeeze(-1)
        attention_weights = nn.functional.softmax(attention_weights, dim=1)

        sentence_vectors = (word_outputs * attention_weights.unsqueeze(-1)).sum(dim=1)
        return sentence_vectors, attention_weights

class SentenceAttention(nn.Module):
    def __init__(self, hidden_units, num_layers, attention_dim, num_classes):
        super(SentenceAttention, self).__init__()
        self.gru = nn.GRU(2 * hidden_units, hidden_units, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_units, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Linear(2 * hidden_units, num_classes)

    def forward(self, sentence_vectors, sentence_lengths):
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sentence_vectors, sentence_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_sentences)
        sentence_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        attention_weights = self.attention(sentence_outputs)
        attention_weights = self.context_vector(attention_weights).squeeze(-1)
        attention_weights = nn.functional.softmax(attention_weights, dim=1)

        doc_vectors = (sentence_outputs * attention_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(doc_vectors)
        return logits, attention_weights

class HierarchicalAttentionNetwork(BaseModel):
    def __init__(self, vocabulary_size, embedding_dim, word_hidden_units, word_layers, word_attention_dim, sentence_hidden_units, sentence_layers, sentence_attention_dim, num_classes):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.word_attention = WordAttention(vocabulary_size, embedding_dim, word_hidden_units, word_layers, word_attention_dim)
        self.sentence_attention = SentenceAttention(word_hidden_units, sentence_layers, sentence_attention_dim, num_classes)

    def forward(self, documents, word_lengths, sentence_lengths):
        sentence_vectors, word_attention_weights = self.word_attention(documents, word_lengths)
        logits, sentence_attention_weights = self.sentence_attention(sentence_vectors, sentence_lengths)
        return logits, word_attention_weights, sentence_attention_weights