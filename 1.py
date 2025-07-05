import torch, torch.nn as nn

class WordEmbedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, x):
        # x: (batch, max_words)
        return self.embedding(x)  # → (batch, max_words, emb_dim)

# test
batch = torch.randint(1,100, (4,10))
we = WordEmbedder(100, 16)
print(we(batch).shape)  # (4,10,16)
