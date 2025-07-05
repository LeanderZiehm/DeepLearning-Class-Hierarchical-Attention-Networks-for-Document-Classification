import torch
import torch.nn as nn
import random

# Sample data (e.g., Russian vs English names)
data = [
    ("Ivanov", 0),
    ("Petrov", 0),
    ("Sidorov", 0),
    ("Smith", 1),
    ("Johnson", 1),
    ("Williams", 1),
]

# Simple char‑to‑index map
chars = list(set("".join(name for name, _ in data)))
stoi = {ch: i+1 for i, ch in enumerate(chars)}  # leave 0 for padding
print("Character to index mapping:", stoi)


vocab_size = len(stoi) + 1
print("Vocabulary size:", vocab_size)

def encode(name):
    return torch.tensor([stoi[ch] for ch in name], dtype=torch.long)

# GRU model
class NameGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=16, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)              # (batch, seq_len, embed_dim)
        out, h = self.gru(x)              # h: (1, batch, hidden_dim)
        return self.fc(h.squeeze(0))     # (batch, n_classes)

# Simple training loop
model = NameGRU(vocab_size)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    name, label = random.choice(data)
    x = encode(name).unsqueeze(0)    # add batch dim
    y = torch.tensor([label])

    logits = model(x)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Test inference
for name, _ in data:
    x = encode(name).unsqueeze(0)
    preds = model(x).argmax(dim=1).item()
    print(name, "→", "Russian" if preds==0 else "English")
