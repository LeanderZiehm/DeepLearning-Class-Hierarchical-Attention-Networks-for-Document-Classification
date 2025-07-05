What does padding do? Why Padding? padding_idx nn.Embedding
loss.backward()
loss.forward()

nn.init.xavier_uniform_

clip_grad_norm_

torch.optim.Adam

torch.optim.lr_scheduler.StepLR

https://chatgpt.com/c/686803d5-8da8-8005-83d8-a97b6718a2fb


































Words makes up Sentences and Sentences make Documents. 



GRU faster and simpler 2 layers 


Document Embedding. 






Imports
import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PyTorch Layers and Modules
nn.Module (Base class for all models)

nn.Embedding

nn.GRU (Gated Recurrent Unit)

nn.Linear

nn.Dropout

nn.CrossEntropyLoss

Functional Operations (from torch.nn.functional)
F.softmax

F.tanh

Sequence Utilities (from torch.nn.utils.rnn)
pack_padded_sequence

pad_packed_sequence

Tensor Operations
torch.arange

torch.sum

torch.zeros

torch.argmax

tensor.view(...)

tensor.size()

tensor.squeeze(...)

tensor.unsqueeze(...)

tensor.masked_fill(...)

tensor.to(device)

tensor.copy_()

tensor.dim()

.requires_grad

.cpu()

Training Utilities
optimizer.zero_grad()

loss.backward()

optimizer.step()

torch.nn.utils.clip_grad_norm_()

model.train(), model.eval()

with torch.no_grad(): (inference context)

torch.save(...)

Random Data Utilities
torch.randint(...)

NumPy Functions
np.full(...)

np.zeros(...)

np.int64

Model Weight Initialization
nn.init.xavier_uniform_

nn.init.zeros_

Optimizers and Schedulers
torch.optim.Adam

torch.optim.lr_scheduler.StepLR

