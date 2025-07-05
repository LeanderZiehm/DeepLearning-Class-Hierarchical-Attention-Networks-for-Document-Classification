Starting Fake News Detection Analysis Pipeline
Loading data
Dataset splits:
Train: 35918 samples
Validation: 4490 samples
Test: 4490 samples

Label distribution:
Train set:
label
fake    18785
true    17133
Name: count, dtype: int64
Validation set:
label
fake    2348
true    2142
Name: count, dtype: int64
Test set:
label
fake    2348
true    2142
Name: count, dtype: int64
Building vocabulary
Building vocabulary
Processing texts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 35918/35918 [00:51<00:00, 703.85it/s]
Vocabulary size: 50000
Encoding labels
Vocabulary size: 50000
Label mapping: {'fake': 0, 'true': 1}
Creating datasets and data loaders
Convertng text to hierarchical format: 100%|████████████████████████████████████████████████████████████████████████████████████| 35918/35918 [00:44<00:00, 800.56it/s]
Convertng text to hierarchical format: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4490/4490 [00:05<00:00, 797.81it/s]
Convertng text to hierarchical format: 100%|██████████████████████████████████████████████████████████████████████████████████████| 4490/4490 [00:05<00:00, 807.79it/s]
Training HAN model
Model created with 10,141,802 parameters
Using device: cuda
Training started
Training samples:   35918
Validation samples: 4490
Epoch 1/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2245/2245 [00:30<00:00, 74.33it/s, Loss=0.0009, Acc=1.0000]

Epoch 1/10:
Train Loss: 0.0236, Train Accuracy: 0.9925
Validation Loss: 0.0064, Validation   Accuracy: 0.9987
Learning Rate: 0.001000
New best model saved Validation Accuracy: 0.9987

Epoch 2/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2245/2245 [00:30<00:00, 72.71it/s, Loss=0.0001, Acc=1.0000]

Epoch 2/10:
Train Loss: 0.0067, Train Accuracy: 0.9986
Validation Loss: 0.0049, Validation   Accuracy: 0.9989
Learning Rate: 0.001000
New best model saved Validation Accuracy: 0.9989

Epoch 3/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2245/2245 [00:30<00:00, 72.54it/s, Loss=0.0001, Acc=1.0000]

Epoch 3/10:
Train Loss: 0.0041, Train Accuracy: 0.9987
Validation Loss: 0.0028, Validation   Accuracy: 0.9998
Learning Rate: 0.000700
New best model saved Validation Accuracy: 0.9998

Epoch 4/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2245/2245 [00:30<00:00, 72.57it/s, Loss=0.0003, Acc=1.0000]

Epoch 4/10:
Train Loss: 0.0017, Train Accuracy: 0.9996
Validation Loss: 0.0033, Validation   Accuracy: 0.9996
Learning Rate: 0.000700
No improvement. Patience: 1/3

Epoch 5/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2245/2245 [00:30<00:00, 72.59it/s, Loss=0.0001, Acc=1.0000]

Epoch 5/10:
Train Loss: 0.0018, Train Accuracy: 0.9995
Validation Loss: 0.0030, Validation   Accuracy: 0.9996
Learning Rate: 0.000700
No improvement. Patience: 2/3

Epoch 6/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2245/2245 [00:31<00:00, 72.34it/s, Loss=0.0002, Acc=1.0000]

Epoch 6/10:
Train Loss: 0.0014, Train Accuracy: 0.9996
Validation Loss: 0.0033, Validation   Accuracy: 0.9993
Learning Rate: 0.000490
No improvement. Patience: 3/3
Early stopping triggered after 6 epochs
Training completed with the best validation accuracy being: 0.9998
Model loaded from files/new_best_han_model.pth

RESULTS

None




✅ Final Output Breakdown
Model loaded successfully from:
data/new_best_han_model.pth

Validation accuracy (from training):
0.9997775800711743 → nearly perfect accuracy on the validation set

Model Configuration (used to reconstruct the model):

python
Copy
Edit
{
    'vocabulary_size': 50000, 
    'embedding_dimmentions': 200, 
    'word_gru_hidden_units': 50, 
    'word_gru_layers': 1, 
    'word_attention_dimmentions': 100, 
    'sentence_gru_hidden_units': 50, 
    'sentence_gru_layers': 1, 
    'sentence_attention_dimmention': 100, 
    'number_of_classes': 2
}
Model's state dict (weights): First 10 keys are:

bash
Copy
Edit
[
 'word_attention.embedding.weight',
 'word_attention.word_gru.weight_ih_l0',
 'word_attention.word_gru.weight_hh_l0',
 'word_attention.word_gru.bias_ih_l0',
 'word_attention.word_gru.bias_hh_l0',
 'word_attention.word_gru.weight_ih_l0_reverse',
 'word_attention.word_gru.weight_hh_l0_reverse',
 'word_attention.word_gru.bias_ih_l0_reverse',
 'word_attention.word_gru.bias_hh_l0_reverse',
 'word_attention.word_attention.weight'
]
Checkpoint keys:

python
Copy
Edit
[
 'model_state_dict',
 'model_config',
 'highest_validation_accuracy',
 'epoch',
 'optimizer_state_dict',
 'scheduler_state_dict',
 'label_encoder',
 'vocabulary_size',
 'number_of_classes',
 'training_results'
]