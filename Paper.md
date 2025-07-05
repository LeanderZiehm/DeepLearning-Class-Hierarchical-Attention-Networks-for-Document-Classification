# Not in paper:
Optimizaer Adam
Dropout 0.1

# Loss:
```
We use the negative log likelihood of the correct labels as training loss:
L = −
X
d
log pdj , (12)
where j is the label of document d.
```

# GRU
We use a bidirectional GRU (Bahdanau et al., 2014)


# Testing
The experimental results on all data sets are shown
in Table 2. We refer to our models as HN-{AVE,
MAX, ATT}. Here HN stands for Hierarchical
Network, AVE indicates averaging, MAX indicates
max-pooling, and ATT indicates our proposed hierarchical attention model. Results show that HNATT gives the best performance across all data sets.
