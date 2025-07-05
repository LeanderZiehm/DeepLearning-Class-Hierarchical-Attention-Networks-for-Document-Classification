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

# Parameters and Vocabulary
3.3 Model configuration and training
We split documents into sentences and tokenize each
sentence using Stanford’s CoreNLP (Manning et al.,
2014). We only retain words appearing more than
5 times in building the vocabulary and replace the
words that appear 5 times with a special UNK token.
We obtain the word embedding by training an unsupervised word2vec (Mikolov et al., 2013) model
on the training and validation splits and then use the
word embedding to initialize We.
The hyper parameters of the models are tuned
on the validation set. In our experiments, we set
the word embedding dimension to be 200 and the
GRU dimension to be 50. In this case a combination of forward and backward GRU gives us
100 dimensions for word/sentence annotation. The
word/sentence context vectors also have a dimension
of 100, initialized at random.
For training, we use a mini-batch size of 64 and
documents of similar length (in terms of the number
of sentences in the documents) are organized to be a
batch. We find that length-adjustment can accelerate
training by three times. We use stochastic gradient
descent to train all models with momentum of 0.9.
We pick the best learning rate using grid search on
the validation set.



# Testing
The experimental results on all data sets are shown
in Table 2. We refer to our models as HN-{AVE,
MAX, ATT}. Here HN stands for Hierarchical
Network, AVE indicates averaging, MAX indicates
max-pooling, and ATT indicates our proposed hierarchical attention model. Results show that HNATT gives the best performance across all data sets.


