# Techniques in training AI models

1. Finding the right learning rate
2. Effects of batch size
3. Epochs and early stop

# Finding the right learning rate

- Ideal result of training a ML model is reaching global minimum
- Too large learning rate will overshoot the learning
- Too low learning rate will make the training so slow
- One not good and time consuming option is to try various learning rates like 0.00001, 0.0001, 0.001 ...etc
- A better option is to use cyclical learning rate strategy 

## Cyclical learning rate strategy

1. Start with a low learning rate and gradually increase it until reaching a prespecified max value.
2. At each lr obeserve the loss. At first it will be stagnent then at some point it drop and then eventually go back up
3. Calculate the rate of loss decrease at each learning rate
4. Select the point with the highest rate of decrease.

[Here is an example of how to do this](https://github.com/beringresearch/lrfinder/tree/master/examples)

Import packages

```python
import tensorflow as tf
from tensorflow_datasets import tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lrfinder import LRFinder
```

Loading dataset

```python
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
  training_sentences.append(str(s.tonumpy()))
  training_labels.append(l.tonumpy())
  
for s,l in test_data:
  testing_sentences.append(str(s.tonumpy()))
  testing_labels.append(l.tonumpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
```

Vocabulary and Tokenization

```python
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = ""

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) 
tokenizer.fit_on_texts(training_sentences) 
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences) 
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
```

Create the model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```

Find learning rate

```python
BATCH = 512
train_ds = tf.data.Dataset.from_tensor_slices((padded, training_labels_final))
train_ds = train_ds.batch(BATCH)
STEPS_PER_EPOCH = np.ceil(len(train_data) / BATCH)
lr_finder = LRFinder(model)
lr_finder.find(train_ds, start_lr=1e-6, end_lr=1, epochs=5,
               steps_per_epoch=STEPS_PER_EPOCH)
               
learning_rates = lr_finder.get_learning_rates()
losses = lr_finder.get_losses()

def plot_loss(learning_rates, losses, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
    f, ax = plt.subplots()
    ax.set_ylabel("loss")
    ax.set_xlabel("learning rate (log scale)")
    ax.plot(learning_rates[n_skip_beginning:-n_skip_end],
            losses[n_skip_beginning:-n_skip_end])
    ax.set_xscale(x_scale)
    return(ax)

axs = plot_loss()
axs.axvline(x=lr_finder.get_best_lr(sma=20), c='r', linestyle='-.')
```

## The plot looks like this

![learning rate](https://github.com/tinkerhub/Practical-AI-Bootcamp/blob/main/Resources/Day%204/Screenshot%202021-08-28%20at%209.04.15%20PM.png)


Lets get the best learning rate and set it as model learning rate

```python
best_lr = lr_finder.get_best_lr(sma=20)
K.set_value(model.optimizer.lr, best_lr)
print(model.optmizer.lr)
```


# Effect of batch size
1. Too large of a batch size will lead to poor generalization
2. Smaller batch size allows model to start learning before having to see the entire data

# Epochs and early stop 

- During training, weights in the neural networks are updated so that the model performs better on the training data.
- For a while, improvements on the training set correlate positively with improvements on the test set.
- However, there comes a point where you begin to overfit on the training data and further "improvements" will result in lower generalization performance.
- Early stopping is a technique used to terminate the training before overfitting occurs.

```python
from tensorflow.keras.callbacks import EarlyStopping

earlystop_callback = EarlyStopping(
  monitor='val_accuracy', min_delta=0.0001, patiece=1)
```

When you do the `model.fit()` pass `earlystop_callback` as param 

```python
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final), callbacks=[earlystop_callback])
model.evaluate(testing_padded)
```

