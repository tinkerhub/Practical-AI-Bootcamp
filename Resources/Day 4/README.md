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
3. Calculate the rate if decrease at each learning rate
4. Select the point with the highest rate of decrease.

[Here is an example of how to do this](https://github.com/surmenok/keras_lr_finder/blob/master/examples/Example.ipynb)

Import packages

```python
import tensorflow as tf
from tensorflow_datasets import tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
model.summary()

```
Find learning rate

```python
lr_finder = LRFinder(model)
lr_finder.find(x_train, y_train, 0.0001, 1, 512, 5)
lr_finder.plot_loss()
```

