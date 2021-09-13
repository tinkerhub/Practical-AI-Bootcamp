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
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import numpy as np
from lrfinder import LRFinder

```

Loading dataset

```python
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```

Apply preprocessing

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
```

Create the model

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
```

Find learning rate

```python
BATCH = 128

lr_finder = LRFinder(model)
STEPS_PER_EPOCH = np.ceil(len(ds_train) / BATCH)
lr_finder.find(ds_train, start_lr=1e-6, end_lr=1, epochs=5,
               steps_per_epoch=STEPS_PER_EPOCH)
learning_rates = lr_finder.get_learning_rates()
losses = lr_finder.get_losses()
```

Plotting best LR 
```python
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

![learning rate](https://github.com/tinkerhub/Practical-AI-Bootcamp/blob/main/Resources/Day%204/lr_finder.png)


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
  monitor='val_accuracy', min_delta=0, patiece=5)
```

When you do the `model.fit()` pass `earlystop_callback` as param 

```python
model.fit(ds_train, epochs=num_epochs, validation_data=ds_test, callbacks=[earlystop_callback])
model.evaluate(ds_test)
```

What values of `min_delta` and  `patience` to use ?

- If dataset doesn't contain large variations use larger patience
- If on CPU use small patience, on GPU use large patience
- For models like GAN it might be better to use small patience and save model checkpoints
- Set min delta based on running few epochs and checking validation loss

# Save and load models

Use `model.save` for saving a full model

```python
model.save('saved_model/model')
```

Method for loading model and doing prediction

```python
model = tf.keras.models.load_model('saved_model/model')

def predict(image, model=model):
    img = tf.keras.preprocessing.image.load_img(
        image, target_size=(28, 28), color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    score = tf.nn.softmax(pred[0])
    return np.argmax(score)

```
