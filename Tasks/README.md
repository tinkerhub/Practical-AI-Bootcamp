# Tasks for the bootcamp

## Day 3

You learnt [data augumentation in tf.data](https://github.com/tinkerhub/Practical-AI-Bootcamp/blob/main/Resources/Day%203/README.md#tfdata) and data augumentation on input pipelines. Click [here](https://www.tensorflow.org/text) to find a tool for applying preprocessing to text data. Apply `text.normalize_utf8` to the below loaded dataset

```python
import tensorflow as tf

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)
```

## Day 5

Apply `Earlystopping` for the model we trained in last session with monitor validation loss and patience 3

## Day 6

Do an Analysis by using different Pre-Trained models, (_atleast 3_), to solve any classification based machine learning problem.

## Day 7

Add a `cnn_model` to the web app project - you can add a train.py and predict.py inside your `cnn_model` folder

You can use this [resource](https://www.tensorflow.org/tutorials/images/cnn)

