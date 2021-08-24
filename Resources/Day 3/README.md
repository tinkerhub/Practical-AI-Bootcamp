# Dataset and performance
- Data Preparation
- Data Reading 
- Data augumentation 

# Data Preparation
- Using datasets coming with the framework
- Using public and other datasets

## Datasets coming with the framework
1. Downloading datasets are quiet an effort
2. Even after downloading its painful to find and use it's structure and formating 
3. One possibilities is to explore the datasets coming with the framework

### TensorFlow example

```python
import tensorflow_datasets as tfds
print(tfds.list_builders())

dataloader = tfds.load("cifar10", as_supervised=True)
train, test = dataloader["train"], dataloader["test"]

```

### Pytorch example
[Datasets available in torchvision](https://pytorch.org/vision/stable/datasets.html)

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

```

## Public datasets or datasets from work
- Use packages like pandas
- Use tf.data or torch.utils.data

### Use packages like pandas

```python
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df.head()

"""
for index, row in df.iterrows():
    print(row[0], row[4])
"""    

train, test = train_test_split(df_text_genre, test_size=0.2, random_state=42, shuffle=True)
```
- [You can load pandas dataframe to tensorflow](https://www.tensorflow.org/tutorials/load_data/pandas_dataframe)

## Data Reading
tf.data and torch.utils.data are APIs that enables you to build input pipelines for your machine learning models

### tf.data

```python
import tensorflow as tf

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)
for line in dataset.take(5):
  print(line.numpy())
 
```
- [Find more on tf.data here](https://www.tensorflow.org/guide/data)

### torch.utils.data

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

- [Find more on torch.utils.data here](https://pytorch.org/docs/stable/data.html)
```

## Data Augumentation
You can apply data augumentation techniques on the input pipelines

### Tensorflow example

```python

import tensorflow_datasets as tfds

dataloader = tfds.load("cifar10", as_supervised=True)
train, test = dataloader["train"], dataloader["test"]

train = train.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
).cache().map(
    lambda image, label: (tf.image.random_flip_left_right(image), label)
).map(
    lambda image, label: (tf.image.random_contrast(image, lower=0.0, upper=1.0), label)
).shuffle(
    100
).batch(
    64
).repeat()
```


