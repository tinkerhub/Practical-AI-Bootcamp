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
- Using tf.data or torch.utils.data

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

