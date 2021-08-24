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

