# Finding the right machine learning model for a problem

- Should I use machine learning for this problem ?
- What kind of ML task is this ?
- Machine learning or deep learning ?
- What model(s) ?

# Should I use machine learning model for this problem ?
1. We must find a useful pattern
2. The pattern must generalize(handle new examples)
3. The pattern should be stationary
4. But do we have sufficent data ?

# What kind of machine learning task is this ?

## Supervised Learning 
Dataset is manually laballed.
### Classification 
prediction of a label
- cat-dog image classifier
### Regression
prediction of a quantity
- Prediction of stock market

## Un-supervised Learning 
Dataset is not labelled. Used where labelled data is elusive, or too expensive to get
### Clustering 
You don't know the labels. 
- Find different user groups using your website
### Anomaly detection 
Find unusual behaviour
- Credit card theft
### Association rules 
How various items are grouped in a supermarket
### Autoencoders
Learn input data and try to re-create or create different version of the input data
- Generate clear images from noisy ones

## Semi-supervised Learning
Training dataset with both labelled and unlabelled data
- Medical applicatiions, where a small amount of labelledata can make significant improvement in accuracy
- This training method can be done using [GAN](https://blogs.nvidia.com/blog/2017/05/17/generative-adversarial-networks/)

## Reinforcement learning
Learns from situations by reinforcing favourable situations.
### Self driving cars
- Lane following
- parking
- various policy based algorithms
### Automation and robots 
- [Rubiks cube solving robot](https://openai.com/blog/solving-rubiks-cube/)
- [cool data centres](https://deepmind.com/blog/article/safety-first-ai-autonomous-data-centre-cooling-and-industrial-control)

# Machine learning or deep learning
A rule of thumb I use is to try basic machine learning algorithms before jumping to deep learning
## Machine learning
- Machine learning need only limited but structured data
- Mostly used for tabular data
- Non-tabular data need feature engineering
## Deep learning
- Deep learning need more data
- Can accomodate unstructured data
- Need more computation power
- Deployment might also need GPU systems

