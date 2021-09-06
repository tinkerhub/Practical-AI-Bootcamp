# Transfer Learning Where When and How to use

1. What is Transfer Learning
2. What we can do
3. When we can do
4. Where we use it
5. Advantages & Benefits
6. How to use
7. Best practices

## What is Transfer Learning

Storing the knowledge gained while solving one problem and applying it to a different but related problem.

## What we can do

- Take the last output layer of the Neural Network.
- just delete that and also the weight feeding into that.
- create a new set of randomly intialised weights just for the last layer

_Thumb Rule: just Re-train the last layer , ie output layer (one/two)_

## When we can do

- Less data
- Trying to learn from a task and transfer some knowledge to another task
- same input
- More data for the first task then the second


## Where we can use

- Computer vision and Natural language processing tasks like _sentiment analysis_
- Image Recognition
- Speech recognition

 _Industries:_
- Autonomous driving
- Gaming
- Health care
- Spam filtering

## Advantages and Benefits

- Better initial model
- Higher Learning rate
- Higher accuracy after training
- Faster Training
- Low level features from first task are helpful for learning second task 
- No need to build model from skratch

## How to use


_Model building_
~~~python
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import BatchNormalization, Dropout
from keras.models import Model

res = ResNet50(weights ='imagenet', include_top = False, 
               input_shape =(256, 256, 3)) 

~~~
_we can add the rest of the classifier. This takes the output from the pre-trained convolutional layers and inputs it into a separate classifier that gets trained on the new dataset._

~~~python

x = res.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(101, activation ='softmax')(x)

model = Model(res.input, x)

model.compile(optimizer ='Adam', 
              loss ='categorical_crossentropy', 
              metrics =['accuracy']) 
 ~~~
 
_Structure of our model_
~~~python
model.summary() 
~~~

_Train the model_

~~~python
model.fit_generator(train_flow, epochs = 5, validation_data = valid_flow)
~~~

## Best Practices

- Take Advantage of Pre-trained models
- compatible
- Large Dataset doesnot need TL
- overfitting


