#!/usr/bin/env python
# coding: utf-8

# In[9]:


from tensorflow.keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import BatchNormalization, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import VGG16


# In[2]:


res = ResNet50(weights ='imagenet', include_top = False, 
               input_shape =(32,32, 3)) 
res.trainable = False


# In[7]:


class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#loading the dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train/255.0
x_test=x_test/255.0


# In[5]:


x= res.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(10, activation ='softmax')(x)

model = Model(res.input, x)
model.compile(optimizer ='Adam', 
              loss ="sparse_categorical_crossentropy", 
              metrics =["sparse_categorical_accuracy"]) 
model.summary() 


# In[6]:


model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))


# In[7]:


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))


# In[3]:


Xception = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
Xception.trainable = False


# In[4]:


x= Xception.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(10, activation ='softmax')(x)

model = Model(Xception.input, x)
model.compile(optimizer ='Adam', 
              loss ="sparse_categorical_crossentropy", 
              metrics =["sparse_categorical_accuracy"]) 
model.summary() 


# In[5]:


model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))


# In[6]:


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))


# In[10]:


vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(32,32, 3))
vgg16.trainable = False


# In[11]:


x = vgg16.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(10, activation ='softmax')(x)

model = Model(vgg16.input, x)
model.compile(optimizer ='Adam', 
              loss ="sparse_categorical_crossentropy", 
              metrics =["sparse_categorical_accuracy"]) 
model.summary()


# In[12]:


model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))


# In[13]:


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))


# In[ ]:




