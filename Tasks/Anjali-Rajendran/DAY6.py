import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import BatchNormalization, Dropout
from keras.models import Model

from tensorflow.keras.datasets import cifar10
# setting class names
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#loading the dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

# Normalizing Images
x_train=x_train/255.0
x_train.shape
x_test=x_test/255.0
x_test.shape

# loading resnet file model
resnet = ResNet50(weights ='imagenet', include_top = False, 
               input_shape =(32,32, 3)) 
resnet.trainable = False

x= resnet.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(10, activation ='softmax')(x)

model = Model(resnet.input, x)
model.compile(optimizer ='Adam', 
              loss ="sparse_categorical_crossentropy", 
              metrics =["sparse_categorical_accuracy"]) 
model.summary() 
model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))


# loading vgg16 model
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(32,32, 3))
vgg16.trainable = False

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
model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))


# loading densnet121 model
den121 = DenseNet121(weights ='imagenet', include_top = False, 
               input_shape =(32,32, 3)) 
den121.trainable = False

x = den121.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x) 
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(10, activation ='softmax')(x)

model = Model(den121.input, x)
model.compile(optimizer ='Adam', 
              loss ="sparse_categorical_crossentropy", 
              metrics =["sparse_categorical_accuracy"]) 
model.summary() 
model.fit(x_train,y_train, epochs = 5, validation_data = (x_test,y_test))
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
