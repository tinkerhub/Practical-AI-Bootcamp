from keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras import datasets,layers,models
from keras.models import Model

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input

# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input




from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


(trainX, trainY),(testX,testY)=datasets.cifar10.load_data()
vgg = VGG16(input_shape=[32,32,3], weights='imagenet', include_top=False)
# resnet50 = ResNet50(input_shape=[32,32,3], weights='imagenet', include_top=False)
# inceptionv3= InceptionV3(input_shape=[32,32,3], weights='imagenet', include_top=False)

for layers in vgg.layers:
  layers.trainable=False

x=Flatten()(vgg.output)
prediction=Dense(10,activation='softmax')(x)

model=Model(inputs=vgg.input,outputs=prediction)
print(model.summary())

model.compile(loss='categorical-crossentropy',optimizer='adam',metrics=['accuracy'])

trainX=trainX/255
testX=testX/255
trainY_ohe= to_categorical(trainY)
testY_ohe= to_categorical(testY)

r=model.fit(np.array(trainX),np.array(trainY_ohe),batch_size=32,epochs=5)

# dataAugmentaion = ImageDataGenerator(rotation_range = 30, zoom_range = 0.20, 
# fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True, 
# width_shift_range = 0.1, height_shift_range = 0.1)

# # training the model
# model.fit_generator(dataAugmentaion.flow(trainX, trainY, batch_size = 32),
#  validation_data = (testX, testY),
#  epochs = 5)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
