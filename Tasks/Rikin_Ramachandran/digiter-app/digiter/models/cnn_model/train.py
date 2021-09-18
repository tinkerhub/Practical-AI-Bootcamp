import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train_o, y_train), (x_test_o, y_test) = mnist.load_data()

x_train=np.reshape(x_train_o,(60000,28,28,1))
y_train=tf.keras.utils.to_categorical(y_train)
x_test=np.reshape(x_test_o,(10000,28,28,1))
y_test=tf.keras.utils.to_categorical(y_test)

model= tf.keras.models.Sequential()
l1= tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1))
l2= tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')
l3= tf.keras.layers.Flatten()
l4= tf.keras.layers.Dense(10, activation='softmax')

model.add(l1)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(l2)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(l3)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(l4)

model.compile(optimizer ='Adam', 
              loss ='categorical_crossentropy',
              metrics =['accuracy']) 


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=512)

model.save('saved_model/model')
