import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('saved_model/my_model')

def predict(image, model=model):
    img = tf.image.resize(image, (28, 28))
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    score = tf.nn.softmax(pred[0])
    return np.argmax(score)

