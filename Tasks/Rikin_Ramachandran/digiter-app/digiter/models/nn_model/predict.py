import tensorflow as tf
import numpy as np
import os
MODEL_PATH = os.environ.get("MODEL_PATH")
model = tf.keras.models.load_model(MODEL_PATH)

def predict(image, model=model):
    img = tf.image.resize(image, (28, 28))
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    score = tf.nn.softmax(pred[0])
    return np.argmax(score)

