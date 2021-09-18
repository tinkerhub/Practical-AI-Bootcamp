import tensorflow as tf

from tensorflow.keras import models
import cv2

import numpy as np

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image_path) -> str:
    """ predict image class to cifar10 classes 

        input -> image path
        output -> class string
    """

    img = cv2.imread(image_path)
    img = cv2.resize(img,(32,32))
    img = img/255.0
    img = tf.expand_dims(img, axis=0)

    model = models.load_model('cifar10.h5')
    score = model.predict(img)
    prediction = np.argmax(score)

    prediction = class_names[prediction]
    return prediction

if __name__ == "__main__":
    img = "./static/uploads/upload.png"

    print(predict_image(img))