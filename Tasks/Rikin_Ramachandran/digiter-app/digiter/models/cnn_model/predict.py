model = tf.keras.models.load_model('saved_model/cnn_model.h5')

def predict(image, model=model):
    img = tf.keras.preprocessing.image.load_img(
        image, target_size=(28, 28), color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    score = tf.nn.softmax(pred[0])
    return np.argmax(score)

