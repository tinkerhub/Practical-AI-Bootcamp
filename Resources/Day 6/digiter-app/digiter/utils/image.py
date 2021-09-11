import tensorflow as tf

def decode_image(base_64_image):
    """Decode image from base64 to numpy array"""
    image_bytes = tf.io.decode_base64(base_64_image)
    image_tensor = tf.image.decode_image(image_bytes)
    return image_tensor
