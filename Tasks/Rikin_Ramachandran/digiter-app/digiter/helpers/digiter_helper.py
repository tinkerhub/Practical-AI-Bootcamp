from digiter.models.cnn_model.predict import predict
from digiter.utils.image import decode_image

def get_digit(image):
    """
    Get digit from image.
    """
    image = decode_image(image)
    digit = predict(image)
    return {
        "digit": digit
    }
