import tensorflow as tf
import tensorflow_text as text


directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)
dataset_preprocessed = map(lambda str: tf_text.normalize_utf8(str), dataset)

for i, data in enumerate(dataset):
    if i < 10:
        print(data.numpy())
    else:
        break
