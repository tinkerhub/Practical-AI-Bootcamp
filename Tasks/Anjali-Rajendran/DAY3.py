# DAY 3
import tensorflow as tf
import tensorflow_text as tf_text

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)

data115 = dataset.map(lambda y : tf_text.normalize_utf8(y)).shuffle(300)
for line in data115.take(10):
  print(line.numpy())
