# On device machine learning models

## Tensorflow lite

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

- Read [this](https://www.tensorflow.org/lite/guide/inference) how to make inferences using tensorflow lite model
- [TensorflowLite ios example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)


