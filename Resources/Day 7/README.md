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

## Pytorch quantization

```python
import torch

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

# create a model instance
model_fp32 = M()
# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

# run the model
input_fp32 = torch.randn(4, 4, 4, 4)
res = model_int8(input_fp32)
```
- [Pytorch mobile](https://pytorch.org/mobile/ios/)
