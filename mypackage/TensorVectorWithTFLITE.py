import tensorflow as tf
import numpy as np
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:/Users/Bagas/Documents/vito/models/tflite4/model4.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class TensorVectorWithTFLITE(object):
    def __init__(self, FileName=None):
        self.FileName = FileName
    def process(self):
        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        input_data = np.array(img, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        return list(output_data)