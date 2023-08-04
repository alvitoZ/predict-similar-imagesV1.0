import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %matplotlib inline

global embed

#use hub model
embed = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")

#use saved_model.load / local model
# embed = tf.saved_model.load("C:/Users/Bagas/Documents/vito/convertModelToTflite/convert_model4")

class TensorVector(object):

    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self):

        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        return list(feature_set)