import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import matplotlib.pyplot as plt 
import base64
from PIL import Image
import io
import math
from math import sqrt
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %matplotlib inline

global embed
# embed = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
embed = hub.KerasLayer("tf2-preview_mobilenet_v2_feature_vector_4")


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


def convertBase64(FileName):
    """
    Return the Numpy array for a image 
    """
    with open(FileName, "rb") as f:
        data = f.read()
        
    res = base64.b64encode(data)
    
    base64data = res.decode("UTF-8")
    
    imgdata = base64.b64decode(base64data)
    
    image = Image.open(io.BytesIO(imgdata))
    
    return np.array(image)

# plt.imshow(convertBase64("image.jpg"))
# plt.show()


def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim

#validate result
def validate(value):
    message = value
    if(value >= 0.5):
        value
    else:
        message = "not similar"
    return message

#predict single image
def predictSingleSimilar(image1, image2):   
    helper1 = TensorVector(image1)
    vector1 = helper1.process()
    helper2 = TensorVector(image2)
    vector2 = helper2.process()
    return json.dumps({
            "index":0,
            "image":[image1, image2],
            "result":validate(cosineSim(vector1, vector2)),
            # "result":"{:.2f}".format(cosineSim(vector1, vector2)),
        })

# x = str(input("input image 1 : "))
# y = str(input("input image 2 : "))
# print(predictSingleSimilar("image.jpg", "predict2/20230724_170942.jpg"))
# print(predictSingleSimilar(x, y))


#variables
# folder_images = "predict2"
x = str(input("input 1 image for compare : "))
folder_images = str(input("input image folder : "))

#create and return images, image vectors
listImagesAndVector = [
    {"images":f"{folder_images}/{i}","vectors":np.array(TensorVector(f"{folder_images}/{i}").process())}
    for i in os.listdir(folder_images)
    ]

# print(listImagesAndVector)

def predictMultipleSimilar(image,arr):   
    helper1 = TensorVector(image)
    vector = helper1.process()
    result = []
    for index in range(len(arr)):
        result.append({
            "index":index,
            "image":arr[index].get('images'),
            "result":validate(cosineSim(vector, arr[index].get('vectors'))),
        })
    return json.dumps(result)
         

# print(predictMultipleSimilar(arr=listImagesAndVector, image="image.jpg"))
print(predictMultipleSimilar(x, listImagesAndVector))

