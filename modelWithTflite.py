import numpy as np
import os
import matplotlib.pyplot as plt 
import json

from mypackage import convertBase64, cosineSim, TensorVectorWithTFLITE, validate

# plt.imshow(convertBase64("image.jpg"))
# plt.show()


#beginning predict single similar images
#predict single image
def predictSingleSimilar(image1, image2):   
    helper1 = TensorVectorWithTFLITE(image1)
    vector1 = helper1.process()
    helper2 = TensorVectorWithTFLITE(image2)
    vector2 = helper2.process()
    return json.dumps({
            "index":0,
            "image":[image1, image2],
            "result":validate(cosineSim(vector1, vector2)),
        })

#variables
# x = str(input("input image 1 : "))
# y = str(input("input image 2 : "))

# how call the function
# print(predictSingleSimilar("image.jpg", "predict2/20230724_170942.jpg"))
# print(predictSingleSimilar(x, y))
#end predict single similar images


#beginning predict multiple similar images
#variables
# x = str(input("input 1 image for compare : "))
# folder_images = str(input("input image folder : "))
x = "image.jpg"
folder_images = "predict2"

#create and return images, image vectors
listImagesAndVector = [
    {"images":f"{folder_images}/{i}","vectors":np.array(TensorVectorWithTFLITE(f"{folder_images}/{i}").process())}
    for i in os.listdir(folder_images)
    ]

#predict multiple similar images
def predictMultipleSimilar(image,arr):   
    helper1 = TensorVectorWithTFLITE(image)
    vector = helper1.process()
    result = []
    for index in range(len(arr)):
        result.append({
            "index":index,
            "image":arr[index].get('images'),
            "result":validate(cosineSim(vector, arr[index].get('vectors'))),
        })
    return json.dumps(result)

#how call the function
# print(predictMultipleSimilar(arr=listImagesAndVector, image="image.jpg"))
# print(predictMultipleSimilar(x, listImagesAndVector))
#end predict multiple similar images
