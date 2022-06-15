from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


image_path = './dataset/covid/lancet-case2a.jpg'
# image_path = './dataset/normal/IM-0466-0001.jpeg'
model_path = './Covid19.h5'


def loadImage(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)   
    img_tensor /= 255.    
    return img_tensor 



def predictImage(imgTensor, ):
    pass


if __name__ == "__main__":
    model = load_model(model_path)
    imgTensor = loadImage(image_path)
    pred = model.predict(imgTensor)
    print(pred)
    