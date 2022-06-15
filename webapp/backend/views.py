from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from PIL import Image
import io



def loadImage(img):
    # img = image.load_img(img_path, target_size=(224, 224))
    img = img.resize((224, 224))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)   
    img_tensor /= 255.    
    return img_tensor 

def makePrediction(img_tensor):
    model_path = os.path.join(settings.MODELS, "Covid19.h5")
    model = load_model(model_path)
    pred = model.predict(img_tensor)
    return pred

def indexPage(request):
    if request.method == 'POST':
        f = request.FILES['img_file'].read()
        img = Image.open(io.BytesIO(f))
        img = img.convert('RGB')
       
        imgTensor = loadImage(img)
        
        predictions = makePrediction(imgTensor)
        print(predictions)
        return JsonResponse({
            "status" : "success"
        })

    return  render(request, 'index.html')