from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import  accuracy_score

from imutils import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import cv2
import os 

import warnings
warnings.filterwarnings('ignore')

# agumets parsed
ag = argparse.ArgumentParser()
ag.add_argument("-m", "--model",type=str, default="Clowncovid19.model",
                 help="Path to  output loss/accuracy plot")
ag.add_argument("-p", "--plot", type=str, default="plot.png",
                help="Path to  output loss/accuracy plot" )

args = vars(ag.parse_args)

# Initalizing Learning Rate, Number of Epochs and Batch Size
LR = 1e-3
Epochs = 25
batch_size = 8

print("[Loading Images]....\n")

imagePaths = list(paths.list_images('./dataset')) # gets the the paths in that folder
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2] # return either normal or covid 

    image = cv2.imread(imagePath) # reading each image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    image = cv2.resize(image, (224,224)) # resize to 224x224 pixels

    data.append(image) 
    labels.append(label)

# converting the data and labels to np arrays
data = np.array(data) / 255.0 # division is to scale the intensities to range[0,1]
labels = np.array(labels)

''' 
 perfoming hot encoding on labels 
 i,e labels normal, covid are converted into binary values 0,1

'''
print("\n[Binarizing The Labels]....\n")
lb = LabelBinarizer()
labels = lb.fit_transform(labels) 
labels = to_categorical(labels) # convert single line to pairs of opp binaries ie [[0],[1]] -> [0,1] or [1,0]

print("\n[Spliting Data]....\n")
# data formation for train and test ; 80% train and 20% test split is done
x_train, x_test, y_train, y_test = train_test_split( data, labels,
                                                     test_size = 0.20, 
                                                     stratify = labels,
                                                     random_state = 42)

# initializing traing data argument object

trainAug = ImageDataGenerator(
    rotation_range = 15,
    fill_mode = 'nearest'
)


print("\n[Creating CNN Layer For  Model]....\n")
# loading VGG16 convolution neutral network model

# base layer model
clownBaseModel = VGG16( weights = 'imagenet', 
                    include_top = False,
                    input_tensor = Input(shape = (224,224,3))) # input layer


# head layer model that will be placed on the top of  based model

clownHeadModel = clownBaseModel.output
clownHeadModel = AveragePooling2D(pool_size = (4,4))(clownHeadModel)
clownHeadModel = Flatten(name = 'flatten')(clownHeadModel)
clownHeadModel = Dense(64, activation = 'relu')(clownHeadModel)
clownHeadModel = Dropout(0.5)(clownHeadModel)
clownHeadModel = Dense(2, activation = 'softmax')(clownHeadModel)


print("\n[Layers Created Successfully]....\n")

print("\n[Creating Model From Layers ]....")

clownModel = Model(inputs = clownBaseModel.input, outputs = clownHeadModel)

print("\n[Model Sucessfully Created]....")

# looping over all layers in the base model and freeze them so they will
# *not* be updating it  during the first training process

for layer in clownBaseModel.layers:
    layer.trainable = False


print("\n[Compiling Model]....")

clownOptimizer = Adam(lr = LR, decay = LR/Epochs)
clownModel.compile(loss = "binary_crossentropy", 
                    optimizer = clownOptimizer, 
                    metrics = ["accuracy"])


print("\n[Training Head]....")

H = clownModel.fit_generator(
    trainAug.flow(x_train, y_train, batch_size = batch_size),
    steps_per_epoch = len(x_train) // batch_size,
    validation_data = (x_test, y_test),
    validation_steps = len(x_test) // batch_size,
    epochs = Epochs
)

print("\n[Evaluating Network]....")
print("\n[Making Predictions On Test Set]....")

predIDxs = clownModel.predict(x_test, batch_size = batch_size)
predIDxs = np.argmax(predIDxs, axis = 1)

print("\n[Getting Classification Report]....\n")
print(classification_report(y_test.argmax(axis = 1), predIDxs, target_names = lb.classes_))

print("\n[Getting Confusion Matrix And Accuracy Score]....\n")
cm = confusion_matrix(y_test.argmax(axis = 1), predIDxs)
total = sum(sum(cm))
accuracy_score = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
specificity = cm[1,1] / (cm[1,0] / cm[1,1])

print(cm)
print('Accuracy_score : ',accuracy_score)
print('Sensitivity : {:.4f}'.format(sensitivity))
print('Specificity : {:.4f}'.format(specificity))


print("\n[Ploting Training Loss vs Accuracy Graph]....\n")

N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')

print("\n[Saving The Covid Model]....\n")
clownModel.save('ClownCovid19.model', save_format = 'h5')
