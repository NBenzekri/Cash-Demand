import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
from prepareData import scaler_y
from prepareData import scaler_x
import os

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
#############################################################
# make prediction with the loaded model

####

FeaturesTest = [36 ,95900,695,676,126700 ,88100 ,122800 ,768 ,659 ,741 ,419300]
xaa = np.array(FeaturesTest).reshape ((1,11 )).astype(float)
xaa = scaler_x.transform(xaa) 
xaa = xaa.reshape(xaa.shape +(1,))
tomorrowDemand = loaded_model.predict(xaa)
print("tomorrowDemand scalled: ", tomorrowDemand[0])
prediction = scaler_y.inverse_transform(np.array(tomorrowDemand).reshape ((len(tomorrowDemand), 1))).astype(int)
print ("la demande reelle est 95900 et la prediction est: ", prediction[0])

