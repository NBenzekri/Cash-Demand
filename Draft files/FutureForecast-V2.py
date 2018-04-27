import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from sklearn import preprocessing
import os
data_file_name = "TrainingDataNotNull.csv"
#read and prepare data from datafile
data_csv = pd.read_csv(data_file_name, delimiter = ',',header=None, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
data = data_csv[1:]
data.columns = ['SumRetrait','CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']

#scaling data
scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
data = np.array(data).reshape ((len(data),12 ))
data = scaler_x.fit_transform(data)
x = data[-1][1:] 
print(x)
# pd.options.display.float_format = '{:,.0f}'.format

# # y=data['SumRetrait'].astype(float)
# # cols=['CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
# # x=data[cols].astype(float)
# # print("longeur de y",len(y))
# # train_end = round((len(y)*Training_range)/100)
# # print("Training data count: ",train_end,"/",len(y))

# print(x.head())
# print(y.head())
# scaling data
# # scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
# # x = np.array(x).reshape ((len(x),11 ))
# # x = scaler_x.fit_transform(x)
# # print(x[-1,:])

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

#FeaturesTest = [267,61200,695,677,70600,116700,130200,768,659,741,419300]
#xaa = np.array(FeaturesTest).reshape ((1,11 )).astype(float)
 
#x = x.reshape(x.shape +(1,))
print("print FeaturesTest scalled: ")
print(x)
tomorrowDemand = loaded_model.predict(x)
print("tomorrowDemand scalled: ", tomorrowDemand[0])
prediction = scaler_x.inverse_transform(np.array(tomorrowDemand).reshape ((len(tomorrowDemand), 1))).astype(int)
print ("la demande reelle est 95900 et la prediction est: ", prediction[0])

