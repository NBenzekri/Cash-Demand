import numpy as np
import pandas as pd 
from sklearn import preprocessing
import os

#Data File
data_file_name = "../../data3.csv"
# Training data %
Training_range = 80

#read and prepare data from datafile
data_csv = pd.read_csv(data_file_name, delimiter = ';',header=None, usecols=[2,5,6,7,8,9,10,11,12,13,14])

#Lire ligne par ligne
data = data_csv[1:]
#Renommer les colonne
data.columns = ['SumRetrait','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
# print (data.head(10))
# pd.options.display.float_format = '{:,.0f}'.format
#supprimer les lignes dont la valeur est null ( au moins une valeur null)
data = data.dropna ()
#Output Y avec son type
print(data.describe(include = "all"))
print("")
y=data['SumRetrait'].astype(float)
print(y.describe(include = "all"))
cols=['ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
x=data[cols].astype(float)
print("longeur de y",len(y))

train_end = round((len(y)*Training_range)/100)
print("Training data count: ",train_end,"/",len(y))

#scaling data-Normalisation 
scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
#construir le format des input (3 Dimensions pour les reseaux de neuronne)
x = np.array(x).reshape ((len(x),10))
x = scaler_x.fit_transform(x)
#print(x[-1,:])
scaler_y = preprocessing.MinMaxScaler(feature_range =(-1, 1))
y = np.array(y).reshape ((len(y), 1))
y = scaler_y.fit_transform(y)
