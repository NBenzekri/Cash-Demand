import numpy as np
import pandas as pd 
from sklearn import preprocessing
import os


# Training data %
Training_range = 80

##Data File
data_file_name = "../FinalCost.csv"
#Features count
fCount = 11

data_csv = pd.read_csv(data_file_name, delimiter = ';',header=None, usecols=[3,4,5,6,7,8,9,10,11,12,16,17])
#Lire ligne par ligne
data = data_csv[1:]
#Renommer les colonne
data.columns = ['ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer',
'MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer', 'PoidTot', 'SumRetrait']

# print (data.head(10))
# pd.options.display.float_format = '{:,.0f}'.format
#supprimer les lignes dont la valeur est null ( au moins une valeur null)
data = data.dropna ()
#Output Y avec son type
y=data['SumRetrait'].astype(float)
cols=['ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer',
'MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer', 'PoidTot']
x=data[cols].astype(float)
print("longeur de y",len(y))
train_end = round((len(y)*Training_range)/100)
print("Training data count: ",train_end,"/",len(y))

# print(x.head())
# print(y.head())

#scaling data-Normalisation
scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
#construir le format des input (3 Dimensions pour les reseaux de neuronne)
x = np.array(x).reshape ((len(x),fCount ))
x = scaler_x.fit_transform(x)
print(x[-1,:])
scaler_y = preprocessing.MinMaxScaler(feature_range =(-1, 1))
y = np.array(y).reshape ((len(y), 1))
y = scaler_y.fit_transform(y)
