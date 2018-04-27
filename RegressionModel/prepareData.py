import numpy as np
import pandas as pd 
from sklearn import preprocessing
import os

#Data File
data_file_name = "../DailyDemandDataFactors.csv"
# Training data %
Training_range = 80

#read and prepare data from datafile
data_csv = pd.read_csv(data_file_name, delimiter = ';',header=None, usecols=[3,4,5,6,7,8,9,10,11,12,13,14])
#Lire ligne par ligne
data = data_csv[1:]
#Renommer les colonne
data.columns = ['SumRetrait','CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
# print (data.head(10))
# pd.options.display.float_format = '{:,.0f}'.format
#supprimer les lignes dont la valeur est null ( au moins une valeur null)
data = data.dropna()
#Output Y avec son type
y=data['SumRetrait'].astype(float)
cols=['CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
x=data[cols].astype(float)
print("longeur de y",len(y))
train_end = round((len(y)*Training_range)/100)
print("Training data count: ",train_end,"/",len(y))

