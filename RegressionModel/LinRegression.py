import numpy as np
import pandas as pd 
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import time
import os


showPlot=True

#prepare data
data_file_name = "../FinalCost.csv"

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
print(data.head())
x_train ,x_test ,y_train ,y_test = train_test_split( x,y, test_size=0.2	, random_state=1116)
print(type(y_test))
#print(y_test)
print(x.shape)
#Design the Regression Model
regressor =LinearRegression()
##training
regressor.fit(x_train,y_train)

#Make prediction
y_pred =regressor.predict(x_test)
# print (y_pred)
# print("---- test----")
#print(y_test)

YArray = y_test.as_matrix()
testData = pd.DataFrame(YArray)
preddData = pd.DataFrame(y_pred)
meanError = np.abs((YArray - y_pred)/YArray)*100
meanError2 = np.abs((YArray - y_pred))
print("Mean: ", meanError.mean()," - ", meanError2.mean())

dataF = pd.concat([testData,preddData], axis=1)
dataF.columns =['Real demand','predicted Demand']
dataF.to_csv('Predictions.csv')
print(">>> Test values saved into amina.csv file ")

#vendredi;2018-03-16;116700;179,10370,;753,685,127100,119800,145500,760,721,768,4000;GAB_02
Xnew = [[179,10370,753,685,127100,119800,145500,760,721,768,4000]]
# make a prediction
ynew = regressor.predict(Xnew)
# show the inputs and predicted outputs
print("X= 116700 , Predicted=%s" %  ynew[0])

if showPlot:
	pyplot.plot(y_pred,'r-', label='forecast')
	pyplot.plot(YArray,'b-',label='actual')
	pyplot.legend()
	pyplot.show()





