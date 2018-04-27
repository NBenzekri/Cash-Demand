import numpy as np
import pandas as pd 
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot
import time
import os


showPlot=True

#prepare data
data_file_name = "../DailyDemandDataFactors.csv"

data_csv = pd.read_csv(data_file_name, delimiter = ',',header=None, usecols=[3,4,5,6,7,8,9,10,11,12,13,14])
data = data_csv[1:]
data.columns = ['SumRetrait','ConsommationHier','CountTransaction','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
data = data.dropna ()
y=data['SumRetrait'].astype(int)
cols=['ConsommationHier','CountTransaction','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
x=data[cols].astype(int)
x_train ,x_test ,y_train ,y_test = train_test_split( x,y, test_size=0.2	, random_state=1116)
print(type(y_test))
#Design the Regression Model
regressor =LinearRegression()
##training
regressor.fit(x_train,y_train)

#Make prediction
y_pred =regressor.predict(x_test)
# print (y_pred)
# print("---- test----")
# print(y_test)

# for i in range(len(y_pred)):
# 	print("Real = %s  , Predicted = %s" % (y_test[i], y_pred[i]))
YArray = y_test.as_matrix()
print(YArray)
testData = pd.DataFrame(YArray)
preddData = pd.DataFrame(y_pred)
diff = np.abs((YArray - y_pred)/YArray)*100
print(diff)
print("mean: %s", diff.mean())
dataF = pd.concat([testData,preddData], axis=1)
dataF.columns =['Real demand','predicted Demand']
dataF.to_csv('amina.csv')
print(">>> Test values saved into amina.csv file ")

Xnew = [[292,147800,822,742,121200,127100,178300,768,759,741,419300]]
# make a prediction
ynew = regressor.predict(Xnew)
# show the inputs and predicted outputs
print("X= 140400 , Predicted=%s" %  ynew[0])

if showPlot:
	pyplot.plot(y_pred,'r-', label='forecast')
	pyplot.plot(YArray,'b-',label='actual')
	pyplot.legend()
	pyplot.show()





