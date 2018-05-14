import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as pyplot
from sklearn import preprocessing
from sklearn.metrics import r2_score
#read and prepare data from datafile
#Data File
train_range =80


data_file_name = "FinalCost.csv"
data_csv = pd.read_csv(data_file_name, delimiter = ';',header=None, usecols=[3,4,5,6,7,8,9,10,11,12,16,17])
#Lire ligne par ligne
data = data_csv[1:]
print(data.shape)
#Renommer les colonne
data.columns = ['ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMmJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer','PoidTot','SumRetrait']


# print (data.head(10))
# pd.options.display.float_format = '{:,.0f}'.format
#supprimer les lignes dont la valeur est null ( au moins une valeur null)
data = data.dropna ()
print(data.shape)
#Output Y avec son type
y=data['SumRetrait'].astype(float)

train_end =round((len(y)*train_range)/100)
cols=['ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMmJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer','PoidTot']
x=data[cols].astype(float)

ones = np.ones([len(x),1])
x = np.concatenate((ones,x),axis=1)

print('Normalisation')
scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
x = scaler_x.fit_transform(x)
scaler_y = preprocessing.MinMaxScaler(feature_range =(-1, 1))
y = np.array(y).reshape ((len(y), 1))
y = scaler_y.fit_transform(y)

x_train=x[0: train_end ,]
x_test=x[train_end +1: ,]
y_train=y[0: train_end]
print(y_train.shape)
y_test=y[train_end +1:] 

theta = np.zeros([1,12])
alpha = 0.001
iters = 10000
def computeCost(x,y,theta):
	tobesummed = np.power(((x @ theta.T)-y),2)
	return np.sum(tobesummed)/(2 * len(x))


def gradientDescent(x,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
	
        theta = theta - (alpha/len(x)) * np.sum(x * (x @ theta.T - y), axis=0)
	
        cost[i] = computeCost(x, y, theta)
    
    return theta,cost

#running the gd and cost function
theta,cost = gradientDescent(x_train,y_train,theta,iters,alpha)
finalCost = computeCost(x_train,y_train,theta)
print('le cost final')
#print(finalCost)


def computeCostgeneraliser(x,y,theta):
	tobesummed = np.power(((x @ theta.T)-y),2)
	return np.sum(tobesummed)/(2 * len(x))
ErreurGeneral =computeCostgeneraliser(x_test,y_test,theta)
print('the general cost')
print(ErreurGeneral)
#predict 
def predict(x,theta):

	y=(x @ theta.T)
	
	return y
	
forecast = predict(x_test,theta)
y_pred =scaler_y.inverse_transform(np.array( forecast).reshape ((len( forecast), 1)))
y_test2=scaler_y.inverse_transform(np.array( y_test).reshape ((len(y_test), 1)))
#print(y_pred)
# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse
def mse(Y, Y_pred):
    mse = sum((Y - Y_pred) ** 2) / len(Y)
    return mse

# Model Evaluation - R2 Score
def r2_score2(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

diff = np.abs((y_test2 - y_pred)/y_test2)*100
print("Mean: ", diff.mean())
print("MSE: ", mse(y_pred, y_test2)[0])
print("RMSE: ", rmse(y_pred, y_test2)[0])
print("R^2: ", r2_score2(y_pred, y_test2))


testData = pd.DataFrame(y_test2)
preddData = pd.DataFrame(y_pred)
dataF = pd.concat([testData,preddData], axis=1)
dataF.columns =['Real demand','predicted Demand']
dataF.to_csv('RegressionResult.csv')
print(">>> Test values saved into RegressionResult.csv file ")
print("*** Ploting the result...")


pyplot.plot(y_pred,'r-', label='Forecast')
pyplot.plot(y_test2,'b-',label='Actual')
pyplot.legend()
pyplot.show()
