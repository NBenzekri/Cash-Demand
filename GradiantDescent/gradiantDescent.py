import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as pyplot
from sklearn import preprocessing
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
print('amiiiiiiiiiiiiiiina')
print(y_train.shape)
y_test=y[train_end +1:] 

theta = np.zeros([1,12])
alpha = 0.01
iters = 4000
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
print(finalCost)


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
diff = np.abs((y_test2 - y_pred)/y_test2)*100
print(diff)
print("mean: %s", diff.mean())


testData = pd.DataFrame(y_test2)
preddData = pd.DataFrame(y_pred)
dataF = pd.concat([testData,preddData], axis=1)
dataF.columns =['Real demand','predicted Demand']
dataF.to_csv('RegressionResult.csv')
print(">>> Test values saved into Demandprediction.csv file ")
print("*** Ploting the result...")


pyplot.plot(y_pred,'r-', label='forecast')
pyplot.plot(y_test2,'b-',label='actual')
pyplot.legend()
pyplot.show()
#predict 
# def predict(x_tst):
	# xaa = np.array(x_tst).reshape ((1,11 )).astype(float)
	# xaa = scaler_x.transform(xaa) 
	# y_tst = np.sum(xaa @ g.T)
	# y_tst= scaler_y.inverse_transform(np.array(y_tst).reshape ((1, 1)))
	# return y_tst
# pred = np.zeros([len(y_test),1])
# err = np.zeros([len(y_test),1])
# y_test= scaler_y.inverse_transform(np.array(y_test).reshape ((len(y_test), 1)))

# print(len(y_test))
# for i in range(0,len(y_test)):
	# pred[i] = np.abs(predict(x_test[i]))
	# err[i] = np.abs((y_test[i] - pred[i])/y_test[i])*100	
	# print('--------- Real :',y_test[i],'------Predicted :' ,pred[i],'---- Erreur',err[i],'%')
# if(err.mean()>50):
	# x= ' bad results'
# else:
    # x= ' good results'
# print('\n -------- Erreur Moyen est : ',err.mean(), '%', x)





# print('----- erreur\n',err)
		
# def predicttest(test):	
	# for i in range(0,len(y_test)):
		# pred[i] = predict(test[i])
		# err[i] = np.abs((y_test[i] - pred[i])/y_test[i])*100	
	# return pred,err
	
#predicted, erreur = predicttest(x_test)
# print(predicted)
# print(erreur)
# print(erreur.mean())
# # FeaturesTest = [[1,95900,695,676,126700 ,88100 ,122800 ,768 ,659 ,741 ,419300]]
# FeaturesTest = [1,77300,822,693,93200,103700,87400,768,	759,741,419300]
# # xaa = np.array(FeaturesTest).reshape ((1,11 )).astype(float)
# # xaa = scaler_x.transform(xaa) 
# # print(xaa.shape)
# # xaa = xaa.reshape(xaa.shape +(1,))
# print('Result')
# y_train1=predict(FeaturesTest)

# print(y_train1)
# diff = np.abs((61200 - y_train1)/61200)*100
# # print(diff)
# print("mean: %s", diff.mean())
# # y_train1= scaler_y.inverse_transform(np.array(y_train1).reshape ((1, 1)))
# # print(y_train1)
# # fig, ax = plt.subplots()  
# # ax.plot(np.arange(iters), cost, 'r')  
# # ax.set_xlabel('Iterations')  
# # ax.set_ylabel('Cost')  
# # ax.set_title('Error vs. Training Epoch')  

# # finalCost = computeCost(X,y,g)
# # print(finalCost)