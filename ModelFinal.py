import numpy as np
import pandas as pd 
from sklearn import preprocessing
from keras.layers.core import Dense, Dropout, Activation
from keras.activations import linear
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib import pyplot
from  prepareData import x,y,scaler_x,scaler_y,train_end
import time
import os

#**** Hyperparameters *****
#Random seed
seed = 2016
#LSTM Batch - data per iteration 
batchsize = 7
# Epochs or iterations
epochs_number = 200
#LSTM units - output_dim
layer_input_units = 4 
#Regularisateur contre overfitting
Dropout_reg_value = 0.1
# optimizer used - sgd or rmsprop
optimizer_used = 'rmsprop'
# Data shuffle mode
shuffleData = False
#Data File
data_file_name = "DailyDemandDataFactors.csv"
#show Plots
showPlot = True

# read and prepare data from datafile
# # data_csv = pd.read_csv(data_file_name, delimiter = ';',header=None, usecols=[3,4,5,6,7,8,9,10,11,12,13,14])
# Lire ligne par ligne
# # data = data_csv[1:]
# Renommer les colonne
# # data.columns = ['SumRetrait','CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
# print (data.head(10))
# pd.options.display.float_format = '{:,.0f}'.format
# supprimer les lignes dont la valeur est null ( au moins une valeur null)
# # data = data.dropna ()
# Output Y avec son type
# # y=data['SumRetrait'].astype(float)
# # cols=['CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
# # x=data[cols].astype(float)
# # print("longeur de y",len(y))
# # train_end = round((len(y)*Training_range)/100)
# # print("Training data count: ",train_end,"/",len(y))

# print(x.head())
# print(y.head())

# scaling data-Normalisation
# # scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
# construir le format des input (3 Dimensions pour les reseaux de neuronne)
# # x = np.array(x).reshape ((len(x),11 ))
# # x = scaler_x.fit_transform(x)
# # print(x[-1,:])
# # scaler_y = preprocessing.MinMaxScaler(feature_range =(-1, 1))
# # y = np.array(y).reshape ((len(y), 1))
# # y = scaler_y.fit_transform(y)

# Split train and test data
x_train=x[0: train_end ,]
x_test=x[train_end +1: ,]
y_train=y[0: train_end]
y_test=y[train_end +1:] 
x_train=x_train.reshape(x_train.shape +(1,))
x_test=x_test.reshape(x_test.shape + (1,))
print("Data well prepared")
print ('x_train shape ', x_train.shape)
print ('y_train', y_train.shape)

#Design the model - LSTM Network
#generateur des nombres aléatoire des poids
np.random.seed(seed)
#nomer le model = fit1
fit1 = Sequential ()
fit1.add(LSTM(
	activation="tanh", 
	input_shape=(11, 1), 
	units=layer_input_units))
fit1.add(Dropout(Dropout_reg_value))
fit1.add(Dense(units =1))
fit1.add(Activation(linear))

#rmsprop or sgd
fit1.compile(loss="mean_squared_error",optimizer=optimizer_used)
start = time.time()

#train the model
fit1.fit(x_train , y_train , batch_size = batchsize, epochs =epochs_number, shuffle=shuffleData)
t = round((time.time() - start) ,3)
print("************* Training Sammary ****************")
print("Training Time: ", t," sec")
print(fit1.summary ())
#Model error
print("************* Training Vs Test MSE Error ****************")
score_train = fit1.evaluate(x_train ,y_train ,batch_size =batchsize)
score_test = fit1.evaluate(x_test , y_test ,batch_size =batchsize)
print("in  train  MSE = ",round(score_train,4))
print("in test  MSE = ",round(score_test ,4))

#Make prediction
pred1 = fit1.predict(x_test)
pred1 = scaler_y.inverse_transform(np.array(pred1).reshape ((len(pred1), 1))).astype(int)
real_test = scaler_y.inverse_transform(np.array(y_test).reshape ((len(y_test), 1))).astype(int)
##############################
##Save the Model weights to json file
##
# serialize model to JSON
model_json = fit1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
fit1.save_weights("model.h5")
print(">>>> Model saved to model.h5 in the disk")

##############################
#save prediction
testData = pd.DataFrame(real_test)
preddData = pd.DataFrame(pred1)
dataF = pd.concat([testData,preddData], axis=1)
dataF.columns =['Real demand','predicted Demand']
dataF.to_csv('Demandprediction.csv')
print(">>> Test values saved into Demandprediction.csv file ")
print("*** Ploting the result...")

if showPlot:
	pyplot.plot(pred1, label='forecast')
	pyplot.plot(real_test,label='actual')
	pyplot.legend()
	pyplot.show()

