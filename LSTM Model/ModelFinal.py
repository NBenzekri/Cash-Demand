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
import datetime
import os

#**** Hyperparameters *****
#Random seed
seed = 2016
#Features count
fCount = 11
#LSTM Batch - data per iteration 
batchsize = 7
# Epochs or iterations
epochs_number = 1000
#LSTM units - output_dim
layer_input_units = 10
#Regularisateur contre overfitting
Dropout_reg_value = 0.1
# optimizer used - sgd or rmsprop
optimizer_used = 'rmsprop'
# Data shuffle mode
shuffleData = False
#Data File
#data_file_name = "../DailyDemandDataFactors.csv"
#show Plots
showPlot = True

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
print ('y_test', y_test.shape)

#Design the model - LSTM Network
#generateur des nombres aléatoire des poids
np.random.seed(seed)
#nomer le model = fit1
fit1 = Sequential ()
fit1.add(LSTM(
	activation="tanh", 
	input_shape=(fCount, 1), 
	units=layer_input_units))
fit1.add(Dropout(Dropout_reg_value))
fit1.add(Dense(units =1))
fit1.add(Activation(linear))

#rmsprop or sgd
fit1.compile(loss="mse",optimizer=optimizer_used)
start = time.time()

#train the model
fit1.fit(x_train , y_train , batch_size = batchsize, epochs =epochs_number, shuffle=shuffleData)
t = round((time.time() - start) ,3)
print("************* Training Sammary ****************")
traintime = str(datetime.timedelta(seconds=t))
print("Training Time: ", traintime )
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

meanError = np.abs((real_test - pred1)/real_test)*100
meanError2 = np.abs((real_test - pred1))
print("mean: ", meanError.mean()," - ", meanError2.mean())
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

