import numpy as np
import pandas as pd 
from sklearn import preprocessing
from keras.layers.core import Dense, Dropout, Activation
from keras.activations import linear
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib import pyplot

#read and prepare data from datafile
data_file_name = "DailyDemandDataFactors.csv"
data_csv = pd.read_csv(data_file_name, delimiter = ';',header=None, usecols=[3,4,5,6,7,8,9,10,11,12,13,14])
yt = data_csv[1:]
data = yt
data.columns = ['SumRetrait','CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
# print (data.head(10))
pd.options.display.float_format = '{:,.0f}'.format
data = data.dropna ()
y=data['SumRetrait'].astype(int)
cols=['CountTransaction','ConsommationHier','MSemaineDernier','MSemaine7','ConsoMmJrAnP','ConsoMmJrMP','ConsoMMJrSmDer','MoyenneMoisPrec','MoyenneMMSAnPrec','MoyenneMMmAnPrec','ConsommationMaxMDer']
x=data[cols].astype(float)
print("longeur de y",len(y))
print(x.head())
print(y.head())
#scaling data
scaler_x = preprocessing.MinMaxScaler(feature_range =(-1, 1))
x = np.array(x).reshape ((len(x),11 ))
x = scaler_x.fit_transform(x)
scaler_y = preprocessing.MinMaxScaler(feature_range =(-1, 1))
y = np.array(y).reshape ((len(y), 1))
y = scaler_y.fit_transform(y)

# Split train and test data
train_end = 80
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
seed = 2016
np.random.seed(seed)
fit1 = Sequential ()
fit1.add(LSTM(
	output_dim = 6,
	activation='tanh',
	input_shape =(11, 1)))
fit1.add(Dropout(0.01))
fit1.add(Dense(output_dim =1))
fit1.add(Activation(linear))
#rmsprop or sgd
batchsize = 7
fit1.compile(loss="mean_squared_error",optimizer="rmsprop")
#train the model
fit1.fit(x_train , y_train , batch_size = batchsize, nb_epoch =200, shuffle=False)

print(fit1.summary ())

#Model error
score_train = fit1.evaluate(x_train ,y_train ,batch_size =batchsize)
score_test = fit1.evaluate(x_test , y_test ,batch_size =batchsize)
print("in  train  MSE = ",round(score_train,4))
print("in test  MSE = ",round(score_test ,4))

#Make prediction
pred1=fit1.predict(x_test)
pred1 = scaler_y.inverse_transform(np.array(pred1).reshape ((len(pred1), 1)))
real_test = scaler_y.inverse_transform(np.array(y_test).reshape ((len(y_test), 1))).astype(int)


# rmse = np.square(np.subtract(real_test, pred1)).mean()
# print('Test RMSE: %.3f' % rmse)

# serialize model to JSON
model_json = fit1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
fit1.save_weights("model.h5")
print("Saved model to disk")

####
# FeaturesTest = [36,95900,695,676,126700,88100,122800,768,659,741,419300]
# xaa = np.array(FeaturesTest).reshape ((1,11 ))
# xaa = scaler_x.fit_transform(xaa) 
# xaa=xaa.reshape(xaa.shape +(1,))
# tomorrowDemand = fit1.predict(xaa)
# prediction = scaler_y.inverse_transform(np.array(tomorrowDemand).reshape ((len(tomorrowDemand), 1))).astype(int)
# print ("la demande est: ",prediction)
####
#save prediction
testData = pd.DataFrame(real_test)
preddData = pd.DataFrame(pred1)
dataF = pd.concat([testData,preddData], axis=1)
dataF.columns =['Real demand','predicted Demand']
dataF.to_csv('Demandprediction2.csv')



pyplot.plot(pred1, label='forecast')
pyplot.plot(real_test,label='actual')
pyplot.legend()
pyplot.show()
