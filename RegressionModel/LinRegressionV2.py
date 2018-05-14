
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from  prepareData import x,y,train_end
import time
import os

#inputX = sm.add_constant(x)
#print(inputX.shape)
print(y.shape)
x_train ,x_test ,y_train ,y_test = train_test_split( x,y, test_size=0.2,random_state=0)
# Note the difference in argument order

model = sm.OLS(y_train, x_train).fit() ## sm.OLS(output, input)
predictions = model.predict(x_test)
YArray = y_test.as_matrix()
pred = predictions.as_matrix()

# Print out the statistics
print(model.summary())
m = np.abs((YArray - pred)/YArray)*100
print('----------------- Error moy --------------------', m.mean(),'\n')

## test 
Xnew = [[292,147800,822,742,121200,127100,178300,768,759,741,419300]]
#inputY = sm.add_constant(Xnew)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("X= 140400 , Predicted=%s" %  ynew[0])


pyplot.plot(pred, label='forecast')
pyplot.plot(y_test,label='actual')
pyplot.legend()
pyplot.show()

