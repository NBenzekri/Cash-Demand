
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from matplotlib import pyplot
from sklearn.cross_validation import train_test_split
from  prepareData import x,y,train_end
import time
import os

inputX = sm.add_constant(X)
x_train ,x_test ,y_train ,y_test = train_test_split( inputX,y, test_size=0.2,random_state=0)
# Note the difference in argument order

model = sm.OLS(y_train, x_train).fit() ## sm.OLS(output, input)
predictions = model.predict(x_train)

# Print out the statistics
print(model.summary())

pyplot.plot(predictions, label='forecast')
pyplot.plot(y_train,label='actual')
pyplot.legend()
pyplot.show()

