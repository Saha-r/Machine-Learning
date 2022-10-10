# Machine-Learning
simple + perplex proj

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
​
!wget https://github.com/jadijadi/machine_learning_with_python_jadi/blob/main/FuelConsumption.csv
df = pd.read_csv("C:\\Users\\SAZGAR\\Desktop\\FuelConsumption.csv")
df.head()
df.describe()
cdf = df[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']]
cdf.head(9)
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlable("FUELCONSUMPTION_COMB")
plt.ylable("CO2EMISSIONS")
plt.show()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlable("ENGINESIZE")
plt.ylable("CO2EMISSIONS")
plt.show()
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red')
plt.xlable("CILINDERS")
plt.ylable("CO2EMISSIONS")
plt.show()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf [~msk]
print(msk)
print(~msk)
print(cdf)
print(train)
print(test)
ax1 = fig.add_subplot(111)
ax1.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
ax1.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='red')
plt.xlable("ENGINESIZE")
plt.ylable("CO2EMISSIONS")
plt.show()
pip install scikit-learn
import sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlable(["ENGINESIZE"])
plt.ylable(["Emission"])
