__author__ = 'nav'

import pickle
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

XElements = np.zeros(23)
yColumn = np.array([])
clf = linear_model.LinearRegression()
charge_test = np.array([])

with open('X_Elements.pkl','rb') as fid:
        XElements = pickle.load(fid)

with open('Y_Column.pkl','rb') as fid:
        yColumn = pickle.load(fid)

with open('my_dumped_linreg.pkl','rb') as fid:
        clf = pickle.load(fid)

with open ('my_dumped_maxChargeTest.pkl', 'rb') as fid:
        charge_test = pickle.load(fid)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(XElements,
                                                                         yColumn,
                                                                         test_size=0.4,
                                                                         random_state = 0)

print(clf.coef_)
print('\n')
print('R^2 of predictor: ', clf.score(X_test,y_test))
predictColumn = clf.predict(X_test)
print('RMSE of predictor: ', mean_squared_error(y_test, predictColumn), '\n')

lenTestArray = len(y_test)
fixedVal = np.empty(lenTestArray)
fixedVal.fill(0.8)
print('RMSE Default p value of 0.8: ', mean_squared_error(y_test, fixedVal))

pVal = 0.76
fixedVal2 = np.empty(lenTestArray)
fixedVal2.fill(pVal)
print('RMSE Default p value of ', pVal, ': ', mean_squared_error(y_test, fixedVal2))

# charge_min = charge_test.min()
# charge_max = charge_test.max()
# y_min = y_test.min()
# y_max = y_test.max()

# Charge State vs. MLE Plots
'''plt.scatter(charge_test, predictColumn, c='g', alpha=0.5)
plt.title('Maximum Charge State vs. Predicted MLE values')
plt.xlabel('Maximum Charge State')
plt.ylabel('MLE values')
plt.show()

plt.scatter(charge_test, y_test, c='b', alpha=0.5)
plt.title('Maximum Charge State vs. Observed MLE values')
plt.xlabel('Maximum Charge State')
plt.ylabel('MLE values')
plt.show()'''

# Predicted values vs. Observed values
plt.scatter(predictColumn, y_test, c='b', alpha=0.5)
plt.hlines(y=0, xmin=0,xmax=1.5)
plt.xlabel('Predicted values')
plt.ylabel('Observed values')
xIdeal = [0,1,1.5]
yIdeal = [0,1,1.5]
plt.plot(xIdeal, yIdeal, 'g')
plt.show()

# Residual Plots
plt.scatter(clf.predict(X_train), clf.predict(X_train) - y_train, c='b', s=40,alpha=0.5)
plt.scatter(clf.predict(X_test), clf.predict(X_test) - y_test, c='g', s=40)
plt.hlines(y=0, xmin=0, xmax=2)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')
plt.show()

'''
plt.scatter(charge_test, y_test)
plt.xlabel('Maximum Charge States')
plt.ylabel('Observed MLE')
plt.xlim(charge_min, charge_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()'''

#predicted = cross_validation.cross_val_predict()