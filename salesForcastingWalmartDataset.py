# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 09:15:15 2019

@author: Yutish-pc
"""

# importing libraries..........................................
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# getting the dataset
train_data = pd.read_csv('train.csv') 
features_data = pd.read_csv('features.csv')
stores_data = pd.read_csv('stores.csv')


#data preprocessing..................................................

#selecting column in dataset
result = pd.merge(train_data,
                  features_data[['Store','Date','Temperature','Fuel_Price']],
                  how = 'inner',
                  left_on = ['Store','Date'],
                  right_on = ['Store','Date'])

dataset = pd.merge(result,stores_data,on = 'Store')

# creating dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_data = LabelEncoder()
dataset.iloc[:, 4] = labelEncoder_data.fit_transform(dataset.iloc[:, 4])
 
labelEncoder_0 = LabelEncoder()
dataset.iloc[:, 0] = labelEncoder_0.fit_transform(dataset.iloc[:, 0])
labelEncoder_1 = LabelEncoder()
dataset.iloc[:, 1] = labelEncoder_1.fit_transform(dataset.iloc[:, 1])
labelEncoder_7 = LabelEncoder()
dataset.iloc[:, 7] = labelEncoder_7.fit_transform(dataset.iloc[:, 7])

dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)

onehotencoder  = OneHotEncoder(categorical_features= [0,1,7])
dataset = onehotencoder.fit_transform(dataset).toarray()

# setting X and y
X = dataset
y = train_data.iloc[:,3].values

# splitting of data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, 
                                                 random_state = 0)

# fearute scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


# making the model (Random Forest regression)................................
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20 , random_state = 0)
regressor.fit(X_train,y_train)


# predictiong the result
y_pred = regressor.predict(X_test)


# plotting the results.............................................
from sklearn.metrics import accuracy_score, confusion_matrix 


cm = confusion_matrix(y_test.astype(np.int64), y_pred.astype(np.int64))
acc = accuracy_score(y_test.astype(np.int64), y_pred.astype(np.int64))

plt.plot(y_test)
plt.show()
plt.plot(y_pred)
plt.show()


fig, ax= plt.subplots()
ax.scatter(y_test,y_pred,edgecolors=(0, 0, 0))
ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Ground truth vs Predicted')
plt.show()
