# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:54:25 2019

@author: Geethanjali
"""

import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import spectral_clustering
import sklearn.cluster as cluster
import math
import statsmodels.api as api 
import sklearn.metrics as metrics
data1 = pandas.read_csv('C:\\Users\\Vanathi\\My scripts\\Assignments\\Assignment 3 blackboard\\Purchase_Likelihood.csv', 
                        delimiter=',')

k=len(data1['A'].unique())
print("k =", k)
print('\n 2.a- Number of Parameters in the model :',k ) 

Data1 = data1['A'].astype('category')
y = Data1
y_category = y.cat.categories

#Train Data
DataTrain = data1[['group_size', 'homeowner', 'married_couple']].astype('category')
data1.dtype
#
#
n = data1.shape[0]
marginalCount = pandas.DataFrame(data1.groupby('A').size())
print('\n Target variable is A, marginal counts of categories is given by:')
print('\n', marginalCount)

#Set Dummy value
X = pandas.get_dummies(DataTrain)

#X = pandas.get_dummies(DataTrain[['group_size']])
#X=X.join(Data[['homeowner', 'married_couple']])
X = api.add_constant(X, prepend=True)
#
print('Solution: 2h')
logit = api.MNLogit(y, X)
print("\n Name of Target Variable:", logit.endog_names)
print("\n Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton',maxiter=100,tol=1e-7,full_output=True)
thisParameter = thisFit.params 

print("\n Model Parameter Estimates:\n", thisFit.params)
print("\n Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

y_confusion = metrics.confusion_matrix(y, y_predictClass)
print("\n Confusion Matrix (Row is True, Column is Predicted) = \n")
print('\n',y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predictClass)
print("\n Accuracy Score = ", y_accuracy)

#Question i
new_d=pandas.concat([DataTrain[['group_size','homeowner','married_couple']], thisFit.predict(X)], axis = 1)
odd=pandas.DataFrame((new_d[1]/new_d[0]),columns =['A=1/A=0'])
new_d=pandas.concat([new_d,odd], axis = 1)
new_d.loc[new_d['A=1/A=0'] == new_d['A=1/A=0'].max()]
#Maximum Value
print("\n The maximum value of odd of Prob(A=1)/Prob(A=0) is :",new_d['A=1/A=0'].max())

## Define a function to visualize the percent of a particular target category by a nominal predictor
'''
def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pandas.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100

   print("Frequency Table: \n")
   print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

   return
'''
#
print('Solution:2f')
print("\n Contingency Table:")
con_data = pandas.crosstab(index= [data1.group_size, data1.homeowner, data1.married_couple], columns= data1.A,margins = True, dropna = True)
x = con_data.drop('All', 1)
print (con_data)
percentTable = con_data.div(x.sum(1), axis='index')*100
print("Percent Table: \n")
print(percentTable)
