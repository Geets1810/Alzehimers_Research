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
from itertools import combinations
import itertools
from tabulate import tabulate

# Define a function to visualize the percent of a particular target category by an interval predictor
def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   dataTable = inData
   dataTable['LE_Split'] = ((dataTable.iloc[:,0]).isin(split))

   crossTable = pandas.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * np.log2(proportion)
      print('Row = ', iRow, 'Entropy =', rowEntropy)
      print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
   return(tableEntropy)
   
data1 = pandas.read_csv('C:\\Users\\Vanathi\\My scripts\\Assignments\\Assignment 3 blackboard\\CustomerSurveyData.csv', 
                        delimiter=',')
data2 = data1[['CarOwnership']].dropna()
data31 = data1[['CreditCard']].fillna(value='Missing')
data32 = data1[['JobCategory']].fillna(value='Missing')
data_new1 = data2.join(data31)
data_new2 = data2.join(data32)

# Horizontal frequency bar chart of Cylinders
data_new1.groupby('CreditCard').size().plot(kind='barh')
# Horizontal frequency bar chart of Cylinders
data_new2.groupby('JobCategory').size().plot(kind='barh')
#print(data1)
#print(list(data1))
#print(data1['CreditCard']

crossTable = pandas.crosstab(index = data_new1['CreditCard'], columns = data_new1['CarOwnership'],
                             margins = True, dropna = True)   
print(crossTable)

p1 = int(crossTable.loc['All','Lease'])/int(crossTable.loc['All','All'])
p2 = int(crossTable.loc['All','None'])/int(crossTable.loc['All','All'])
p3 = int(crossTable.loc['All','Own'])/int(crossTable.loc['All','All'])
EV = -((p1*np.log2(p1))+(p2*np.log2(p2))+(p3*np.log2(p3)))
print('\n a)The Entropy metric is: ',EV)

#Cross-table for credit card
leftc = []
rightc = []
index_split = []
Creditcard = data_new1['CreditCard'].unique()

for i in range(1, math.ceil(len(Creditcard)/2)) :
    for x in combinations(Creditcard, i):
        leftc.append(x)
        index_split.append(len(x))
        for s in itertools.combinations(Creditcard,len(Creditcard)-i):
            if (not set(x)&set(s)):
                rightc.append(s)
            


res = []
for i in range(len(leftc)):
    inData1 = data_new1[['CreditCard', 'CarOwnership']].dropna()
    EV = EntropyIntervalSplit(inData1, leftc[i])
    res.append(EV)

print('Credit Card')
resultdf = pandas.DataFrame({'Index of Split':index_split, 'Left_Branch': leftc, 'Right_Branch':rightc, 'Entropy': res})

min_entropy = resultdf['Entropy'].min()
print(resultdf)
print('\n optimal split for the CreditCard predictor:',min_entropy)
#

crossTable = pandas.crosstab(index = data_new2['JobCategory'], columns = data_new1['CarOwnership'],
                             margins = True, dropna = True)   
print(crossTable)

#
p1 = int(crossTable.loc['All','Lease'])/int(crossTable.loc['All','All'])
p2 = int(crossTable.loc['All','None'])/int(crossTable.loc['All','All'])
p3 = int(crossTable.loc['All','Own'])/int(crossTable.loc['All','All'])
EV = -((p1*np.log2(p1))+(p2*np.log2(p2))+(p3*np.log2(p3)))
print('\n a) The Entropy metric is: ',EV)

# The cross table for Job Category

leftc = []
rightc = []
index_split = []
JobCategory = data_new2['JobCategory'].unique()

for i in range(1, math.ceil(len(JobCategory)/2)):
    for x in combinations(JobCategory, i):
        leftc.append(x)
        index_split.append(len(x))
        for s in itertools.combinations(JobCategory,len(JobCategory)-i):
            if (not set(x)&set(s)):
                rightc.append(s)
            


res = []
for i in range(len(leftc)):
    inData1 = data_new2[['JobCategory', 'CarOwnership']].fillna("Missing")
    EV = EntropyIntervalSplit(inData1, leftc[i])
    res.append(EV)

print('Job Category')
resultdf1 = pandas.DataFrame({'Index of Split':index_split, 'Left_Branch': leftc, 'Right_Branch':rightc, 'Entropy': res})

min_entropy = resultdf1['Entropy'].min()
print(resultdf1)
print('\n optimal split for the JobCategory predictor:',min_entropy)