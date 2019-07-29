# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:44:43 2019

@author: Geethanjali
"""
# Load the PANDAS library
import pandas
import math
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy
from scipy import linalg as LA2
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import iqr
import matplotlib.pyplot as plt
# Read the data into dataframe
data1= pandas.read_csv('C:\\Users\\Vanathi\\My scripts\\Assignments\\Assignment 2 blackboard\\Groceries.csv',
                            delimiter=',', usecols=['Customer', 'Item'])


# Create frequency 
nItemPerCustomer = data1.groupby(['Customer'])['Item'].count()

# Convert the Sale Receipt data to the Item List format
#a)	(2 points) How many customers in this market basket data?
ListItem = data1.groupby(['Customer'])['Item'].apply(list).values.tolist()
nCustomer = len(ListItem)
print("Solution-(a)-Customers in this market basket data = ", nCustomer)

#b)(2 points) How many unique items in the market basket across all customers?
print("Solution-(b)-Unique items in the market basket across all customers = ", len(set(data1['Item'])))

#c)(5 points) Create a dataset which contains the number of distinct items in each customer’s market basket. 
#Draw a histogram of the number of unique items.
# What are the median, the 25th percentile and the 75th percentile in this histogram?

freqTable = pandas.value_counts(nItemPerCustomer).reset_index()
freqTable.columns = ['Item', 'Frequency']
freqTable = freqTable.sort_values(by = ['Item'])
print(freqTable)
nItemPerCustomer.describe()
#print('IQR is:',iqr(nItemPerCustomer))
#h = (2*iqr(nItemPerCustomer))/np.cbrt(nItemPerCustomer.size)

#u = np.log10(h)

#v = np.sign(u) * np.ceil(np.abs(u))

#print('1a. Recommended bandwidth:', h)
print("Solution-(c)")
print("The median is =", np.percentile(nItemPerCustomer, 50))
print("The 25th percentile is =", np.percentile(nItemPerCustomer, 25))
print("The 75th percentile is =", np.percentile(nItemPerCustomer, 75))
plt.hist(nItemPerCustomer,bins = 32)
plt.xlabel("Number of unique Items")
plt.ylabel("Frequency")
plt.show()

#d)	(5 points) Find out the k-itemsets which appeared in the market baskets of at least seventy five (75) customers.
#  How many itemsets have you found?  Also, what is the highest k value in your itemsets?
# Convert the Sale Receipt data to the Item List format

ListItem = data1.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, (75/9835), max_len = 32, use_colnames = True)
#print("frequent_itemsets=", frequent_itemsets)
from tabulate import tabulate
import pandas as pd
print("frequent_itemsets=",tabulate(frequent_itemsets, headers='keys', tablefmt='psql'))
print("Soultion-(d)-Highest k-value is=4")
#print(ItemIndicator)
#e)	(5 points) Find out the association rules whose Confidence metrics are at least 1%.  
#How many association rules have you found? 
# Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.
# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("Solution-(e)- Number of association rules are=", len(assoc_rules))

#Graph the Support metrics on the vertical axis against the Confidence metrics on 
#the horizontal axis for the rules you found in (e). 
# Please use the Lift metrics to indicate the size of the marker. 

plt.figure(figsize=(6,4))
print("Solution-(f)-Graph")
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'],c="g")
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")

#g)	(5 points) List the rules whose Confidence metrics are at least 60%.  
#Please include their Support and Lift metrics.
# Find the frequent itemsets
#frequent_itemsets = apriori(ItemIndicator, min_support = 0.1, max_len = 7, use_colnames = True)

# Discover the association rules
assoc_rules1 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
assoc_rules1
print("Solution-(g)-List the rules whose Confidence metrics are at least 60%.=",tabulate(assoc_rules1, headers='keys', tablefmt='psql'))

