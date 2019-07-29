# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:55:05 2019

@author: Vanathi
"""
# Load the PANDAS library
import pandas
import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import combinations

#a)	(1 point) What is the number of possible itemsets?
n=7
print ('Total number of items', n)

print('Solution-(a)-Possible item set =', (pow(2,n)-1))


#b)	(3 points) List all the possible 1-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 1))

print ('The 1- itemsets are', p)
print ('Solution-(b)-The number of possible itemsets are',len(p))

#c)	(3 points) List all the possible 2-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 2))

print ('The 2- itemsets are', p)
print ('Solution-(c)-The number of possible itemsets are',len(p))