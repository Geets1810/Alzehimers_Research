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

#Suppose a market basket can possibly contain these seven items: A, B, C, D, E, F, and G.

#a)	(1 point) What is the number of possible itemsets?
n=7
print ('Total number of items',n)

print('Solution-(a)-Possible item set =', (pow(2,n)-1))


#b)	(3 points) List all the possible 1-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 1))

print ('The 1- itemsets are:')
print('\n')
print(p)
print('\n')

print ('Solution-(b)-The number of possible itemsets are',len(p))

#c)	(3 points) List all the possible 2-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 2))

print ('The 2- itemsets are:')
print('\n')
print(p)
print('\n')
print ('Solution-(c)-The number of possible itemsets are',len(p))

#d)	(3 points) List all the possible 3-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 3))

print ('The 3- itemsets are:')
print('\n')
print(p)
print('\n')
print ('Solution-(d)-The number of possible itemsets are',len(p))

#e)	(3 points) List all the possible 4-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 4))

print ('The 4- itemsets are:')
print('\n')
print(p)
print('\n')
print ('Solution-(e)-The number of possible itemsets are',len(p))

#f)	(3 points) List all the possible 5-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 5))

print ('The 5- itemsets are:')
print('\n')
print(p)
print('\n')
print ('Solution-(e)-The number of possible itemsets are',len(p))

#g)	(3 points) List all the possible 6-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 6))

print ('The 6- itemsets are:')
print('\n')
print(p)
print('\n')
print ('Solution-(e)-The number of possible itemsets are',len(p))

#e)	(3 points) List all the possible 7-itemsets.
p = list(itertools.combinations(['A','B','C','D','E','F','G'], 7))

print ('The 7- itemsets are:')
print('\n')
print(p)
print('\n')
print ('Solution-(e)-The number of possible itemsets are',len(p))
