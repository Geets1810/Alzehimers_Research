import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import sklearn.cluster as cluster
import math
Spiral= pandas.read_csv('C:\\Users\\Vanathi\\My scripts\\Assignments\\Assignment 2 blackboard\\Spiral.csv', 
                        delimiter=',', usecols=['x', 'y'])
nObs = Spiral.shape[0]

#a)	(5 points) Generate a scatterplot of y (vertical axis) versus x (horizontal axis).
#  How many clusters will you say by visual inspection?
print("\n")
print("Solution-4a-Scatter plot")
print("\n")
plt.scatter(Spiral[['x']], Spiral[['y']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

print("\n")
print("Solution-4b-Scatter plot after k-mean")
print("\n")
trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)
Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
'''
trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
'''
#c) (5 points) Apply the nearest neighbor algorithm using the Euclidean distance.
#  How many nearest neighbors will you use?
import sklearn.neighbors

print("\n")
print("Solution-4c-using 3 nearest neighbors")
print("\n")
# Three 
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 8, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')

distances = distObject.pairwise(trainData)

#d)	(5 points) Generate the sequence plot of the first nine eigenvalues, 
#starting from the smallest eigenvalues. 
# Based on this graph, do you think your number of nearest neighbors (in a) is appropriate?

Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

print("\n")
print("Solution-4d-using 3 nearest neighbors verification graph")
print("\n")
# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0,9,1), evals[0:9,],c="g")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

#e)	(5 points) Apply the K-mean algorithm on your first two eigenvectors that correspond
# to the first two smallest eigenvalues. 
#Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?
# Inspect the values of the selected eigenvectors 

Z = evecs[:,[0,1]]
print(Z[[0]].mean(), Z[[0]].std())
print(Z[[1]].mean(), Z[[1]].std())

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=0).fit(Z)

Spiral['SpectralCluster'] = kmeans_spectral.labels_
print("\n")
print("Solution-4e- scatterplot using the K-mean cluster identifier to control the color scheme")
print("\n")
plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['SpectralCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
