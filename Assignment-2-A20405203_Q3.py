
#Load the PANDAS library
import pandas
import math
import numpy 
from numpy import ma
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy
from scipy import linalg as LA2
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

# Read the data into dataframe
trainData= pandas.read_csv('C:\\Users\\Vanathi\\My scripts\\Assignments\\Assignment 2 blackboard\\cars.csv',
                       delimiter=',', usecols=['Horsepower', 'Weight'])

nCars = trainData.shape[0]

# a)	(15 points) List the Elbow values
# and the Silhouette values for your 1-cluster to 15-cluster solutions. 
'''''
nClusters = numpy.zeros(15)

TotalWCSS = numpy.zeros(15)
Inertia = numpy.zeros(15)
'''''
#Reference:https://pythonprogramminglanguage.com/kmeans-elbow-method/
#Reference:https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

x1 = numpy.array(trainData["Horsepower"])
x2 = numpy.array(trainData["Weight"])
X = numpy.array(list(zip(x1, x2))).reshape(len(x1), 2)

# k means determine k
meandistortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    meandistortions.append(sum(numpy.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


K = range(2,16)
score = []
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    preds = kmeanModel.fit_predict(X)
    score.append(metrics.silhouette_score(X, preds, metric='euclidean'))  

print("Solution-3a-List the Elbow values and the Silhouette values for your 1-cluster to 15-cluster")
print("\n")
print("Elbow-Values", meandistortions)
print("\n")
print("Silhouette-Score", score)
print("\n")
print("Solution-3b-Number of clusters to be suggested from below graphs is 4")
print("\n")
#Plot the Silhouette
plt.plot(K, score, 'bx-',c="g")
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('The Silhouette Method showing the optimal k')
plt.show()
print("\n")
# Plot the elbow
plt.plot(K, meandistortions, 'bx-',c="g")
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()



