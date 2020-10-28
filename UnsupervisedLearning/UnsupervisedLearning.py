# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:52:08 2020

@author: Levente Országh
"""

#The chosen dataset: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

import numpy as np;
import urllib;
from matplotlib import pyplot as plt;
from sklearn import cluster;
from sklearn import metrics;

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt';
raw_data = urllib.request.urlopen(url);
data = np.loadtxt(raw_data, delimiter="\t");
X = data[:,0:3];
y = data[:,3];

fig = plt.figure(1);
plt.title('Scatterplot of datapoints with labels');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=y);
plt.show();

K = 7;
kmeans_cluster = cluster.KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);
ypred = kmeans_cluster.predict(X);
sse = kmeans_cluster.inertia_;
centers = kmeans_cluster.cluster_centers_;

fig = plt.figure(2);
plt.title('Scatterplot of datapoints with clusters');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=ypred);
plt.scatter(centers[:,0],centers[:,1],s=50,c='red');
plt.show();

Max_K = 31;
SSE = np.zeros((Max_K-2));
DB = np.zeros((Max_K-2));
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = cluster.KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(X);
    ypred = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = metrics.davies_bouldin_score(X,ypred);
    
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

K = 4;
kmeans_cluster = cluster.KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);
ypred = kmeans_cluster.predict(X);
centers = kmeans_cluster.cluster_centers_;

fig = plt.figure(5);
plt.title('Scatterplot of datapoints with 4 clusters');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=ypred);
plt.scatter(centers[:,0],centers[:,1],s=50,c='red');
plt.show();

K = 6;
kmeans_cluster = cluster.KMeans(n_clusters=K, random_state=2020);
kmeans_cluster.fit(X);
ypred = kmeans_cluster.predict(X);
centers = kmeans_cluster.cluster_centers_;

fig = plt.figure(6);
plt.title('Scatterplot of datapoints with 6 clusters');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=ypred);
plt.scatter(centers[:,0],centers[:,1],s=50,c='red');
plt.show();