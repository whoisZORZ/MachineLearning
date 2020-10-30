# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:59:38 2020

@author: Levente Országh
"""

#The chosen dataset: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

import numpy as np;
import urllib;
from matplotlib import pyplot as plt;
from sklearn import model_selection as ms;
from sklearn import decomposition as decomp;

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt';
raw_data = urllib.request.urlopen(url);
data = np.loadtxt(raw_data, delimiter="\t");
X = data[:,0:3];
y = data[:,3];

X_train, X_test, y_train, y_test = ms.train_test_split(X, 
             y, test_size=0.3, random_state=2020);

pca = decomp.PCA();
pca.fit(X_train);

fig = plt.figure(2);
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio))+1;
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 

fig = plt.figure(3);
plt.title('Cumulative explained variance ratio plot');
cum_var_ratio = np.cumsum(var_ratio);
x_pos = np.arange(len(cum_var_ratio))+1;
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,cum_var_ratio, align='center', alpha=0.5);
plt.show(); 

PC_train = pca.transform(X_train);
fig = plt.figure(4);
plt.title('Scatterplot for training dataset');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_train[:,0],PC_train[:,1],s=50,c=y_train,cmap = 'tab10');
plt.show();

PC_test = pca.transform(X_test);
fig = plt.figure(5);
plt.title('Scatterplot for test dataset');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_test[:,0],PC_test[:,1],s=50,c=y_test,cmap = 'tab10');
plt.show();