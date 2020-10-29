# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:51:40 2020

@author: Levente Országh
"""

#The chosen dataset: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

import itertools;
import numpy as np;
import urllib;
from matplotlib import pyplot as plt;
from sklearn import model_selection as ms;
from sklearn import linear_model as lm;
from sklearn import naive_bayes as nb;
from sklearn import metrics;

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt';
raw_data = urllib.request.urlopen(url);
data = np.loadtxt(raw_data, delimiter="\t");
X = data[:,0:3];
y = data[:,3];
target_names = ['skin','non-skin'];

X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.3, 
                                shuffle = True, random_state=2020);

logreg_classifier = lm.LogisticRegression();
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);
cm_logreg_train = metrics.confusion_matrix(y_train, ypred_logreg);
ypred_logreg = logreg_classifier.predict(X_test);
cm_logreg_test = metrics.confusion_matrix(y_test, ypred_logreg);
yprobab_logreg = logreg_classifier.predict_proba(X_test);

plt.figure(1);
plot_confusion_matrix(cm_logreg_train, classes=target_names,
    title='Confusion matrix for training dataset (logistic regression)');
plt.show();

plt.figure(2);
plot_confusion_matrix(cm_logreg_test, classes=target_names,
   title='Confusion matrix for test dataset (logistic regression)');
plt.show();

naive_bayes_classifier = nb.GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);
cm_naive_bayes_train = metrics.confusion_matrix(y_train, ypred_naive_bayes);
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);
cm_naive_bayes_test = metrics.confusion_matrix(y_test, ypred_naive_bayes);
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);

plt.figure(3);
plot_confusion_matrix(cm_naive_bayes_train, classes=target_names,
    title='Confusion matrix for training dataset (naive Bayes)');
plt.show();

plt.figure(4);
plot_confusion_matrix(cm_naive_bayes_test, classes=target_names,
   title='Confusion matrix for test dataset (naive Bayes)');
plt.show();

fpr_logreg, tpr_logreg, _ = metrics.roc_curve(y_test, yprobab_logreg[:,1], pos_label=1);
roc_auc_logreg = metrics.auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = metrics.roc_curve(y_test, yprobab_naive_bayes[:,1], pos_label=1);
roc_auc_naive_bayes = metrics.auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(5);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (area = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (area = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();