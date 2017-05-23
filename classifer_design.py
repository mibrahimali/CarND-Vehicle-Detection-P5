__author__ = 'Mohamed Ibrahim Ali'
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC , LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle

dataset = joblib.load("dataset.p")
X_train = dataset['train_data']
y_train = dataset['train_label']
X_test = dataset['test_data']
y_test = dataset['test_label']


# Use a linear SVC
svc = SVC(C=10.0, gamma='auto', kernel='rbf')
# svc = LinearSVC(C=10.0)
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


# save classifier for later uses
joblib.dump(svc, 'trained_svm_classifier_2.pkl')

"""
386.47 Seconds to train SVC...
Test Accuracy of SVC =  0.9955
My SVC predicts:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  0.]
For these 10 labels:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  0.]
0.314 Seconds to predict 10 labels with SVC
    """