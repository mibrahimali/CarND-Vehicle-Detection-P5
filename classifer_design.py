__author__ = 'Mohamed Ibrahim Ali'
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle

dataset = pickle.load(open("dataset.p",'rb'))
X_train = dataset['train_data']
y_train = dataset['train_label']
X_test = dataset['test_data']
y_test = dataset['test_label']


# Use a linear SVC
svc = SVC(C=10.0, gamma='auto', kernel='rbf')

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
joblib.dump(svc, 'trained_svm_classifier.pkl')

"""
vehicles dataset size = 8792  and Non-vehicles dataset size = 8968
68.55 Seconds to extract HOG features...
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 7056
1881.73 Seconds to train SVC...
Test Accuracy of SVC =  0.9924
My SVC predicts:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
For these 10 labels:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
0.35199 Seconds to predict 10 labels with SVC
My SVC predicts:  [[  9.99904450e-01   9.55498679e-05]
 [  9.99990127e-01   9.87309506e-06]
 [  3.00000090e-14   1.00000000e+00]
 [  9.98796542e-01   1.20345753e-03]
 [  9.99903221e-01   9.67794164e-05]
 [  9.96015284e-01   3.98471579e-03]
 [  2.51729295e-12   1.00000000e+00]
 [  1.02171862e-09   9.99999999e-01]
 [  9.99785036e-01   2.14963663e-04]
 [  1.62964774e-01   8.37035226e-01]]
For these 10 labels:  [ 0.  0.  1.  0.  0.  0.  1.  1.  0.  1.]
0.24683 Seconds to predict 10 labels with SVC
    """