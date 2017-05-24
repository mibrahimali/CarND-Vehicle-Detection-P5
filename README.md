# **Vehicle Detection**
---

[//]: # (Image References)

[image1]: ./output_images/test1_labeled_output "Final Labeled output of test image 1"
[image2]: ./output_images/test2_labeled_output "Final Labeled output of test image 2"
[image3]: ./output_images/test3_labeled_output "Final Labeled output of test image 3"
[image4]: ./output_images/test4_labeled_output "Final Labeled output of test image 4"
[image5]: ./output_images/test5_labeled_output "Final Labeled output of test image 5"
[image6]: ./output_images/test6_labeled_output "Final Labeled output of test image 6"
[image7]: ./test_images/test1.jpg "test image 1"
[image8]: ./test_images/test2.jpg "test image 2"
[image9]: ./readme_images/car_not_car.png "Dataset Exploration"
[image10]: ./readme_images/HOG_feature.png "HOG Features Extraction"
[image11]: ./output_images/test1_search_area "Window Search area"
[image12]: ./output_images/test1_classifier_output "Window Search area"
[image13]: ./output_images/test1_heat_map "Window Search area"
[image14]: ./output_images/heatmap_1.png "Heat map of test image 1"


Overview
---
Self-Driving Cars have three main pillars : Perception, planning and control. Perception Module is responsible for Environmental awareness such as object, lane marking and traffic signs detection and tracking

In this project, The main goal is to write a software pipeline to detect and track vehicles in a video using advanced Computer vision techniques, starting from suitable feature extraction and classifier training process ending with track and false positive detection removal on long sequence video segments.


| Input Image | Output Image 	| 
|:-------------:|:---------------------:| 
| ![alt text][image7]    	| ![alt text][image1]  			| 

The Project
---

The goals / steps of this project are the following:

* Performing feature extraction on a labeled training set of images and train a classifier SVM classifier
* Implementing a sliding-window technique and use your trained classifier to search for vehicles in images.
* create pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimating a bounding box for vehicles detected.

### **Dataset Exploration and Features Extraction**
---
Labeled images were taken from the GTI vehicle image database GTI, the KITTI vision benchmark suite, and examples extracted from the project video itself. All images are 64x64 pixels. A third data set released by Udacity was not used here. In total there are 8792 images of vehicles and 8968 images of non vehicles. Images of the GTI data set are taken from video sequences, to overcome this problem shuffling of dataset done before training the classifier. Shown below is an example of each class (vehicle, non-vehicle) of the data set. 

![alt text][image9]

Training a classifier to differentiate between cars and not cars pictures can be done using multiple techniques, one can use deep learning methods using only raw image data or using simple classifiers and instead of raw data we trained it using pre-extracted features from these raw data.

for the sake of this project, Histogram of Oriented Gradients (HOG) and Spatial color information features was extracted from raw image data and used as dataset for classifier training.
as an initial trial RGB color channels of images was used, but didn't give the best results. YUV color space was the best one.
after many trails for the best parameters of HOG features extracted from images, here is the final parameters

1- color space = 'YUV'  
2- HOG orientations = 12  
3- HOG pixels per cell = 8  
4- HOG cell per block = 2  
5- HOG channel = 0  # to reduce complexity of calculation

![alt text][image10]

And also to not depend on only one feature type,  image raw spatial information was taken part in feature vector used for training classifier by resizing image for 64x64 pix to 32x32.

Dataset Preparation code can be found in "features_extraction.py" file    
    
#### **Important Note**
Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

This was one of the main important steps in preparing dataset , SKlearn Standard Scaler was used to normalize our dataset before feeding it to training pipeline.

### **Classifier Design**
---
as a preparation step, I split Dataset into Training and testing sets with ratio of 80%-20%.
SVM classifier is used for car classification, after tuning trails, a SVM classifier with 'rpf' kernal and C value = 10.0 is used. classifier give 99.5% accuracy on test set.

Classifier design code can be found in "classifer_design.py" file  

### **Search Method Design**
---
our classifier was trained to differentiate between cars and non-cars images, it can't understand full front-camera image data. for this purpose a search policy should divide raw FC image into parts were our classifier can test this part if it's a car image. window search technique was used for this purpose. 
Multiple window sizes and overlapping ratio was tests, and the best outputs goes for this values

| Window Size 	| Overlap Ration in XY 	| Start and End position in Y dir | 
|:-------------:|:---------------------:|:---------------------:| 
| [64,64]    	| [0.75, 0.75]   	| [400, 500]	|
| [96,96]   	| [0.75, 0.75] 		| [400, 600]	|
| [128,128]	| [0.75, 0.75]		| [400, 650]	|

![alt text][image11]

this function is implemented in lines[105-150] in "support_functions.py" file and used in "vehicle_detector.py" file.

bellow is an image representing car-classified windows using this search strategy.
![alt text][image12]


### **Detection Fusion and False Positive Rejection**
---
using multiple scale windows with high overlap ration can results in many overlapping detection for the same object. these outputs should be combine together in order to predict overall size of object. also due to non-prefect classifier, some false positive detection can be found and must be rejected.

a heat-map was build from these detections in order to combine overlapping detections and remove false positives.

Also to make sure of false positive rejection, multi-frame accumulated heatmap is used by storing the heatmap of the last N frames (N can be 5 or 8) and do the same thresholding and labeling on average of these heatmaps.

'''
from collections import deque
history = deque(maxlen = 10)
history.append(current_heat_map)
'''

code for this part can be found in "support_functions.py" lines [279-283]


![alt text][image14] 

applying threshold on this heat map can reject false positive. threshold is set to 3 overlapping detection.

![alt text][image13]

### **Final Output**
---
final output of detection pipeline is threshold heat-map after applying label function to calculate final objects bounding boxes


Full Images and video process pipeline can be found in "Vehicle_detector.py" file

![alt text][image1]

Project video output can be found [here](http://www.cvlibs.net/datasets/kitti/). 

Potential shortcomings:
---
1- depending Hand crafted features from images can't insure overall generality of pipeline

2- Dataset size is relatively small, this can affect detection accuracy.


Possible improvements:
---
1- extracting mode training data from Udacity data set and using Data Augmentation techniques to improve overall classifier accuracy.  

2- Using Deep Learning Techniques To have a single pipeline takes Front Camera image and predict vehicles positions and bounding boxes.

3- Using Kalman Filters for measurement tracking to increase pipeline stability.