__author__ = 'Mohamed Ibrahim Ali'
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from skimage.feature import hog
import pickle
from support_functions import *
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip




# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = mpimg.imread('test_images/test6.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255
heat = np.zeros_like(image[:,:,0]).astype(np.float)

colorspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
y_start_stop = [400, 656]  # Min and max in y to search in slide_window()

X_scaler = joblib.load('preprocessing_scaler.pkl')
clf = joblib.load('trained_svm_classifier_2.pkl')

def process_video(image):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400,592],
                        xy_window=(96, 96), xy_overlap=(0.6, 0.6))

    # windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400,464],
    #                     xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    #
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[528,656],
                        xy_window=(128, 128), xy_overlap=(0.5, 0.5))


    # windows_4 = slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
    #                     xy_window=(256, 256), xy_overlap=(0.8, 0.8))

    hot_windows = search_windows(image, windows, clf, X_scaler, color_space=colorspace,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel)

    # search_area = draw_boxes(image, windows_3,color=(0,0,255),thick=7)
    # search_area = draw_boxes(search_area, windows_2,color=(0,255,0),thick=5)
    # search_area = draw_boxes(search_area, windows_1,color=(255,0,0),thick=3)
    # plt.figure()
    # plt.imshow(search_area)
    #
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    # plt.figure()
    # plt.imshow(window_img)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    #
    # plt.show()
    return draw_img


challenge_output = 'test_video_out.mp4'
clip2 = VideoFileClip('test_video.mp4')
challenge_clip = clip2.fl_image(process_video)
challenge_clip.write_videofile(challenge_output, audio=False)


# challenge_output = 'project_video_out.mp4'
# clip2 = VideoFileClip('project_video.mp4')
# challenge_clip = clip2.fl_image(process_video)
# challenge_clip.write_videofile(challenge_output, audio=False)
