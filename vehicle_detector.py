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


color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = False  # Histogram features on or off
hog_feat = True  # HOG features on or off

X_scaler = joblib.load('preprocessing_scaler.pkl')
svc = joblib.load('trained_svm_classifier.pkl')

def process_video(image):
    image = image.astype(np.float32) / 255
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400,500],
                        xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400,500],
                        xy_window=(96, 96), xy_overlap=(0.75, 0.75))
    #
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400,600],
                        xy_window=(128, 128), xy_overlap=(0.75, 0.75))


    # windows_4 = slide_window(image, x_start_stop=[None, None], y_start_stop=[300, None],
    #                     xy_window=(256, 256), xy_overlap=(0.8, 0.8))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)


    # search_area = draw_boxes(search_area, windows,color=(255,0,0),thick=5)
    # plt.figure()
    # plt.imshow(search_area)
    #
    # draw_image = np.copy(image)
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    # return window_img
    # plt.figure()
    # plt.imshow(window_img)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 0)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # return np.dstack((heatmap, heatmap, heatmap))
    # return heatmap
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image)*255, labels)

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

if __name__ == "__main__":

    # images_path = glob.glob('./test_images/*.jpg')
    # for image_path in images_path:
    #     image = mpimg.imread(image_path)
    #     image = image.astype(np.float32)/255
    #     output_image = process_video(image)
    #
    #     plt.figure()
    #     plt.imshow(output_image)
    # plt.show()


    # challenge_output = 'test_video_out.mp4'
    # clip2 = VideoFileClip('test_video.mp4')
    # challenge_clip = clip2.fl_image(process_video)
    # challenge_clip.write_videofile(challenge_output, audio=False)


    challenge_output = 'project_video_out.mp4'
    clip2 = VideoFileClip('project_video.mp4')
    challenge_clip = clip2.fl_image(process_video)
    challenge_clip.write_videofile(challenge_output, audio=False)
