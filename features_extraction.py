__author__ = 'Mohamed Ibrahim Ali'
from support_functions import *


if __name__ == "__main__":
    # Read Vehicles and non-Vehicles images pathes from hard using glob module
    vehicles = glob.glob('./vehicles/*/*.png')
    # dataset shuffling to reduce effect of time series images
    np.random.shuffle(vehicles)

    non_vehicles = glob.glob('./non-vehicles/*/*.png')

    # calculate some statistics of data set

    vehicles_size = len(vehicles)
    non_vehicles_size = len(non_vehicles)
    # print statistics about dataset
    print("vehicles dataset size =", vehicles_size, " and Non-vehicles dataset size =", non_vehicles_size)

    # features extraction parameters
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    t = time.time()
    car_features = extract_features(vehicles, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(non_vehicles, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # save it into pickle file for later uses
    joblib.dump(X_scaler, 'preprocessing_scaler.pkl')
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Save the dataset for later use
    dist_pickle = {"train_data": X_train, "train_label": y_train,"test_data": X_test, "test_label": y_test}
    pickle.dump(dist_pickle, open("dataset.p", "wb"))