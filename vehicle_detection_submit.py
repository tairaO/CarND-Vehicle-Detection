# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 06:34:37 2018

@author: taira
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import random
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
from IPython.display import HTML


#%%
# read image names
cars_dirlist    = os.listdir('./vehicles/vehicles')
noncars_dirlist = os.listdir('./non-vehicles/non-vehicles')

cars    = []
notcars = []

for dirname in cars_dirlist:
    tmp= glob.glob('./vehicles/vehicles/' + dirname + '/*.png')
    cars.extend(tmp)

for dirname in noncars_dirlist:
    tmp= glob.glob('./non-vehicles/non-vehicles/' + dirname + '/*.png')
    notcars.extend(tmp)

num_cars    = len(cars)
num_notcars = len(notcars)
print("Cars' sample number: " + str(num_cars))
print("NonCars' sample number: " + str(num_notcars))

#%%
# check images
i = random.randint(0, num_cars)
j = random.randint(0, num_notcars)

tmp_car = mpimg.imread(cars[i])
tmp_notcars = mpimg.imread(notcars[j])
fig = plt.figure()
plt.subplot(121)
plt.imshow(tmp_car)
plt.title('Car sample image')
plt.subplot(122)
plt.imshow(tmp_notcars)
plt.title('Noncar sample image')
fig.tight_layout()

#%%
# specify wach parameter's value

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations default 9
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions #checked
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
# Hog sub sampling
ystart = y_start_stop[0]
ystop = y_start_stop[1]

#%%
# see hog feature image of car
i = random.randint(0, num_cars)
j = random.randint(0, num_notcars)

tmp_car = mpimg.imread(cars[i])
tmp_notcar = mpimg.imread(notcars[j])
tmp_car_yuv = cv2.cvtColor(tmp_car, cv2.COLOR_RGB2YUV)
tmp_notcar_yuv = cv2.cvtColor(tmp_notcar, cv2.COLOR_RGB2YUV)

hog_car_images = []

for i in range(3):
    features, hog_image = get_hog_features(tmp_car_yuv[:,:,i], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
    hog_car_images.append(np.copy(hog_image))


hog_notcar_images = []

for i in range(3):
    features, hog_image = get_hog_features(tmp_notcar_yuv[:,:,i], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
    hog_notcar_images.append(np.copy(hog_image))    


fig = plt.figure()
plt.subplot(3,4,1)
plt.imshow(tmp_car_yuv[:,:,0], cmap='gray')
plt.title('Car ch1')
plt.subplot(3,4,2)
plt.imshow(hog_car_images[0], cmap='gray')
plt.title('Car ch1 Hog')
plt.subplot(3,4,3)
plt.imshow(tmp_notcar_yuv[:,:,0], cmap='gray')
plt.title('Notcar ch1')
plt.subplot(3,4,4)
plt.imshow(hog_notcar_images[0], cmap='gray')
plt.title('Notcar ch1 Hog')
plt.subplot(3,4,5)
plt.imshow(tmp_car_yuv[:,:,1], cmap='gray')
plt.title('Car ch2')
plt.subplot(3,4,6)
plt.imshow(hog_car_images[1], cmap='gray')
plt.title('Car ch2 Hog')
plt.subplot(3,4,7)
plt.imshow(tmp_notcar_yuv[:,:,1], cmap='gray')
plt.title('Notcar ch2')
plt.subplot(3,4,8)
plt.imshow(hog_notcar_images[1], cmap='gray')
plt.title('Notcar ch2 Hog')
plt.subplot(3,4,9)
plt.imshow(tmp_car_yuv[:,:,2], cmap='gray')
plt.title('Car ch3')
plt.subplot(3,4,10)
plt.imshow(hog_car_images[2], cmap='gray')
plt.title('Car ch3 Hog')
plt.subplot(3,4,11)
plt.imshow(tmp_notcar_yuv[:,:,2], cmap='gray')
plt.title('Notcar ch3')
plt.subplot(3,4,12)
plt.imshow(hog_notcar_images[2], cmap='gray')
plt.title('Notcar ch3 Hog')
fig.tight_layout()

#%%
# extract features and split data set to train and test set

t = time.time()
# extract features 
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)


print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

#%%

# Use linear SVC and search best parameter by means of grid search
parameters = {'C':[0.0001, 0.001, 0.01]}
#parameters = {'C':[0.01]}
svc = LinearSVC()
clf = GridSearchCV(svc, parameters)


# fit and check the training time for the SVC
t=time.time()
clf.fit(X_train,  y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 8))
print('Best parameter = ', clf.best_params_)


#%%
# save trained classifier
joblib.dump(clf, 'classifier.pkl')
joblib.dump(X_scaler, 'standard_scaler.pkl')

#%%%
#load trained classifier 
clf = joblib.load('classifier.pkl') 
X_scaler = joblib.load('standard_scaler.pkl')

#%%
# read sample image
image = cv2.imread('bbox-example-image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
draw_image = np.copy(image)


#%%
# test find_cars
scale = 2.0
ystart = 370
ystop  = 562

out_img, hot_windows = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, 
                                 pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                 color_space=color_space,cells_per_step = 1)

plt.imshow(out_img)

#%%
def multi_window_finds_car(image):
    hot_windows = []
    scales              = [0.5, 1.0, 1.25, 1.5, 2.0]
    ystarts             = [390, 385,  380, 390, 370]
    ystops              = [444, 481,  508, 562, 626]
    tmp_cells_per_steps = [  4,   1,    1,   2,   2]
    # TODO: sppropriate define_appropriate y_start and tmp_per_cell
    
    for scale, tmp_ystart, tmp_ystop, tmp_cells_per_step in zip(scales, ystarts, ystops, tmp_cells_per_steps):

        out_img, tmp_hot_windows = find_cars(image, tmp_ystart, tmp_ystop, scale, clf, X_scaler, orient, 
                                 pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                 color_space=color_space, cells_per_step = tmp_cells_per_step)
        hot_windows.extend(tmp_hot_windows)
        draw_img = draw_boxes(np.copy(image), hot_windows)
    return draw_img, hot_windows

#%%


t=time.time()
out_img, hot_windows = multi_window_finds_car(image)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to process 1 image...')
plt.imshow(out_img)
#%%
#see a converted image
plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.title('Car sample image')
plt.subplot(122)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:,:,2], cmap='gray')
plt.title('Converted sample image')
fig.tight_layout()


#%%
# draw heat map and 

for i in range(1,7):
    image = cv2.imread('./test_images/test' + str(i)+ '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_img, hot_windows = multi_window_finds_car(image)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    print(labels[1], 'cars found')
    
    fig = plt.figure()
    plt.imshow(out_img)
    plt.title('test' + str(i))
    fig.tight_layout()
    fig.savefig('./output_images/multi_hogsubsample_example_test'+str(i)+'.png')
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(out_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    fig.savefig('./output_images/heatmap_test'+str(i)+'.png')
    
    fig = plt.figure()
    plt.imshow(draw_img)
    plt.title('test' + str(i))
    fig.tight_layout()
    fig.savefig('./output_images/bbox_example_test'+str(i)+'.png')


#%%
plt.close('all')

#%%

class Vehicles():
    def __init__(self):
        # store last hot windows
        self.current_windows = []
        # store last hot windows
        self.current_centorids_size = []
        # last 3 centorids and those sizes 
        self.recent_centorids_size = []
        # store best estimate of centroids and those sizes
        self.best_estimated = []
        
        self.radius = 50
    
    def draw_best_boxes(self, img, labels):
        # Iterate through all detected cars
        self.current_centorids_size = []
        self.best_estimated = []
        
        
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            centroid = (int((bbox[0][0]+bbox[1][0])/2.0), int((bbox[0][1]+bbox[1][1])/2.0))
            size = (int((bbox[1][0] - bbox[0][0])), int((bbox[1][1]-bbox[0][1])))
            self.current_centorids_size.append([centroid, size])
            
            count = 0
            near_centroids = []
            near_sizes = []
            for past_centroids_size in self.recent_centorids_size:
                for past_centroid, past_size in past_centroids_size:
                    if (centroid[0]-past_centroid[0])**2 + (centroid[1]-past_centroid[1])**2 < self.radius**2:
                        count += 1
                        near_centroids.append(past_centroid)
                        near_sizes.append(past_size)
            
            if count >=3:
#            if count >=0:
                best_centroid_x = centroid[0]
                best_centroid_y = centroid[1]
                best_size_x = size[0]
                best_size_y = size[1]
                for near_centroid, near_size in zip(near_centroids, near_sizes):
                    best_centroid_x += near_centroid[0]
                    best_centroid_y += near_centroid[1]
                    best_size_x += near_size[0]
                    best_size_y += near_size[1]
                n = len(near_centroids)+1
                best_centroid_x /= n
                best_centroid_y /= n
                best_size_x /= n
                best_size_y /= n
                self.best_estimated.append([(best_centroid_x,best_centroid_y), (best_size_x,best_size_y)])
                best_bbox = ((int(best_centroid_x-best_size_x/2), int(best_centroid_y-best_size_y/2)), 
                             (int(best_centroid_x+best_size_x/2), int(best_centroid_y+best_size_y/2)))
                # Draw the box on the image
                cv2.rectangle(img, best_bbox[0], best_bbox[1], (0,0,255), 6)
                cv2.circle(img, (int(best_centroid_x), int(best_centroid_y)), 8,(255,0,0),-1)
#                cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
#                cv2.circle(img, centroid, 8,(255,0,0),-1)
#                cv2.circle(img, centroid, self.radius,(255,0,0),4)
        
        if len(self.recent_centorids_size) == 3:
            del self.recent_centorids_size[0]
        self.recent_centorids_size.append(self.current_centorids_size)
        
        return img
#%%
        

# function that process images in video
def process_image_average(image):
    
    
    out_img, hot_windows = multi_window_finds_car(image)
    vehicles.current_windows = hot_windows
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = vehicles.draw_best_boxes(np.copy(image), labels)
#    cv2.line(draw_img, (0, 360), (draw_img.shape[1], 360), (0,255,0))
#    cv2.line(draw_img, (0, 400), (draw_img.shape[1], 400), (0,255,0))
    return draw_img
#    return out_img
    

#%%
# process videos
vehicles = Vehicles()

output = "./output_images/processed_project_video.mp4"
#clip1 = VideoFileClip('project_video.mp4').subclip(21,23)
clip1 = VideoFileClip('project_video.mp4')
clip = clip1.fl_image(process_image_average) 
clip.write_videofile(output, audio=False)

#%%
clip1.reader.close()
clip1.audio.reader.close_proc()
clip.reader.close()
clip.audio.reader.close_proc()