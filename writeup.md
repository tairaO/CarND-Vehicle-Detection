## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/multi_hogsubsample_example_test1.png
[image5]: ./output_images/multi_hogsubsample_example_test2.png
[image6]: ./output_images/multi_hogsubsample_example_test3.png
[image7]: ./output_images/multi_hogsubsample_example_test4.png
[image8]: ./output_images/heatmap_test1.png
[image9]: ./output_images/heatmap_test2.png
[image10]: ./output_images/heatmap_test3.png
[image11]: ./output_images/heatmap_test4.png
[image12]: ./output_images/heatmap_test5.png
[image13]: ./output_images/heatmap_test6.png
[image14]: ./output_images/bbox_example_test4.png
[video1]: ./output_images/processed_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 31 through 322 of the file called `vehicle_detection_submit.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=16` and `cells_per_block=2`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Here is how I tuned them. I want to prevent my classifier from overfitting therefore I chose the parameters so that features vector can be as small as possible. However, if they are too small, features cannot have enough information to classify. Therefore, I saw the images of different parameters, and I chose the parameters that I slightly can distinguish car from non-car. Also, I finally chose YUV for my color space. I thought HSV and HLS are not good for this project because H space is not continuous. The range of H space is 0-180. Thus, 180 and 0 in H space is almost same in color but much different numerically. I tried YUV, LUV and YCrCb, and chose YUV.
For those reason, I finally settled parameters below:


```
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
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `GridSearchCV()` to tune C parameter. C parameter is the penalty for error term. I tried 0.0001, 0.001 and 0.01 as C parameter. `GridSearchCV()` returned that 0.001 was the best and its accuracy was 0.9903. I thought it was enough. I used 8792 car image and 8968 non-car image and those flipped images as data set (`extract_features()` in `lesson_functions.py`). I also split data set to train set and test set using `train_test_split()`, and its test_size was 0.2. After extracting features, I normalize each feature vectors using `StandardScaler()`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions so that my detector can search as little times as possible. For example, the small car appears near the horizon, so I search small size window only near the horizon. So, I saw each window size and chose appropriate y position in image for each window size. Here is my last windows.:

![alt text][image3]

I used Hog subsampling technique to minimize search time.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched five scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. To optimize classifier, I tried several color space, and optimize position of the windows. For example, I first chose same `ystrat` parameter for all space. However, it works not well. So I changed `ystart` and `ystop`. Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/processed_project_video.mp4)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The threshold is 2 so that I can remove false-positive. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video

Also, I used `Vehicle` class to store cars' centroids and window sizes in last 3 frames. If an detection was not detected in the circle of radius 60 pixel in all last three frame, that probably seems false-positive and doesn't draw bbox in video. Also, I averaged last three frames centroid positions and bbox size.
:

### Here are six frames and their corresponding heatmaps:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image14]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. I refer two problems in my implementation. First, my algorithm detects cars only if it appears in continuous three frame. So, my algorithm has 0.3 second delay for detecting car. Thinking about car accident, it is really long time. In order to improve this, I can use the max value of heatmap. If it is high, the certainty that it is car may be also high. Therefore, even if it is not appeared in last frame, it should be used as car detection. Also, my algorithm can be fail, when cars on an tilted road. This is because my windows positions are fixed, so it cannot deal with the potential position change where cars appear. In order to improve this problem, I can use the IMU's information, and change the window positions.
