## Advanced Lane Finding Project
### Writeup, J-M Tirilä

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[perspective_calibration_image]: ./calibration_images/perspective_calibration_image.png "Perspective calibration image"
[undistorted_calibration_image]: ./calibration_images/undistorted_calibration_image.png "Undistorted calibration image"
[perspective_transformed_image]: ./calibration_images/perspective_transformed_image.png "Undistorted calibration image"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Overall notes on code organization

For this project, I decided to use a traditional Python project structure, with bits of functionality separated into 
respective packages and modules. The outermost `carnd_advanced_lane_detection` directory functions as a container for 
all the assets, written reports and the code. 

The code lives under the innermost `carnd_advanced_lane_detection` 
directory. **When referring to code files, I used the root of this code directory as the starting point to avoid 
excessively long paths.**

The source file orhestrating the lane detection procedure is `detect_lanes.py`. It can be run as a script, or 
one can import and run its detect_lanes function by
```python
from carnd_advanced_lane_detection.detect_lanes import detect_lanes
detect_lanes()
```

This latter approach is primarily useful if one wishes to also import some of the processing functions and run them 
using different parameters.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the source file `carnd_advanced_lane_detection/preparations/calibrate_camera.py` 
The functionality has been divided into three methods: `calibrate_camera`, `_find_chessboard_corners`, and 
`_find_imgpoints`. The primary function to call is `calibrate_camera`, and the others are auxiliary functions 
to perform specific tasks. 

I start by converting the calibration images into grayscale images. Subsequently, I preparing "object points", 
which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is 
fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objpoints` is just a replicated array of coordinates, and `objp` will be appended with a copy of it every 
time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the `(x, y)` 
pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients 
using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using 
the `cv2.undistort()` function and obtained this result: 

![Undistorted calibration image][undistorted_calibration_image]

### Single image processing 

Next, I will outline the steps applied to process a _single image_. At this point, it is assumed that the image 
has already been distortion corrected. In the actual video processing pipeline, distortion correction will be taken 
care of as the first step before proceeding to the steps below. 

#### Perspective transform 

For the perspective transform, I simply chose the source manually by visual inspection from a suitable calibration 
image, presented below. 

![Perspective calibration image][perspective_calibration_image]

The heavy lifting of performing the perspective transform was carried out using OpenCV's `getPerspectiveTransform` 
as per the lecture examples. Here is an example of applying the transform to the frame above:  

![Perspective transformed image][perspective_transformed_image]


##### The source and destination rectangles
##### Applying the transform
##### The inverse transform

#### Extracting the s channel

#### Applying a mask to only include lane pixels

#### Identifying left and right lane pixels

#### Fitting the polynomial
##### Computing the curvature

#### Visualizing the fitted polynomials and the lane area between them

### Putting it all together: 

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

The viedo processing pipeline can be found in the `detect_lanes.py` file. Besides the parts of the file already 
discussed, the main workhorse for the actual video processing part are the `detect_lanes` and `_process_video` 
functions.

For the video processing, as seen in the `_process_video` function (lines FIXME), I used the `moviepy` library and 
its `moviepy.editor.VideoFileClip` class. 

Reading the video using this method is rather straightworward, as is passing each invidivual frame to be processed 
further. Essentially, the `fl_image` method of said class takes care of passing a frame to be processed 
by the function provided as input for `fl_image`.

The related code can be found in the `detect_lanes.py` file and the relevant bit is reproduced below:

```python
clip = VideoFileClip(video_path)
transformed_clip = clip.fl_image(lambda image: _process_image(image, mtx, dist))
transformed_clip.write_videofile(output_path, audio=False)
```

As I need the distortion parameters `mtx` and `dist` for the `_process_image` function, I have wrapped the 
function call in a lambda function so that fl_image can just go on and provide each frame as input, without 
considering the other parameters.

#### The resulting video


FIXME: 
```
Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic 
failures that would cause the car to drive off the road!).
```

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

FIXME: 
```
Briefly discuss any problems / issues you faced in your implementation of this project.  
Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, 
where the pipeline might fail and how I might improve it if I were going to pursue this 
project further.  
```
