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

[perspective_calibration_image]: images/perspective_calibration_image.png "Perspective calibration image"
[undistorted_calibration_image]: images/undistorted_calibration_image.png "Undistorted calibration image"
[perspective_transformed_image]: images/perspective_transformed_image.png "Undistorted calibration image"
[non_bright_norm_threshs]: images/non_brightess_normalized_different_thresholds.png "Non brightness normalized, different thresholds"
[equalized_s_channel_image]: images/saturation_normalization.png "Equalized s channel image"
[brightness_normalization]: images/brightness_normalization.png "Brightness normalization"
[mask_comparison_video]: mask_comparison.mp4 "Mask comparison video"
[final_video]: transformed.mp4 "Transformed final video"

## Overall notes on code organization

For this project, I decided to use a traditional Python project structure, with bits of functionality separated into 
respective packages and modules. The outermost `carnd_advanced_lane_detection` directory functions as a container for 
all the assets, written reports and the code. 

The code lives under the innermost `carnd_advanced_lane_detection` 
directory. **When referring to code files, I used the root of this code directory as the starting point to avoid 
excessively long paths.** So, for example, when talking about the file 
`carnd_advanced_lane_detection.detect_lanes.py`, I will simply refer to it as `detect_lanes.py`.

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


#### Luminosity normalization

As a preliminary normalization step, I used a trick to equalize the luminosity histogram of the image. This is 
performed by means of the following piece of code: 

```python
def normalize_brightness(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
``` 

The results of this transformation can ben seen in the figure below. 

![Brightness normalized image][brightness_normalization]

#### Perspective transform 

For the perspective transform, I simply chose the source manually by visual inspection from a suitable calibration 
image, presented below. 

![Perspective calibration image][perspective_calibration_image]

The heavy lifting of performing the perspective transform was carried out using OpenCV's `getPerspectiveTransform` 
as per the lecture examples. All the perspective transform related code can be found in the 
`image_transformations/perspective_transform.py` file. Much of the code is just a wrapper around 
 OpenCV functions with some convenience default values. 



Here is an example of applying the transform to the frame above:  

![Perspective transformed image][perspective_transformed_image]

##### The source and destination rectangles

The source and destination rectangles are defined by
```python
ROAD_SRC = np.float32([[250, 720], [589, 463], [701, 463], [1030, 720]])
ROAD_DST = np.float32([[250, 720], [250, 0], [1036, 0], [1036, 720]])
```

##### Applying the transform

The actual transform is carried out by OpenCV, through the `warpPerspective` function along with 
`getPerspectiveTransform`. This was all covered in the lecture notes so not repeating here. 

##### The inverse transform

As the detected lines need to be mapped back to the camera perspective, the code also provides a way to 
easily apply the inverse perspective transform. This is done by just appending the `inverse=True` parameter to 
the `road_perspective_transform` function. 


#### Applying a mask to only include lane pixels

After experimenting with different kinds of mask combinations, I ended up using a rather simple mask for this project: 
The mask consist of an aggregate of a saturation mask and gradient magnitude mask. 

##### Extracting the s channel

For the saturation mask, the s channel extraction war performed exactly as per the lecture notes, first performing a 
conversion to the HLS color space and there just selecting the S channel. 

Before applying the saturation mask, I also experimented with various tricks related to normalizing the s channel 
image before masking. For example, I tried using the 
`CLAHE` (for Contrast Limited Adaptive Histogram Equalization) class of OpenCV to have more uniform s value 
spread to the 0 - 255 interval, yet keeping contrast. and hence to be able to choose a more consistent threshold value.

To illustrate the effect of this normalization, here is a figure containing a non-equalized s_channel image and its  
histogram equalized counterpart: 


![Equalized s channel image][equalized_s_channel_image]

Even though the result may not seem much like an improvement, the main benefit is that after the normalization 
throughout the video, the saturation is distributed much more evenly across frames and I was able to choose a much 
higher and consistent saturation threshold than would otherwise have been possible. 

However, in hindsight, this s channel histogram equalization may not have been that useful. Much more benefit 
seems to be gained from the brightness normalization on the original image. Below are a couple of attempts
 at finding a suitable s channel threshold without applying any colorspace normalization to the original images. 
 
Here is the code that performs the brightness normalization: 
 
```python
def normalize_brightness(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
``` 
 
Even though the general shape of the lane line is indeed captured for these easy cases, it is much more difficult
to find the suitable threshold even for a single image, and the effect is even more dramatic when attempting to 
find a suitable global threshold for all the video frames. 

![Looking for a suitable threshold for non-brightness normalized images][non_bright_norm_threshs]

Also, below is a link to a video illustrating some attempts at coming up with a good mask. Even though the 
methods to produce the top and bottom rows in the image are different, the results are almost identical. 

The final implementation can be seen at `masks/combined_masks.py`, the function named `submission_combined`. The method 
code is reproduced below: 

```python
def submission_combined(image):
    s_image = rgb_to_s_channel(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(s_image)
    saturation_masked = saturation_mask(equalized, (220, 255))
    grad_mag_thresholded = mag_thresh(image, 5, (30, 255))
    combined = np.zeros_like(saturation_masked)
    combined[(saturation_masked == 1) & (grad_mag_thresholded == 1)] = 1
    return combined
```

Finally, I prepared a [video][mask_comparison_video] I used to compare the varying preprocessing steps masks, 
trying to identify a suitable one. 

#### Identifying left and right lane pixels, Fitting the polynomial and Computing the curvature

For the left and right lane pixels and the subsequent polynomial fitting and curvature computation  
I basically used the histogram based moving window search directly from the lecture instructions. 

The lane pixel detection related code can be found at `models/line.py`. There are a couple of static methods,
not implemented as instance mainly due to performance reasons: it is more efficient to iterate through an image
for both a left and right line simultaneously, so this is not performed per instance (left & right) but rather 
in one pass. 

The first of the two methods is for the case where search for lane pixels needs to be performed from scratch using
the moving window method from the instruction notes. This code is in a function called 
`find_lane_lines` (lines 122 - 190 of `line.py`). 

The other method performs similar search based on a previous polynomial fit. This code can be found at 
 `detect_line_pixels_based_on_previous_fit` (lines 71 - 95 of `line.py`). 

As for the tracking part, I used a scheme simplified a bit from the project hints. The processing could be more 
sophisticated, but I found out this was enough for my needs. So the process is as follows: 

 * Initially, track the lane lines using the histogram and moving window method window from the lecture
 * Fit a polynomial
 * Subsequently, if there has been a recent polynomial fit, just search for new line pixels around the previous 
   polynomial curve
 * For any given frame, compute the polynomial coefficients as a weighted average of the 15 latest polynomial fits.   
   The weighting scheme makes sure that recent fits are given more emphasis, but should there be frames with missing 
   (raw) polynomials, some kind of an estimate can still be computed as long as not all the previous 15 observations
   have not been faulty. In reality, with this masking and video processing, all the 15 previous sets of 
   coefficients are systematically there, however. 
 
 
 The code that performs the weighted averaging and shifting the recent coeffs array is reproduced here: 
 
 ```python
    WEIGHTS = np.array([1/2, 1/2, 1/2, 1/3, 1/4, 1/4, 1/5, 1/5, 1/6, 1/6, 1/7, 1/7, 1/8, 1/10, 1/10])
    ...
    def _append_last_coefficients(self, coeffs):
        self.recent_coeffs[-1] = coeffs
        self.recent_coeffs = np.roll(self.recent_coeffs, 1)
 
    ... 
    def get_smoothed_coeffs(self):
    
        idx = self.recent_coeffs != np.array(None)
        real_weights = Line.WEIGHTS[idx]
        scaled_real_weights = real_weights / np.sum(real_weights)
        return np.sum(self.recent_coeffs[idx] * scaled_real_weights, axis=0)


```
 
No further tracking of the coefficients is performed. I figured the averaging would also smooth the curvature and 
camera position computations enough. 

As for display of curvature and camera position, I used 

#### Visualizing the fitted polynomials and the lane area between them

For the line visualization part, I pretty much used the code as it was provided in the lectures. 
The relevant bits can be found at `utils/visualize_images.py`, the `return_superimposed_polyfits` function is the one 
that adds the green lane area to the image. 

### Putting it all together: 

#### Pipeline (video)

The video processing pipeline can be found in the `detect_lanes.py` file. Besides the parts of the file already 
discussed, the main workhorse for the actual video processing part are the `detect_lanes` and `_process_video` 
functions.

For the video processing, as seen in the `process_video` function (lines 98 - 111), I used the `moviepy` library and 
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
considering the other parameters. This code is reproduced below: 

```python
    transformed_clip = clip.fl_image(lambda image: _process_image(image, mtx, dist, left, right))
```

##### Computing camera position 

```python
# line.py
def compute_line_position_at_bottom(self):
    coeffs = self.get_smoothed_coeffs()
    pos = coeffs[0] ** 720**2 + coeffs[1] * 720 + coeffs[2]
    return pos

...

# detect_lanes.py
def _compute_camera_offset(left_line, right_line):
    return XM_PER_PIX * (0.5 * (left_line.compute_line_position_at_bottom() + right_line.compute_line_position_at_bottom()) - 640)
```

##### Displaying camera offset and curvature on the image

For superimposing text upon an image, I used OpenCV's `putText` function as follows: 

```python
# detect_lanes.py
cv2.putText(result, "Curvature: {:.2f} m".format(avg_curverad), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(result, "Camera offset from lane center: {:.2f} m {}".format(
    np.absolute(offset),
    offset_dir), (100, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
```

### The result

Here's a [link to my video result][final_video]

---

### Discussion

The project was interesting and some of the work I performed on color space normalizations provided 
me with a lot of insight into various normalization techniques. 

What I need work a little bit more on is various gradient masking techniques. Trying to combine them with my 
saturation masks was not very successful, and I was not quite able to obtain a good grasp of what kinds of 
gradient based techniques to use and when. This is something were my pipeline could likely 
falter: when the image saturation information is insufficient to reliably detect the lane lines, 
gradient based aspects of the mask would probably be valuable, and I did not quite nail it yet. 

Also I am aware that my line tracking is kind of rudimentary and I could have done more to 
detect anomalous observations using the techniques suggested in the project instructions. However,  
I felt my code was robust enough for this particular video so no further fine tuning was needed. 

Another point is that I needed to go on with the submission even though I did not have time yet to 
organize some of the code the way I needed. Also, function documentation quality varies a lot, 
some parts of the code very well documented and others practically undocumented. 
