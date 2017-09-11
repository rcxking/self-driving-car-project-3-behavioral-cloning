# self-driving-car-project-3-behavioral-cloning
Repository for Udacity's Self-Driving Car Nanodegree Project 3 - Behavioral Cloning

[image1]: ./examples/center.jpg "Center Driving"
[image2]: ./examples/center_cw.jpg "Center Clockwise Driving"
[image3]: ./examples/recovery_1.jpg "Recovery 1"
[image4]: ./examples/recovery_2.jpg "Recovery 2"
[image5]: ./examples/recovery_3.jpg "Recovery 3"

1. Included Files

This repository contains the following files:
* model.py: Contains the Python code to create and train the model
* drive.py: Included to facility playback of model.h5
* model.h5: The model of the trained convolutional network for driving the car
* video.mp4: MP4 Video of the car driving 2 Laps in the Simulator after loading model.h5
* README.md: (This file) - Contains the writeup and explanations for this project
* examples/: This folder contains example training data images I generated

2. Running the Simulator with the Trained Model
With Udacity's Included Simulator and the included drive.py file, a user can load my neural network model via:

```sh
python drive.py model.h5
```

3. Model.py

Model.py contains my implementation of a Convolutional Neural Network using the Keras library.  Please refer to this file's comments for my network's architecture and how training was implemented.

It is important to note that as my computer used for training this network has 32 GB of RAM, I did not require a Python generator to aid in loading all ~28k images for training and validation.  However, if a network required even more data, I would opt for the generator to reduce RAM usage.  For my 28k images, my computer used ~10 GB of RAM.

Model Architecture and Training Strategy

1. Model Architecture

My network architecture consists of 4 Convolutional Layers of depth 16.  3 of these layers use 5x5 filters; the remaining convolutional layer uses a 3x3 filter.  These layers are on code lines 163, 174, 178, and 182.  

To introduce nonlinearity, I added a RELU activation layer after each of the 4 convolutional layers.  These are respectively on code lines 164, 175, 179, and 183.

2 Max Pooling Layers with the default size of 2x2 are introduced in my model.  The first max pooling layer is after the first 5x5 convolutional layer on code line 168; the second max pooling layer is after the only 3x3 convolutional layer on code line 187.

To remove unneeded background information from the top and bottom of the image, I used a Cropping2D layer (code line 160).  I used the GIMP image utility to determine an appropriate amount of pixels to crop from the top and bottom.

Finally, image normalization and mean-centering are performed via a Keras lambda layer (code line 154). 


2. Reducing Overfitting

Overfitting was reduced by the addition of 2 Dropout layers (code lines 171 and 190).  These dropout layers have a keep-probability of 0.5.

While developing my network, I also reduced the number of epochs when I noticed that overfitting occurred (i.e. training loss decreased while validation loss increased/oscillated).  However, for the final configuration of my network I noticed that the default of 10 epochs worked perfectly fine.

3. Model Parameter Tuning

As this model used an ADAM optimizer, the learning rate was not tuned manually.

4. Training Data

I created 4 sets of training data.  The first set was of the car driving in a counter-clockwise direction around the track 1 loop while the car was in the middle of the road.  The second set was of the car driving on the middle of the road in a clockwise direction 1 time.  For the third set I created images of the car making small recovery maneuvers; i.e. small corrections from both sides needed to get the car back on the middle of the road.  Finally, the fourth set includes more "extreme" recovery moves; i.e. the car has crossed the yellow or red/white lane lines and a severe correction is needed immediately.

Please refer to the next section for more details about these training sets.

In-Depth Explanations of Network and Data Creation

1. Designing the Network

I started the creation of my network by implementing the LeNet Model.  This model was chosen because of its simplicity in implementation, yet it was also a step-up from a single fully-connected layer.  LeNet served an additional purpose by allowing me to quickly test my training environment (I created and trained this network on my personal computer rather than on AWS).

As an aside, the first two data sets I created for immediate training were the success cases; that is, the car was driving close to the center as possible.  I decided to create the counter-clockwise data set at this time because I did not want the car to "overfit" and assume that the track was always/mostly curving to the left.

The data taken was split such that 80% of the images were used for training while the remaining 20% were used for validation (validation_split = 0.2).  

After ensuring that the LeNet implementation worked (to a degree), the first layers I added were the normalization/mean-centering layer and the Cropping2D layer.  Image normalization was done with a Keras Lambda layer.  As mentioned above, I used the GIMP image program to look at several training images to determine an area to crop; with these parameters I created a Cropping2D layer to ignore these regions.  After these two layers were created, the model performed better.    

At this time, I noticed that the model was overfitting; while my training loss was decreasing per epoch, the validation loss fluctuated wildly.  To solve this problem, I borrowed concepts from my Project 2, adding in 2x2 MaxPooling Layers, and a series of 5x5 and 3x3 Convolutional Layers.  Finally, I added Dropout layers to combat overfitting.  The resulting network now could stay on mostly straight portions of the track reasonably well.

However, I now noticed that the car began to drift to the left, and the car could not handle sharp curves at all.  I realized that I had not created data sets to handle both small and extreme recovery maneuvers.  I quickly went back and generated two more datasets of how to recover from slight drifts to extreme drifts and retrained the network.  This extra data is what allows my neural network model to correct for these drifts.

At this point, the car drove very well, recovering off of the red/white sharp curves gracefully.  However, two problem spots remained.  The first was the bridge.  The bridge is wider than the yellow lane lines; as the car drove off the bridge back onto the road, the car would veer left and crash into the left curb.  The second problem spot were the two off-road sections on the right-hand side (1 at the beginning of the track and the other immediately after the first curve on the bridge).  The car would drive straight onto the off-road section and continue onwards.  To solve this problem, I augmented the recovery datasets with maneuvers to avoid these hazards.

With these cases handled, the resulting model.h5 was generated.

2. Final Model Architecture

Without reexplaining the details, the following is the model architecture:

|Layer                           Description     |
:------------------------------------------------:
|Input              320x160x3 RGB Image          |
|Lambda        Normalization/Mean Centering      |
|Cropping2D    Cropping 65 px top, 30 px bottom  |
|Conv 5x5x16                                     |
|RELU Activation Layer                           |
|2x2 Max Pooling                                 |
|Dropout Layer (Keep Probability of 0.5)         |
|Conv 5x5x16                                     |
|RELU Activation Layer                           |
|Conv 5x5x16                                     |
|RELU Activation Layer                           |
|Conv 3x3x16                                     |
|RELU Activation Layer                           |
|2x2 Max Pooling                                 |
|Dropout Layer (Keep Probability of 0.5)         |
| Flatten Layer                                  |
| Fully Connected Layer 120                      |
| Fully Connected Layer 84                       |
| Fully Connected Layer 1                        |

3. Training Data Creation

As mentioned above, the first two data sets created was that of the car driving in the middle of the road in both the clockwise and counter-clockwise directions, as seen below:

![Counter-Clockwise Driving][image1]
![Clockwise Driving (Note the lake on the right side)][image2]

Next, I created two additional datasets illustrating recoveries on different hazards (i.e. yellow lane lines, red/white striped sharp curves, the bridge, and the off-road zones).  The first dataset was for more gentle recoveries due to normal drifting; the second dataset was for sharp recoveries such as driving off the yellow line.

An example of a recovery from driving onto the red/white lines is as follows:

![Recovery 1][image3]
![Recovery 2][image4]
![Recovery 3][image5]

With these 4 datasets, I also reflected the center, left, and right images and steering angles over the Y-Axis to get extra data.  This simulated driving in the opposite direction.  Thus for each frame, I was able to extract 6 images instead of 3 (center, left, right, reflected center, reflected left, and reflected right images).

After this process was complete, I had a total of 28,980 images.  As mentioned above, these input images were normalized, mean-centered, and then cropped to ignore the top 65 and bottom 30 pixels.

The dataset was split such that 80% of these images were used for training; the remaining 20% were used for validation.

As I used an ADAM optimized, I didn't need to manually tune the learning rate.  I stuck with the default 10 epochs because I noticed that overfitting was no longer a problem with my network.    
