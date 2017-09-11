#!/usr/bin/python

'''
model.py

This script is used to train and generate a neural network
model to have a simulated car stay in the middle of a road. 

Bryant Pong
9/7/17
'''

import csv
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Path to the CSV Files:
PATH_TO_TRAINING_DATA = "/home/bryant/car-data/"
PATH_TO_IMAGE_DATA = PATH_TO_TRAINING_DATA + "IMG/"

# CSV Files containing steering data:

# Counterclockwise Driving
# This log contains driving data for a single counter-clockwise
# loop around Track 1.
CCW_LOG = PATH_TO_TRAINING_DATA + "ccw_driving_log.csv"

# Clockwise Driving
# This log contains driving data for a single clockwise
# loop around Track 1.
CW_LOG = PATH_TO_TRAINING_DATA + "cw_driving_log.csv"

# Recovery
# This log contains driving data for small recovery maneuvers.
# Think of correcting slight drifts to the left and right.
RECOVERY_LOG = PATH_TO_TRAINING_DATA + "recovery_driving_log.csv"

# Extreme Recovery:
# This log contains driving data for more extreme recovery maneuvers.
# This handles cases such as driving onto the area between a yellow lane line
# and the curb, or driving off a sharp curve into the red/white striped area. 
EXTREME_RECOVERY_LOG = PATH_TO_TRAINING_DATA + "extreme_recovery_driving_log.csv"

# Extract all the lines from the CVS files:
lines = []
with open( CCW_LOG ) as csvFile:
    reader = csv.reader( csvFile )
    for line in reader:
        lines.append( line )
with open( CW_LOG ) as csvFile:
    reader = csv.reader( csvFile )
    for line in reader:
        lines.append( line )
with open( RECOVERY_LOG ) as csvFile:
    reader = csv.reader( csvFile )
    for line in reader:
        lines.append( line )
with open( EXTREME_RECOVERY_LOG ) as csvFile:
    reader = csv.reader( csvFile )
    for line in reader:
        lines.append( line )

# Now create two lists that hold the images
# and the steering angles corresponding
# to the image:
images = []
headings = []

'''
As my computer has 32 GB of RAM, I did not need to use
a Python Generator to help with loading all the training
images.  However, a generator is more ideal since I've noticed
that with each frame contributing 6 images (3 for the original
left, right, and center images and an additional 3 for the flipped
left, right, and center images) took a whopping 10 GB on my machine.
'''
for line in lines:

    # First field contains the path to the center image: 

    # Get the file name:
    centerImageName = PATH_TO_IMAGE_DATA + line[ 0 ].split( "/" )[ -1 ]
    leftImageName = PATH_TO_IMAGE_DATA + line[ 1 ].split( "/" )[ -1 ]
    rightImageName = PATH_TO_IMAGE_DATA + line[ 2 ].split( "/" )[ -1 ]

    centerImage = cv2.imread( centerImageName )
    images.append( centerImage )

    # Second field contains the path to the left image:
    leftImage = cv2.imread( leftImageName )
    images.append( leftImage )

    # Third field contains the path to the right image:
    rightImage = cv2.imread( rightImageName )
    images.append( rightImage )

    # Steering angles are stored in the 4th field:
    centerHeading = float( line[ 3 ] )
    headings.append( centerHeading )

    # Add a small constant for correcting steering left and right.
    STEERING_CONSTANT = 0.35
    leftHeading = centerHeading + STEERING_CONSTANT
    headings.append( leftHeading )

    rightHeading = centerHeading - STEERING_CONSTANT
    headings.append( rightHeading )

    # Now flip the images along the Y-Axis and flip the steering angle.
    # This gives us more data to train on:
    flippedCenterImage = cv2.flip( centerImage, 1 )
    flippedCenterHeading = -1.0 * centerHeading
    images.append( flippedCenterImage )
    headings.append( flippedCenterHeading )

    flippedLeftImage = cv2.flip( leftImage, 1 )
    flippedLeftHeading = -1.0 * leftHeading
    images.append( flippedLeftImage )
    headings.append( flippedLeftHeading )

    flippedRightImage = cv2.flip( rightImage, 1 )
    flippedRightHeading = -1.0 * rightHeading
    images.append( flippedRightImage )
    headings.append( flippedRightHeading )

print( "Done loading images and headings" )

# These Numpy arrays will containg the training
# data.  The network will train on the images
# and the labels will be the steering headings.
X_train = np.array( images )
y_train = np.array( headings )

# Including the flipped images, we're working with ~28000 images here.
print( len( X_train ) )
print( len( y_train ) )

# Main function:
def main():
   
    print( "Now training network" )

    # Neural network model:
    model = Sequential()

    # The input images are 320x160 RGB ones:

    # Image Preprocessing:
    # Perform image normalization and mean-centering:
    model.add( Lambda( lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))

    # Cropping Layer.  We want to crop 65 pixels from the top and 30 pixels
    # from the bottom.  This allows the network to ignore the background
    # and the hood of the car.  The values were determined by opening
    # images in the GIMP program and manually cropping out undesired areas.
    model.add( Cropping2D( cropping = ( ( 65, 30 ), ( 0, 0 ) ) ) ) 

    # Convolutional Layer 1.  Input = 320x160x3.  Output = 316x156x16 
    model.add( Conv2D( 16, 5, 5, input_shape = ( 160, 320, 3 ) ) )
    model.add( Activation( 'relu' ) )

    # Max Pooling Layer 1 (defaults to 2x2 pooling filter).
    # Input: 316x156x16.  Output = 158x78x16
    model.add( MaxPooling2D() )

    # Dropout Layer 1.  Keep probability of 0.5
    model.add( Dropout( 0.5 ) )
    
    # Convolutional Layer 2.  Input = 158x78x16.  Output = 154x74x16. 
    model.add( Conv2D( 16, 5, 5 ) )
    model.add( Activation( 'relu' ) )

    # Convolutional Layer 3.  Input = 154x74x16.  Output = 150x70x16.
    model.add( Conv2D( 16, 5, 5 ) )
    model.add( Activation( 'relu' ) )

    # Convolutional Layer 4.  Input = 150x70x16.  Output = 148x78x16.
    model.add( Conv2D( 16, 3, 3 ) )
    model.add( Activation( 'relu' ) )

    # Max Pooling Layer 2 (defaults to 2x2 pooling filter).
    # Input: 148x78x16.  Output = 74x39x16.
    model.add( MaxPooling2D() )

    # Dropout Layer 2.  Keep probability of 0.5.
    model.add( Dropout( 0.5 ) )

    # Flatten the input image into an array of 74*39*16 = 46176:
    model.add( Flatten() )

    # Dense Network of 120 neurons
    model.add( Dense( 120 ) )

    # Dense Network of 84 neurons
    model.add( Dense( 84 ) )

    # The network needs to learn the appropriate steering heading, thus
    # there's only a single output needed (steering heading).
    model.add( Dense( 1 ) )

    # As this is a regression network, I will be using the Mean-Squared Error
    # instead of the softmax loss function.  The Adam Optimizer is also used.  
    model.compile( loss = 'mse', optimizer = 'adam' )

    '''
    The ~28k images will be split into a training data set (80% of the images)
    and a validation data set (20% of the images).  This means that we're using 
    ~6000 images for validation.
    '''
    model.fit( X_train, y_train, validation_split = 0.2, shuffle = True )

    # Save the model:
    model.save( "model.h5" )

    print( "Done training network" )

# Main function runner:
if __name__ == "__main__":
    main()
