#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:15:52 2018

@author: chloeloughridge
"""

# FOR STUDENTS

# This is a file for constructing the model's architecture

# import our libraries
import keras
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from keras.models import Model

# function for constructing the model's architecture
def create_model(pic_width, pic_height, color_channels, num_classes):
    
    # create the model's input layer
    # TODO
    
    # THIS IS WHERE MODEL ARCHITECTURE DESIGN STARTS: ENTERING CREATIVE FREEDOM ZONE
    # TODO
    
    # instantiate the model
    # TODO
    
    # return the model
    return model

# if you run this file from the command line, the following will execute
if __name__ == "__main__":
    model = create_model(28, 28, 1, 10)
    print(model.summary())