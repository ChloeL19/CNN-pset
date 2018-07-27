#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:08:41 2018

@author: chloeloughridge
"""

#FOR STUDENTS
# This file is for uploading and preprocessing the mnist dataset

# first, let's import keras
import keras
# for visualizing data we need this:
import matplotlib.pyplot as plt
# Keras has the MNIST dataset preloaded. Yay for us!
from keras.datasets import mnist

def preprocess_mnist(verbose=True):
    # import mnist data from keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    # if in verbose mode, print an image from the dataset along with its label
    # TODO
    
    # We need to reshape the X_train and X_test data so that each image within it is a 28 by 28 pixel square.
    # Like this: [the number of training examples, image width, image height, the number of color channels]
    # TODO
    
    # We need to normalize the X data before feeding into our model
    # This means we want to put each pixel value in the range 0 to 1
    # TODO
    
    # We also need to convert the Y data into one-hot vectors.
    # TODO
    
    # we ultimately want our function to return (X_train, Y_train) and (X_test, Y_test)
    return (X_train, Y_train), (X_test, Y_test)