import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
	train_files = ['data_batch_1', 'data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    test_file = 'test_batch'
    x_train = np.empty(shape=[0, 3072])
    y_train = np.empty(shape=[0,])
    
    for file in train_files:
      f = open(data_dir+file, 'rb')
      dict = pickle.load(f, encoding='bytes')
      images = dict[b'data']
      labels = dict[b'labels']
      imagearray = np.array(images)
      labelarray = np.array(labels)
      x_train = np.concatenate((x_train, imagearray),axis=0)
      y_train = np.concatenate((y_train, labelarray),axis=0)

    f = open(data_dir+test_file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    images = dict[b'data']
    labels = dict[b'labels']
    x_test = np.array(images)
    y_test = np.array(labels)
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid