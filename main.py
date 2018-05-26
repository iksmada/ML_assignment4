import argparse
from os import listdir, path, makedirs
import operator
import csv
import pickle
import re
from time import time

import numpy as np
import cv2

from sklearn import  model_selection

from joblib import Parallel, delayed
import multiprocessing

def centered_crop(img, new_height=299, new_width=299):
    width = np.size(img, 1)
    height = np.size(img, 0)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    c_img = img[top:bottom, left:right, :]
    return c_img


def resize(image, size=299, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    width = None
    height = None
    (h, w) = image.shape[:2]

    if h > w:
        width = size
    else:
        height = size

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

start_time = time()
parser = argparse.ArgumentParser(description='Dog breed classifier')
parser.add_argument('-s', '--size', type=int, help='Train Size to use', default=-1)
parser.add_argument('-i', '--input-train', type=str, help='Path to files containing the train dataset',
                    default='MO444_dogs/train')
parser.add_argument('-t', '--input-test', type=str, help='Path of files containing the test dataset',
                    default='MO444_dogs/test')
parser.add_argument('-v', '--input-val', type=str, help='Path of files containing the validation dataset',
                    default='MO444_dogs/val')

args = vars(parser.parse_args())
print(args)
SIZE = args["size"]
TRAIN = args["input_train"]
TEST = args["input_test"]
VAL = args["input_val"]

train_image_path = []
train_classes = []
for filename in listdir(TRAIN):
    if filename.endswith(".jpg"):
        train_image_path.append(TRAIN + '/' + filename)
        train_classes.append(int(filename.split("_")[0]))

test_image_path = []
for filename in listdir(TEST):
    if filename.endswith(".jpg"):
        test_image_path.append(TEST + '/' + filename)

val_image_path = []
val_classes = []
for filename in listdir(VAL):
    if filename.endswith(".jpg"):
        val_image_path.append(VAL + '/' + filename)
        val_classes.append(int(filename.split("_")[0]))

if SIZE > 0:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(train_image_path, train_classes, train_size=SIZE)
else:
    x_train = train_image_path
    y_train = train_classes
    x_test = val_image_path
    y_test = val_classes

images = []
for path in x_train:
    img = cv2.imread(path)
    img = resize(img, 299)
    img = centered_crop(img, 299, 299)
    images.append(img)


print("--- %s seconds ---" % (time() - start_time))
