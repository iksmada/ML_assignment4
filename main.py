import argparse
from os import listdir, path, makedirs
import operator
import csv
import pickle
import re
from time import time
import numpy as np

from sklearn import  model_selection

from joblib import Parallel, delayed
import multiprocessing


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
