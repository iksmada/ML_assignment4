import argparse
from os import listdir, environ, path, makedirs
import operator
import csv
import pickle
import re
from time import time

import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from keras import applications, utils, layers, models, callbacks, preprocessing
import pydot
import h5py
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from joblib import Parallel, delayed
import multiprocessing


start_time = time()
parser = argparse.ArgumentParser(description='Dog breed classifier')
parser.add_argument('-s', '--size', type=int, help='Train Size to use', default=-1)
parser.add_argument('-n', '--num-classes', type=int, help='Number of classes to train with',
                    default=83)
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs',
                    default=20)
parser.add_argument('-d', '--dense', type=int, help='Number of dense layers',
                    default=3)
parser.add_argument('-a', '--augmentation', type=int, help='Images generated using augmentation per image',
                    default=1)
parser.add_argument('-i', '--input-train', type=str, help='Path to files containing the train dataset',
                    default='MO444_dogs2/train')
parser.add_argument('-t', '--input-test', type=str, help='Path of files containing the test dataset',
                    default='MO444_dogs2/test')
parser.add_argument('-v', '--input-val', type=str, help='Path of files containing the validation dataset',
                    default='MO444_dogs2/val')

args = vars(parser.parse_args())
print(args)
SIZE = args["size"]
TRAIN = args["input_train"]
TEST = args["input_test"]
VAL = args["input_val"]
NUM_CLASSES = args["num_classes"]
EPOCHS = args["epochs"]
DENSE = args["dense"]
AUG = args["augmentation"]

test_image_path = []
test_classes = []
for clazz in np.sort(listdir(TEST)):
    if path.isdir(TEST + "/" + clazz) and int(clazz) < NUM_CLASSES:
        for filename in listdir(TEST + "/" + clazz):
            if filename.endswith(".jpg"):
                test_image_path.append(TEST + "/" + clazz + '/' + filename)
                test_classes.append(int(clazz))

batch_size = 16

test_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
batch_size = 1
test_generator = test_datagen.flow_from_directory(
    TEST,
    classes=["{:02d}".format(x) for x in range(NUM_CLASSES)],
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
test_generator.num_classes = 83

model_name = "inceptionv3-aug"

model = models.load_model(model_name + ".h5")
print("Loaded: " + model_name)

score = model.evaluate_generator(test_generator, verbose=2, steps=len(test_generator.filenames)//batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_generator.class_mode = None

prob = model.predict_generator(test_generator, verbose=2, steps=len(test_generator.filenames)//batch_size)

Y_pred = np.argmax(prob, axis=1)
accuracy = (len(test_classes) - np.count_nonzero(Y_pred - test_classes) + 0.0)/len(test_classes)
print("Accuracy on test set of %d samples: %f" % (len(test_classes), accuracy))

cm = confusion_matrix(test_classes, Y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)

print("--- %s seconds ---" % (time() - start_time))
