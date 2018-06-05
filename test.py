import argparse
from os import listdir, environ, path, makedirs
import operator
import csv
import pickle
import re
from time import time

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from keras import applications, utils, layers, models, callbacks, preprocessing
import pydot
import h5py
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from joblib import Parallel, delayed
import multiprocessing


start_time = time()
parser = argparse.ArgumentParser(description='Dog breed classifier')
parser.add_argument('-s', '--samples', type=int, help='Train Size to use',
                    default=-1)
parser.add_argument('-c', '--classes', type=int, help='Number of classes to train with',
                    default=83)
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs',
                    default=20)
parser.add_argument('-dl', '--dense', type=int, help='Number of dense layers',
                    default=3)
parser.add_argument('-do', '--drop', type=float, help='Percentage of drop-out',
                    default=0.2)
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
SAMPLES = args["samples"]
TRAIN = args["input_train"]
TEST = args["input_test"]
VAL = args["input_val"]
CLASSES = args["classes"]
EPOCHS = args["epochs"]
DENSE = args["dense"]
DROP = args["drop"]
AUG = args["augmentation"]

test_image_path = []
test_classes = []
for clazz in np.sort(listdir(TEST)):
    if path.isdir(TEST + "/" + clazz) and int(clazz) < CLASSES:
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
    classes=["{:02d}".format(x) for x in range(CLASSES)],
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

y_pred = np.argmax(prob, axis=1)
y_true = np.array(test_classes)

print("Accuracy on test set of %d samples: %f" % (len(y_true), accuracy_score(y_true, y_pred)))


cmat = confusion_matrix(y_true, y_pred)
cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print(cmat)
acc_per_class = cmat.diagonal()/cmat.sum(axis=1)
print("Normalized Accuracy on test set: %f" % (np.mean(acc_per_class)))

print("--- %s seconds ---" % (time() - start_time))
