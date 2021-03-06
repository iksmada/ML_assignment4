import argparse
from os import listdir, environ, path, makedirs
import operator
import csv
import pickle
import re
from time import time

import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from keras import applications, utils, layers, models, callbacks, preprocessing
import pydot
import h5py

from main import f1

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


batch_size = 16

test_datagen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=applications.inception_v3.preprocess_input
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

model_name = "inceptionv3-3"# + str(DENSE) + "-" + str(CLASSES) + "-" + str(SAMPLES) + "-" + str(AUG)
model = models.load_model(model_name + ".h5", custom_objects={"f1": f1})
print("Loaded: " + model_name)

#score = model.evaluate_generator(test_generator, verbose=2, steps=len(test_generator.filenames)//batch_size)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


prob = model.predict_generator(test_generator, verbose=1, steps=len(test_generator.filenames)//batch_size)

y_pred = np.argmax(prob, axis=1)
y_true = test_generator.classes

print("Accuracy on test set of %d samples: %f" % (len(y_true), accuracy_score(y_true, y_pred)))

cmat = confusion_matrix(y_true, y_pred)
cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
print(cmat)
acc_per_class = cmat.diagonal()/cmat.sum(axis=1)
np.set_printoptions(precision=4)
print("Normalized Accuracy on test set: %f" % (np.mean(acc_per_class)))
print("F1 Score on test set: %f" % (f1_score(y_true, y_pred, average="macro")))

wrong_eval = np.where((y_pred - y_true))[0]
max_pred = 0
min_true = 1
i_pred = 0
i_true = [0]
for i in wrong_eval:
    pred_prob = prob[i][y_pred[i]]
    true_prob = prob[i][y_true[i]]
    if pred_prob >= max_pred:
        max_pred = pred_prob
        i_pred = i
    if true_prob <= min_true:
        min_true = true_prob
        i_true.append(i)

if i_pred == i_true[-1]:
    i_true = i_true[-2]
else:
    i_true = i_true[-1]

yellow = (0, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread(TEST + "/" + test_generator.filenames[i_pred])
cv2.putText(img, 'True Class (%d): %.2f' % (y_true[i_pred], prob[i_pred][y_true[i_pred]]), (10, 30), font, 0.5,
            yellow, 2, cv2.LINE_AA)
cv2.putText(img, 'Pred Class (%d): %.2f' % (y_pred[i_pred], prob[i_pred][y_pred[i_pred]]), (10, 70), font, 0.5,
            yellow, 2, cv2.LINE_AA)
print("Worst Pred index: %d" % (i_pred))
cv2.imshow("Worst Pred", img)
img = cv2.imread(TEST + "/" + test_generator.filenames[i_true])
cv2.putText(img, 'True Class (%d): %.2f' % (y_true[i_true], prob[i_true][y_true[i_true]]), (10, 30), font, 0.5,
            yellow, 2, cv2.LINE_AA)
cv2.putText(img, 'Pred Class (%d): %.2f' % (y_pred[i_true], prob[i_true][y_pred[i_true]]), (10, 70), font, 0.5,
            yellow, 2, cv2.LINE_AA)
print("Worst True index: %d" % (i_true))
cv2.imshow("Worst True", img)
cv2.waitKey(0)

print("--- %s seconds ---" % (time() - start_time))
