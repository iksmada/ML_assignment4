import argparse
from os import listdir, environ, path, makedirs
import operator
import csv
import pickle
import re
from time import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn import model_selection
from skimage import img_as_float
import tensorflow as tf
from keras import applications, utils, layers, models, losses, callbacks
import pydot
import h5py
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    elif h < w:
        height = size

    # check to see if the width is None
    if height is not None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif width is not None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (size, size)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

start_time = time()
parser = argparse.ArgumentParser(description='Dog breed classifier')
parser.add_argument('-s', '--size', type=int, help='Train Size to use', default=-1)
parser.add_argument('-n', '--num-classes', type=int, help='Number of classes to train with',
                    default=83)
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs',
                    default=20)
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
NUM_CLASSES = args["num_classes"]
EPOCHS = args["epochs"]

train_image_path = []
train_classes = []
for filename in listdir(TRAIN):
    if filename.endswith(".jpg"):
        clazz = int(filename.split("_")[0])
        if clazz < NUM_CLASSES:
            train_image_path.append(TRAIN + '/' + filename)
            train_classes.append(clazz)

test_image_path = []
for filename in listdir(TEST):
    if filename.endswith(".jpg"):
        test_image_path.append(TEST + '/' + filename)


val_image_path = []
val_classes = []
for filename in listdir(VAL):
    if filename.endswith(".jpg"):
        clazz = int(filename.split("_")[0])
        if clazz < NUM_CLASSES:
            val_image_path.append(VAL + '/' + filename)
            val_classes.append(clazz)

if 2*len(train_classes)//3 > SIZE > 83 == NUM_CLASSES and SIZE >= NUM_CLASSES * 100:
    x_train, x_val, y_train, y_val = model_selection.train_test_split(train_image_path, train_classes, train_size=SIZE, test_size=SIZE//2)
elif 83 < SIZE < 2*len(train_classes)//3:
    sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=SIZE//2, train_size=SIZE)
    for train_index, test_index in sss.split(train_image_path, train_classes):
        train_image_path = np.array(train_image_path)
        train_classes = np.array(train_classes)
        x_train, x_val = train_image_path[train_index], train_image_path[test_index]
        y_train, y_val = train_classes[train_index], train_classes[test_index]

else:
    x_train = train_image_path
    y_train = train_classes
    x_val = val_image_path
    y_val = val_classes

images = []
for path in x_train:
    img = cv2.imread(path)
    img = resize(img, 299)
    img = centered_crop(img, 299, 299)
    images.append(img)

x_train = np.array(images)
y_train = utils.np_utils.to_categorical(y_train, num_classes=NUM_CLASSES)

images = []
for path in x_val:
    img = cv2.imread(path)
    img = resize(img, 299)
    img = centered_crop(img, 299, 299)
    images.append(img)

x_val = np.array(images)
y_val_flat = y_val
y_val = utils.np_utils.to_categorical(y_val, num_classes=NUM_CLASSES)

model_name = "inceptionv3-" + str(NUM_CLASSES) + "-" + str(SIZE)  # + "-"
try:
    model = models.load_model(model_name + ".h5")
    print("Loaded: " + model_name)
except OSError:
    base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    utils.vis_utils.plot_model(base_model, to_file="inceptionv3.png")
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(NUM_CLASSES, activation='softmax', name='my_dense')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    utils.vis_utils.plot_model(model, to_file="my_inceptionv3.png")

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])
earlyStopping = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, mode='auto')
history = model.fit(x_train, y_train, epochs=EPOCHS, verbose=2, validation_data=(x_val, y_val),
                    callbacks=[earlyStopping])
model.save(model_name + ".h5")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
makedirs("plots", exist_ok=True)
plt.savefig("plots/" + model_name + "-acc.png")
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("plots/" + model_name + "-loss.png")
plt.show()

#score = model.evaluate(x_val, y_val, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

prob = model.predict(x_val, batch_size=1, verbose=0)

Y_pred = np.argmax(prob, axis=1)
accuracy = (len(y_val_flat) - np.count_nonzero(Y_pred - y_val_flat) + 0.0)/len(y_val_flat)
print("Accuracy on validation set of %d samples: %f" % (len(x_val), accuracy))

print("--- %s seconds ---" % (time() - start_time))
