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
from keras import applications, utils, layers, models, callbacks, preprocessing
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

train_image_path = []
train_classes = []
for filename in listdir(TRAIN):
    if filename.endswith(".jpg"):
        clazz = int(filename.split("_")[0])
        if clazz < NUM_CLASSES:
            train_image_path.append(TRAIN + '/' + filename)
            train_classes.append(clazz)

val_image_path = []
val_classes = []
for filename in listdir(VAL):
    if filename.endswith(".jpg"):
        clazz = int(filename.split("_")[0])
        if clazz < NUM_CLASSES:
            val_image_path.append(VAL + '/' + filename)
            val_classes.append(clazz)

batch_size = 16
train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    # zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=applications.inception_v3.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN,  # this is the target directory
    classes=["{:02d}".format(x) for x in range(NUM_CLASSES)],
    target_size=(299, 299),  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=applications.inception_v3.preprocess_input
)

validation_generator = test_datagen.flow_from_directory(
    VAL,
    classes=["{:02d}".format(x) for x in range(NUM_CLASSES)],
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical'
)

model_name = "inceptionv3-" + str(DENSE) + "-" + str(NUM_CLASSES) + "-" + str(SIZE) + "-" + str(AUG)
try:
    model = models.load_model(model_name + ".h5")
    print("Loaded: " + model_name)
except OSError:
    base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    # utils.vis_utils.plot_model(base_model, to_file="inceptionv3.png")
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    if DENSE >= 3:
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
    if DENSE >= 2:
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(NUM_CLASSES, activation='softmax', name='my_dense')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    # utils.vis_utils.plot_model(model, to_file="my_inceptionv3.png")
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='auto')
if SIZE > NUM_CLASSES:
    train_size = SIZE
    val_size = SIZE//2
else:
    train_size = len(train_classes)
    val_size = len(val_classes)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(train_size // batch_size) * AUG,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=val_size // batch_size,
    verbose=1,
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

images =[]
for path in val_image_path:
    img = cv2.imread(path)
    img = resize(img, 299)
    img = centered_crop(img, 299, 299)
    images.append(img_as_float(img))

x_val = applications.inception_v3.preprocess_input(np.array(images))


#score = model.evaluate(x_val, y_val, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


prob = model.predict(x_val, batch_size=1, verbose=0)

Y_pred = np.argmax(prob, axis=1)
accuracy = (len(val_classes) - np.count_nonzero(Y_pred - val_classes) + 0.0)/len(val_classes)
print("Accuracy on validation set of %d samples: %f" % (len(x_val), accuracy))

print("--- %s seconds ---" % (time() - start_time))
