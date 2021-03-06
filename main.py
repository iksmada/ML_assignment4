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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from keras import applications, utils, layers, models, callbacks, preprocessing
from keras import backend as K
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


def resize(image, size=299, inter=cv2.INTER_AREA):
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


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


if __name__ == '__main__':
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

    batch_size = 32
    train_datagen = preprocessing.image.ImageDataGenerator(
        shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        preprocessing_function=applications.inception_v3.preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN,  # this is the target directory
        classes=["{:02d}".format(x) for x in range(CLASSES)],
        target_size=(299, 299),  # all images will be resized to 150x150
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    val_datagen = preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.inception_v3.preprocess_input
    )

    validation_generator = val_datagen.flow_from_directory(
        VAL,
        classes=["{:02d}".format(x) for x in range(CLASSES)],
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    model_name = "inceptionv32-" + str(DENSE) + "-" + str(CLASSES) + "-" + str(SAMPLES) + "-" + str(AUG)
    try:
        model = models.load_model(model_name + ".h5", custom_objects={"f1": f1})
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
            x = layers.Dropout(DROP)(x)
        if DENSE >= 2:
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(DROP)(x)
        predictions = layers.Dense(CLASSES, activation='softmax', name='my_dense')(x)
        model = models.Model(inputs=base_model.input, outputs=predictions)
        # utils.vis_utils.plot_model(model, to_file="my_inceptionv3.png")
        model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy', f1])
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='auto')
    if SAMPLES > CLASSES:
        train_size = SAMPLES
        val_size = SAMPLES // 2
    else:
        train_size = len(train_generator.filenames)
        val_size = len(validation_generator.filenames)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=(train_size // batch_size) * AUG,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=val_size // batch_size,
        verbose=2,
        callbacks=[earlyStopping])
    model.save(model_name + ".h5")

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    makedirs("plots", exist_ok=True)
    plt.savefig("plots/" + model_name + "-acc.png")
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("plots/" + model_name + "-loss.png")
    plt.show()
    # summarize history for f1
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('Model F1')
    plt.ylabel('F1 score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/" + model_name + "-f1.png")
    plt.show()

    test_datagen = preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.inception_v3.preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        TEST,
        classes=["{:02d}".format(x) for x in range(CLASSES)],
        target_size=(299, 299),
        batch_size=1,
        shuffle=False,
        class_mode='categorical'
    )


    score = model.evaluate_generator(test_generator, verbose=2, steps=len(test_generator.filenames))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test F1 score:', score[2])

    test_generator.class_mode = None

    prob = model.predict_generator(test_generator, verbose=2, steps=len(test_generator.filenames))

    y_pred = np.argmax(prob, axis=1)
    y_true = test_generator.classes

    print("Accuracy on test set of %d samples: %f" % (len(y_true), accuracy_score(y_true, y_pred)))

    cmat = confusion_matrix(y_true, y_pred)
    cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print(cmat)
    acc_per_class = cmat.diagonal()/cmat.sum(axis=1)
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
