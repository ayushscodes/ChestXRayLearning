from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.keras import utils
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
import warnings
import os
import sys
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf
sys.path.append('../data/')
import classification
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import datagenerators








def baseline_resnet50(num_images_to_use, train_fraction=0.8, validation_fraction=0.1, test_fraction=0.1, temp_folder='../models/baseline/temp', verbose=False):
    """
    train and evaluate baseline resnet50 on ChestXRay dataset
    """

    train, validation, test = classification.generate_uid_sets(images_to_use=num_images_to_use,
                                                               train_fraction=train_fraction,
                                                               validation_fraction=validation_fraction,
                                                               verbose=verbose)

    IMG_FOLDER = '../data/images/'

    train_df = classification.make_dataframe(train)
    train_generator = datagenerators.generator(train_df, IMG_FOLDER, batch_size=3)

    valid_df = classification.make_dataframe(validation)
    validation_generator = datagenerators.generator(valid_df, IMG_FOLDER, batch_size=3)

    test_df = classification.make_dataframe(validation)
    test_generator = datagenerators.generator(
        test_df, IMG_FOLDER, batch_size=3)


    # check if saved model exists
    saved_model_path = "../saved_models/resnet50-multi-label.h5"
    train_hist_path = "../saved_models/train_history_dict_multi-label"
    saved_model = os.path.isfile(saved_model_path)

    if not saved_model: 
        model = resnet50(15)
        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        checkpoint = ModelCheckpoint(
            saved_model_path, monitor='loss', verbose=1, save_best_only='True', mode='min')

        model_run = model.fit_generator(
                        generator,
                        epochs=10,
                        steps_per_epoch=10000, 
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data = validation_generator,
                        validation_steps=1000)

        with open(train_hist_path, 'wb') as f:
            pickle.dump(model_run.history, f)
    
    else: 
        print("EVALUATING SAVED MODEL----------------------------")
        resumed_model = load_model(saved_model_path)
        with open(train_hist_path, 'wb') as f:
            pickle.dump(model_run.history, f)

        train_evaluate_resumed_model = resumed_model.evaluate_generator(train_generator, verbose=0)
        val_evaluate_resumed_model = resumed_model.evaluate_generator(validation_generator, verbose=0)
        test_evaluate_resumed_model = resumed_model.evaluate_generator(test_generator, verbose=0)
        

       


