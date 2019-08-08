
import sys
import tensorflow as tf
import keras
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras import metrics
from keras_tqdm import TQDMCallback
from keras.callbacks import CSVLogger
import os
import pickle
import imageio
from PIL import Image

import datagenerators
sys.path.append('../data/')
import classification

on_cloud = True
if not on_cloud:
    IMG_FOLDER = '../../images/'
else:
    IMG_FOLDER = '../data/images/'
def transfer(train_fraction = 0.8, validation_fraction = 0.1, test_fraction = 0.1,
            verbose = False, log_file = '../../../addie_chambers/saved_models/transfer_log'):

    trainable_vals = [i for i in range(0, 20)]
    dataset_size = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]


    for size in dataset_size:
        for trainable in trainable_vals:
            train, validation, test = classification.generate_uid_sets(images_to_use = size,
                                                                train_fraction = train_fraction,
                                                                validation_fraction = validation_fraction,
                                                                verbose = verbose)
            train_df = classification.make_dataframe(train)
            generator = datagenerators.generator(train_df, IMG_FOLDER)
            validation_generator = datagenerators.generator(classification.make_dataframe(validation), IMG_FOLDER)
        
            # want to reset model every time
            trained_resnet = ResNet50(include_top = False, weights = 'imagenet',
                            input_tensor = None, input_shape = (1024, 1024, 3),
                            pooling = None, classes = 1000)
            x = trained_resnet.output
            x = Flatten()(x)
            predictions = Dense(15, activation='sigmoid')(x)
            num_layers = len(trained_resnet.layers)
            trained = 0
            for i in range(num_layers):
                if trained < trainable and trained_resnet.layers[num_layers - i - 1].count_params() > 0:
                    trained_resnet.layers[num_layers - i - 1].trainable = True
                    trained += 1
                else:
                    trained_resnet.layers[num_layers - i - 1].trainable = False

            model = Model(inputs=trained_resnet.input, outputs=predictions)

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.categorical_accuracy])
            #model.summary()
            trainable_count = int(np.sum([keras.backend.count_params(p) for p in set(model.trainable_weights)]))
            print("Total model trainable parameters are:" , trainable_count)
            print("Restricting total dataset size to" , size)
            res = model.fit_generator(generator, epochs = 5, steps_per_epoch = 100, validation_data=validation_generator,
                            validation_steps = 100)
            train_majority_accuracy = classification.gt_majority_baseline(train)
            valid_majority_accuracy = classification.gt_majority_baseline(validation)
        
            hist = res.history
            hist['train_majority_acc'] = [train_majority_accuracy]
            hist['validation_majority_acc'] = [valid_majority_accuracy]
            with open(log_file + '_' + str(size) + '_' + str(trainable), 'wb') as f:
                pickle.dump(hist, f)

def optimal_transfer(optimal_size, optimal_layer, train_fraction = 0.8, validation_fraction = 0.1,
        log_file = '../../../addie_chambers/saved_models/', verbose = False):
    train, validation, test = classification.generate_uid_sets(images_to_use = optimal_size,
            train_fraction = train_fraction,
            validation_fraction = validation_fraction,
            verbose = verbose)
    train_df = classification.make_dataframe(train)
    generator = datagenerators.generator(train_df, IMG_FOLDER)
    validation_generator = datagenerators.generator(classification.make_dataframe(validation), IMG_FOLDER)

    trained_resnet = ResNet50(include_top = False, weights = 'imagenet',
            input_tensor = None, input_shape = (1024, 1024, 3),
            pooling = None, classes = 1000)
    x = trained_resnet.output
    x = Flatten()(x)
    predictions = Dense(15, activation = 'sigmoid')(x)
    num_layers = len(trained_resnet.layers)
    trained = 0
    for i in range(num_layers):
        if trained < optimal_layer and trained_resnet.layers[num_layers - i - 1].count_params() > 0:
            trained_resnet.layers[num_layers - i - 1].trainable = True
        else:
            trained_resnet.layers[num_layers - i - 1].trainable = False

    model = Model(inputs = trained_resnet.input, outputs = predictions)

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=[metrics.categorical_accuracy])
    res = model.fit_generator(generator, epochs = 100, steps_per_epoch = 100, validation_data = validation_generator,
            validation_steps = 50)
    train_majority_accuracy = classification.gt_majority_baseline(train)
    valid_majority_accuracy = classification.gt_majority_baseline(validation)
    hist = res.history
    hist['train_majority_acc'] = [train_majority_accuracy]

    hist['validation_majority_acc'] = [valid_majority_accuracy]
    with open(log_file + 'transfer_optimal', 'wb') as f:
        pickle.dump(hist, f)
    
    disease_history = {}
    disease_dataframes = classification.split_dataframe_by_diseases(test)
    test_gen = datagenerators.generator(classification.make_dataframe(test), IMG_FOLDER)
    res = model.evaluate_generator(test_gen, steps = 500)
    disease_history['overall'] = res[1]
    disease_history['maj_acc'] = [classification.gt_majority_baseline(test)]

    for dis in disease_dataframes:
        gen = datagenerators.generator(disease_dataframes[dis], IMG_FOLDER)
        res = model.evaluate_generator(gen, steps = 25)
        disease_history[dis] = res[1]

    with open(log_file + 'transfer_optimal_test', 'wb') as f:
        pickle.dump(disease_history, f)

    classification.print_summary(set(), set(), test)

    
if __name__ == '__main__':
    transfer()
    optimal_transfer(3000, 19)


# In[ ]:





# In[ ]:




