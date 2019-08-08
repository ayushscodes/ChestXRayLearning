#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import imageio
import sys
import random
import keras
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('../data/')
import classification

IMG_SIZE = (1024, 1024)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)


def generator(df, relative_directory, image_data_generator = core_idg, vol_size = (1024, 1024), batch_size = 1):
    #Xs = list(uids)
    #Ys = [classification.generate_target_vector(x) for x in Xs]
    #classification.make_train_folder(uids, relative_directory)
    
    #p_split = relative_directory.split('/')
    #parent_directory = ""
    #for i in range(len(p_split) - 2):
    #    parent_directory += p_split[i] + '/
    
    print("code is in image_data_generator---------------------")
    """gen = image_data_generator.flow_from_directory(relative_directory,
                                            target_size = vol_size,
                                            batch_size = batch_size,
                                            class_mode = 'categorical',
                                            classes=['Atelectasis', 'Cardiomegaly',
                                                     'Effusion', 'Infiltration',
                                                     'Mass', 'Nodule', 'Pneumonia',
                                                     'Pneumothorax', 'Consolidation',
                                                     'Edema', 'Emphysema',
                                                     'Fibrosis', 'Pleural_Thickening',
                                                     'Hernia', 'NoFinding'])"""
    gen = image_data_generator.flow_from_dataframe(df,
                                                   directory=relative_directory,
                                                   x_col = 'ImgUID',
                                                   y_col = 'Labels',
                                                   batch_size = batch_size,
                                                   seed = 42,
                                                   shuffle = True,
                                                   class_mode = 'categorical',
                                                   classes = ['Atelectasis', 'Cardiomegaly',
                                                              'Effusion', 'Infiltration',
                                                              'Mass', 'Nodule', 'Pneumonia',
                                                              'Pneumothorax', 'Consolidation',
                                                              'Edema', 'Emphysema',
                                                              'Fibrosis', 'Pleural_Thickening',
                                                              'Hernia', 'NoFinding'],
                                                   target_size=vol_size)
    print('code is after image_data_generator--------------------')
    gen.class_indices = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3,
                        'Mass': 4, 'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8,
                        'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12,
                        'Hernia': 13, 'NoFinding': 14}
    
    #gen.classes = np.asarray([i * 1.0 for i in range(15)])
    #gen.num_classes = 15
                             
    
    return gen

#def training_generator(uids, example_gen = generate_example, image_data_generator = core_idg):

if __name__ == "__main__":
    train, valid, test = classification.generate_uid_sets(10, verbose = False)
    df = classification.make_dataframe(train)
    gen = generator(df, '../../images')

    # should do everything you need to do
    #classification.clean_up('../fake/')





