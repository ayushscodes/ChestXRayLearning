
import pandas as pd
import numpy as np
from enum import Enum
import warnings
import sys
import os
from shutil import copy, rmtree


dirname = os.path.dirname(__file__)
data_file_name= os.path.join(dirname, 'Data_Entry_2017.csv')


class Disease(Enum):
    ATELECTASIS = 'Atelectasis'
    CARDIOMEGALY = 'Cardiomegaly'
    EFFUSION = 'Effusion'
    INFILTRATION = 'Infiltration'
    MASS = 'Mass'
    NODULE = 'Nodule'
    PNEUMONIA = 'Pneumonia'
    PNEUMOTHORAX = 'Pneumothorax'
    CONSOLIDATION = 'Consolidation'
    EDEMA = 'Edema'
    EMPHYSEMA = 'Emphysema'
    FIBROSIS = 'Fibrosis'
    PLEURAL_THICKENING = 'Pleural_Thickening'
    HERNIA = 'Hernia'
    NO_FINDING = 'NoFinding'

def get_label(dis):
    switcher = {
        'Atelectasis': Disease.ATELECTASIS,
        'Cardiomegaly': Disease.CARDIOMEGALY,
        'Effusion': Disease.EFFUSION,
        'Infiltration': Disease.INFILTRATION,
        'Mass': Disease.MASS,
        'Nodule': Disease.NODULE,
        'Pneumonia': Disease.PNEUMONIA,
        'Pneumothorax': Disease.PNEUMOTHORAX,
        'Consolidation': Disease.CONSOLIDATION,
        'Edema': Disease.EDEMA,
        'Emphysema': Disease.EMPHYSEMA,
        'Fibrosis': Disease.FIBROSIS,
        'Pleural_Thickening': Disease.PLEURAL_THICKENING,
        'Hernia': Disease.HERNIA,
        'NoFinding': Disease.NO_FINDING
    }
    return switcher.get(dis)
    
def loc(disease):
    switcher = {
        Disease.ATELECTASIS: 0,
        Disease.CARDIOMEGALY: 1,
        Disease.EFFUSION: 2,
        Disease.INFILTRATION: 3,
        Disease.MASS: 4,
        Disease.NODULE: 5,
        Disease.PNEUMONIA: 6,
        Disease.PNEUMOTHORAX: 7,
        Disease.CONSOLIDATION: 8,
        Disease.EDEMA: 9,
        Disease.EMPHYSEMA: 10,
        Disease.FIBROSIS: 11,
        Disease.PLEURAL_THICKENING: 12,
        Disease.HERNIA: 13,
        Disease.NO_FINDING: 14
    }
    return switcher.get(disease)

def decode(index):
    switcher = {
        0 : Disease.ATELECTASIS,
        1 : Disease.CARDIOMEGALY,
        2 : Disease.EFFUSION,
        3 : Disease.INFILTRATION,
        4 : Disease.MASS,
        5 : Disease.NODULE,
        6 : Disease.PNEUMONIA,
        7 : Disease.PNEUMOTHORAX,
        8 : Disease.CONSOLIDATION,
        9 : Disease.EDEMA,
        10 : Disease.EMPHYSEMA,
        11 : Disease.FIBROSIS,
        12 : Disease.PLEURAL_THICKENING,
        13 : Disease.HERNIA,
        14 : Disease.NO_FINDING
    }
    return switcher.get(index)
    
def labels():
    return {
        Disease.ATELECTASIS,
        Disease.CARDIOMEGALY,
        Disease.EFFUSION,
        Disease.INFILTRATION,
        Disease.MASS,
        Disease.NODULE,
        Disease.PNEUMONIA,
        Disease.PNEUMOTHORAX,
        Disease.CONSOLIDATION,
        Disease.EDEMA,
        Disease.EMPHYSEMA,
        Disease.FIBROSIS,
        Disease.PLEURAL_THICKENING,
        Disease.HERNIA,
        Disease.NO_FINDING
    }

# for now, keeping all of this data -- if later on we don't need it, filter down with usecols
print("file: ", data_file_name)
data = pd.read_csv(data_file_name,
                dtype = {
                    "Image Index": np.unicode_,
                    "Finding Labels": np.unicode_,
                    "Patient ID": np.int32,
                    "Patient Age": np.int32,
                    "Patient Gender": np.unicode_,
                    "View Position": np.unicode_,
                    "OriginalImage[Width": np.unicode_,
                    "Height]": np.unicode_,
                    "OriginalImage[x": np.float32,
                    "y]": np.float32
                })


image_uid_to_disease_mapping = {}
disease_to_image_uids_mapping = {
    Disease.ATELECTASIS : set(),
    Disease.CARDIOMEGALY : set(),
    Disease.EFFUSION : set(),
    Disease.INFILTRATION : set(),
    Disease.MASS : set(),
    Disease.NODULE : set(),
    Disease.PNEUMONIA : set(),
    Disease.PNEUMOTHORAX : set(),
    Disease.CONSOLIDATION : set(),
    Disease.EDEMA : set(),
    Disease.EMPHYSEMA : set(),
    Disease.FIBROSIS : set(),
    Disease.PLEURAL_THICKENING : set(),
    Disease.HERNIA : set(),
    Disease.NO_FINDING : set()
}

for index, row in data.iterrows():
    image_uid = row["Image Index"]
    diseases = row["Finding Labels"].replace(" " , "").split('|')
    image_uid_to_disease_mapping[image_uid] = set()
    for disease in diseases:
        label = get_label(disease)
        image_uid_to_disease_mapping[image_uid].add(label)
        disease_to_image_uids_mapping[label].add(image_uid)

def separate_into_diseases(test_uid_set):
    diseases = {}
    for uid in test_uid_set:
        labels = image_uid_to_disease_mapping[uid]
        for label in labels:
            if label in diseases:
                diseases[label].add(uid)
            else:
                diseases[label] = {uid}
    return diseases

def gt_majority_baseline(img_uid_set, verbose = False):
    label_mapping = {}
    for img in img_uid_set:
        ls = tuple(image_uid_to_disease_mapping[img])
        if ls in label_mapping:
            label_mapping[ls].add(img)
        else:
            label_mapping[ls] = {img}
    
    if verbose:
        print("Makeup of labels looks like this:")
        for l in label_mapping:
            print("  " , l , " -> " , len(label_mapping[l]))
        print("")
        print("")
    maj = None
    for l in label_mapping:
        if (maj == None) or len(label_mapping[maj]) < len(label_mapping[l]):
            maj = l
    if verbose:
        print("Majority of the datapoints are" , maj , "with a frequency of" , (len(label_mapping[maj]) / len(img_uid_set)))
    return len(label_mapping[maj]) / len(img_uid_set)
        
def generate_target_vector(img_uid):
    #feature_vector = [0 for i in range(len(disease_to_image_uids_mapping))]
    feature_vector = []
    for disease in image_uid_to_disease_mapping[img_uid]:
        #feature_vector[loc(disease)] = 1
        feature_vector.append(disease.value)
    return feature_vector

def get_string_disease_from_vector(vec):
    res = ""
    for i in range(len(vec)):
        res += vec[i]
    return res
    
def print_summary(train, valid, test):
    expected_proportion = 0.01
    for s in (train, valid, test):
        if s == train:
            print("Training set has make up:")
        elif s == valid:
            print("Validation set has make up:")
        else:
            print("Test set has make up:")
            
        for key in disease_to_image_uids_mapping:
            exs = len(s.intersection(disease_to_image_uids_mapping[key]))
            print("\t" , key , ":" , exs)
            if exs < len(s) * 0.01:
                message = "Disease " + str(key) + " makes up less than " + str(expected_proportion) + " of this set."
                warnings.warn(message)
        print("\n")
        
def split_dataframe_by_diseases(uids):
    data = {}
    for uid in uids:
        dis = generate_target_vector(uid)
        dis = get_string_disease_from_vector(dis)
        if dis in data:
            data[dis].append([uid, generate_target_vector(uid)])
        else:
            data[dis] = [[uid, generate_target_vector(uid)]]
    
    dfs = {}
    for dis in data:
        dfs[dis] = pd.DataFrame(data[dis], columns=["ImgUID", "Labels"])
    return dfs
        
       
def generate_uid_sets(images_to_use = None, verbose = False, train_fraction = 0.8, validation_fraction = 0.1, test_fraction = 0.1):
    """
    Returns (train_uids, validation_uids, test_uids), where train_uids is the set of image uids to use
    for training, validation_uids is the set of image uids to use for validation, and test_uids is the set of
    image uids to use for testing.
    
    :params
    
    :images_to_use: the number of images to limit our overall corpus to. None if we should use all available images.
    :verbose: wheterh to print out a summary of the designed sets
    :train_fraction: the fraction of the corpus to use for training
    :validation_fraction: the fraction of the corpus to use for validation
    :test_fraction: the fraction of the corpus to use for testing. Required that 1 - train_fraction - validation_fraction = test_fraction.
    """
    
    assert train_fraction + validation_fraction + test_fraction == 1, "train_fraction + validation_fraction + test_fraction must be 1"
    
    if images_to_use == None:
        images_to_use = len(data)
    num_train_images = int(images_to_use * train_fraction)
    num_valid_images = int(images_to_use * validation_fraction)
    num_test_images = images_to_use - num_train_images - num_valid_images
    
    
    train_uids = data['Image Index'].sample(num_train_images, replace = False)
    consideration = pd.concat([data['Image Index'], train_uids]).drop_duplicates(keep = False)
    valid_uids = consideration.sample(num_valid_images, replace = False)
    consideration = pd.concat([consideration, valid_uids]).drop_duplicates(keep = False)
    test_uids = consideration.sample(num_test_images, replace = False)
    
    if verbose:
        print_summary(set(train_uids), set(valid_uids), set(test_uids))
    return (set(train_uids), set(valid_uids), set(test_uids))

def make_dataframe(uid_set):
    data = []
    for uid in uid_set:
        data.append([uid, generate_target_vector(uid)])
    df = pd.DataFrame(data, columns = ['ImgUID', 'Labels'])
    #print(df)
    return df

def make_train_folder(set_to_use, relative_path_to_use):
    try:
        os.makedirs(relative_path_to_use)
        for label in labels():
            p = relative_path_to_use + '/' + label.name
            print("p is" , p)
            os.mkdir(p)
    except OSError:
        print("Failed to make new directory at:" , relative_path_to_use)
    else:
        print("Successfully made directory at:" , relative_path_to_use)
        
    img_loc = '../../images/'
    for uid in set_to_use:
        for label in image_uid_to_disease_mapping[uid]:
            print("label is" , label)
            copy(img_loc + uid, relative_path_to_use + '/' + label.name)
    
def clean_up(relative_path_to_use):
    try:
        rmtree(relative_path_to_use)
    except Exception:
        print("Failed to remove directory at:" , relative_path_to_use)
    else:
        print("Successfully removed directory at:" , relative_path_to_use)
    
if __name__ == "__main__":
    train, valid, test = generate_uid_sets(1000)
    analyze_dataset(train)




