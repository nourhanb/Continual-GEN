from sklearn.model_selection import train_test_split
from torchvision import transforms
from glob import glob

import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm


def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    all_image_path_dmf = glob(os.path.join(image_paths, 'images','*.png'))

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(all_image_path_dmf))):
        img = cv2.imread(all_image_path_dmf[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

    return means,stdevs


def create_dataframes(data_dir_dmf):
    all_image_path_dmf = glob(os.path.join(data_dir_dmf, 'images','*.png'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path_dmf}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(data_dir_dmf, 'meta-dmf.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    print(df_original['duplicates'].value_counts())


    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # now we create a test set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_test = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    print(df_test.shape)

    # This set will be df_original excluding all rows that are in the test set
    # This function identifies if an image is part of the train or test set.
    def get_test_rows(x):
        # create a list of all the lesion_id's in the test set
        val_list = list(df_test['image_id'])
        if str(x) in val_list:
            return 'test'
        else:
            return 'train'

    # identify train and test rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_test'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_test'] = df_original['train_or_test'].apply(get_test_rows)
    # filter out train rows
    df_train_DONT = df_original[df_original['train_or_test'] == 'train']

    print(df_train_DONT['cell_type_idx'].value_counts())
    print(df_train_DONT['cell_type'].value_counts())

    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [2,1,1,5,1,4,2] ## for HAM: [19,12,5,54,0,5,45]
    for i in range(7):
        if data_aug_rate[i]:
            df_train_DONT=df_train_DONT.append([df_train_DONT.loc[df_train_DONT['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)

    train_class_counts = df_train_DONT['cell_type_idx'].value_counts()
    print("Train Set:")
    print(train_class_counts)
    test_class_counts = df_test['cell_type_idx'].value_counts()
    print("Test Set:")
    print(test_class_counts)
    df_train_DONT = df_train_DONT.reset_index()
    df_test = df_test.reset_index()


    return df_train_DONT, df_test


def create_transformations(input_size, norm_mean, norm_std):
    # define the transformation of the train images.
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    # define the transformation of the test images.
    test_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    
    return train_transform, test_transform