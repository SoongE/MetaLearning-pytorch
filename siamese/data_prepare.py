import os
import random
from glob import glob

import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
import wget


def move_image_to_processed_dir(alpha_list, img_dir, desc):
    for alpha in tqdm(alpha_list, desc=desc):
        write_dir1 = img_dir + '/' + os.path.basename(alpha) + '_'
        for char in (os.listdir(alpha)):
            write_dir2 = (write_dir1 + char)
            char_path = os.path.join(alpha, char)
            os.makedirs(write_dir2)
            for drawer in os.listdir(char_path):
                drawer_path = os.path.join(char_path, drawer)
                os.rename(drawer_path, os.path.join(write_dir2, drawer))


def prepare_data():
    background_dir = "data/unzip/background"
    evaluation_dir = "data/unzip/evaluation"
    processed_dir = "data/processed"

    random.seed(5)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if any([True for _ in os.scandir(processed_dir)]):
        return

    # Move 10 of evaluation image for getting more train set.
    if len(glob(evaluation_dir + '/*')) >= 20:
        for d in random.sample(glob(evaluation_dir + '/*'), 10):
            os.rename(d, os.path.join(background_dir, os.path.basename(d)))

    back_alpha = [x for x in glob(background_dir + '/*')]
    back_alpha.sort()

    # Split background data into train, validation data and make test data
    train_alpha = list(np.random.choice(back_alpha, size=30, replace=False))
    train_alpha = [str(x) for x in train_alpha]
    val_alpha = [x for x in back_alpha if x not in train_alpha]
    test_alpha = [x for x in glob(evaluation_dir + '/*')]
    test_alpha.sort()

    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')

    move_image_to_processed_dir(train_alpha, train_dir, 'train')
    move_image_to_processed_dir(val_alpha, val_dir, 'val')
    move_image_to_processed_dir(test_alpha, test_dir, 'test')


def download_omniglot_data():
    BASEDIR = os.path.dirname(os.path.realpath(__file__)) + "/data"

    # make directory
    if not os.path.exists(BASEDIR):
        os.mkdir(BASEDIR)
    if not os.path.exists(os.path.join(BASEDIR, 'unzip')):
        os.mkdir(os.path.join(BASEDIR, 'unzip'))
    if not os.path.exists(os.path.join(BASEDIR, 'raw')):
        os.mkdir(os.path.join(BASEDIR, 'raw'))

    # download zip file
    if not os.path.exists(BASEDIR + '/raw/images_background.zip'):
        print("download background image")
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip",
                      BASEDIR + '/raw')
    if not os.path.exists(BASEDIR + '/raw/images_evaluation.zip'):
        print("download evaluation image")
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
                      BASEDIR + '/raw')

    # if there are no unzipped files
    if not any([True for _ in os.scandir(os.path.join(BASEDIR, "unzip"))]):
        # unzip files
        for d in glob(BASEDIR + '/raw/*.zip'):
            zip_name = os.path.splitext(os.path.basename(d))[0]
            print(f'{zip_name}is being unzipped...', end="")
            with ZipFile(d, 'r') as zip_object:
                zip_object.extractall(BASEDIR + '/unzip/')
            print("success")

        # change folder name
        try:
            os.rename(BASEDIR + '/unzip/images_background', BASEDIR + '/unzip/background')
            os.rename(BASEDIR + '/unzip/images_evaluation', BASEDIR + '/unzip/evaluation')
        except FileNotFoundError as e:
            print(e)
