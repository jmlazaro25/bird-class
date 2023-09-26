import os
import numpy as np
from glob import glob
from skimage.io import imread
from skimage.transform import resize

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_IMAGE_HEIGHT = 224
RAW_IMAGE_WIDTH = 224
RAW_IMAGE_CHANNELS = 3

def get_classes():
    return {
                i: os.path.split(name)[-1] for i, name in
                enumerate(sorted(glob(os.path.join(DATA_DIR, 'test', '*'))))
           }

def load_set(set):
    images = []
    labels = []
    for c_i, c in get_classes().items():
        for image_file in glob(os.path.join(DATA_DIR, set, c, '*')):
            img = imread(image_file)
            if img.shape != (RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, RAW_IMAGE_CHANNELS):
                height, width = img.shape[0:2]
                if height < width:
                    img = img[:, height, :]
                elif height > width:
                    img = img[:width, :, :]
                img = resize(
                    img,
                    (RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, RAW_IMAGE_CHANNELS)
                )
            images.append(img)
            labels.append(c_i)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

import tensorflow as tf

batch_size = 32
dataset_train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    batch_size=batch_size,
    image_size=(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH),
    crop_to_aspect_ratio=True
)
print(dataset_train)
