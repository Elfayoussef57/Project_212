import matplotlib.pyplot as plt
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

def import_data(path):
        path_dir = path
        TEST = 'test'
        TRAIN = 'train'
        VAL ='val'
        return TEST, TRAIN, VAL


def main():
        BATCH_SIZE = 32
        IMG_SIZE = (160, 160)
        directory = "../../chest_xray"

        
      
    


if __name__ == "__main__":
        main()