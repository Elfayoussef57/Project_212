import matplotlib.pyplot as plt
import json
import numpy as np
import os
import tensorflow as tf


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
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE,
                                                validation_split=0.2,
                                                subset='training',
                                                seed=42)
        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE,
                                                validation_split=0.2,
                                                subset='validation',
                                                seed=42)

        
      
    


if __name__ == "__main__":
        main()