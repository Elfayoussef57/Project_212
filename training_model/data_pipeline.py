import os, shutil
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import typing as tp

plt.style.use('ggplot')

labels = ['PNEUMONIA', 'NORMAL']
img_size = 128

def get_data(data_dir: str) -> np.ndarray:
    """
    Function to load and preprocess images from a directory.
    Args:
        data_dir (str): Path to the directory containing images.
    Returns:
        np.ndarray: Array of preprocessed images and their corresponding labels.
    """
    data = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp') 
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                if not img.lower().endswith(valid_exts):
                    print(f"Invalid file extension: {img}")
                    continue
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing {img}: {e}")
    return np.array(data, dtype=object)

def statistics_train(data: np.ndarray) -> None:
    """
    Function to display statistics of the dataset.
    Args:
        data (np.ndarray): Array of images and their corresponding labels.
    """
    listx = []
    for item in data:
        if item[1] == 0:
            listx.append("Pneumonia")
        else:
            listx.append("Normal")
    plt.figure(figsize=(10, 5))
    sns.countplot(x=listx)
    plt.title("Distribution of classes in the training set")
    plt.xlabel("Classes")
    plt.ylabel("Number of images")
    plt.savefig("distribution.png")
    plt.close()

def main():
    # Load and display sample images
    normal_dir = "../data/train/NORMAL"
    pneumonia_dir = "../data/train/PNEUMONIA"
    
    normal_images = os.listdir(normal_dir)
    pneumonia_images = os.listdir(pneumonia_dir)
    
    # Randomly select 9 images from each class
    normal_samples = random.sample(normal_images, 9)
    pneumonia_samples = random.sample(pneumonia_images, 9)

    # Plot Normal images
    plt.figure(figsize=(20, 10))
    for i, img_name in enumerate(normal_samples):
        plt.subplot(3, 3, i+1)
        img_path = os.path.join(normal_dir, img_name)
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.title("Normal X-ray")
    plt.tight_layout()
    plt.savefig("normal.png")
    plt.close()

    # Plot Pneumonia images
    plt.figure(figsize=(20, 10))
    for i, img_name in enumerate(pneumonia_samples):
        plt.subplot(3, 3, i+1)
        img_path = os.path.join(pneumonia_dir, img_name)
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.title("Pneumonia X-ray")
    plt.tight_layout()
    plt.savefig("pneumonia.png")
    plt.close()

    # Generate and save class distribution
    data = get_data("../data/train")
    statistics_train(data)

if __name__ == "__main__":
    main()