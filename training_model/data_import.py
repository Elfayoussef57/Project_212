import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

plt.style.use('ggplot')

def load_data(data_dir: str, labels: List[str], img_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels from directory structure
    Args:
        data_dir: Root directory containing subdirectories for each class
        labels: List of class names (subdirectory names)
        img_size: Target size for resizing images
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    images = []
    labels_list = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                images.append(resized_arr)
                labels_list.append(class_num)
            except Exception as e:
                print(f"Error loading {os.path.join(path, img)}: {e}")
    return np.array(images), np.array(labels_list)

def plot_sample_images(image_paths: List[str], titles: List[str], n_samples: int = 9, figsize: Tuple[int, int] = (20, 10),save_path: str = None) -> None:
    """
    Plot sample images in a grid
    Args:
        image_paths: List of paths to images
        titles: List of titles for the images
        n_samples: Number of samples to display
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    for i in range(min(n_samples, len(image_paths))):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(image_paths[i])
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.title(titles[i])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_class_distribution(labels: np.ndarray, class_names: List[str], figsize: Tuple[int, int] = (10, 5), save_path: str = None) -> None:
    """
    Plot class distribution as countplot
    Args:
        labels: Array of numerical labels
        class_names: List of class names corresponding to numerical labels
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    label_names = [class_names[label] for label in labels]
    sns.countplot(x=label_names)
    plt.title(f"Class Distribution ({' vs '.join(class_names)})")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Configuration
    DATA_DIR = "/workspaces/Project_212/data"
    LABELS = ['PNEUMONIA', 'NORMAL']
    IMG_SIZE = 128
    
    # Paths
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")
    normal_dir = os.path.join(train_dir, "NORMAL")

    # Load data
    train_images, train_labels = load_data(train_dir, LABELS, IMG_SIZE)
    test_images, test_labels = load_data(test_dir, LABELS, IMG_SIZE)

    print(f"Total labels: {len(train_labels)}")

    # Get sample image paths
    pneumonia_images = [os.path.join(pneumonia_dir, img) for img in os.listdir(pneumonia_dir)]
    normal_images = [os.path.join(normal_dir, img) for img in os.listdir(normal_dir)]

    # Create and save visualizations
    plot_sample_images(pneumonia_images,  ["Pneumonia X-ray"] * len(pneumonia_images), save_path="pneumonia_samples.png")
    
    plot_sample_images(normal_images, ["Normal X-ray"] * len(normal_images), save_path="normal_samples.png")
    
    plot_class_distribution(train_labels, ["Pneumonia", "Normal"], save_path="class_comparison.png")

if __name__ == "__main__":
    main()