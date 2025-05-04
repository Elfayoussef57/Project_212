from typing import Tuple,Optional
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class DataPreprocessor:
    """
    Classe pour prétraiter et augmenter les données d'images.
    """
    
    @staticmethod
    def create_augmented_generator() -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Crée un générateur de données avec augmentation.
        
        Returns:
            ImageDataGenerator: Générateur avec augmentation
        """
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    @staticmethod
    def create_basic_generator() -> tf.keras.preprocessing.image.ImageDataGenerator:
        """
        Crée un générateur de données sans augmentation.
        
        Returns:
            ImageDataGenerator: Générateur basique
        """
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    @staticmethod
    def get_data_generators(
        train_dir: str,
        val_dir: str,
        batch_size: int = 32,
        target_size: Tuple[int, int] = (224, 224),
        augment_train: bool = True
    ) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, 
               tf.keras.preprocessing.image.DirectoryIterator]:
        """
        Crée des générateurs pour les données d'entraînement et de validation.
        
        Args:
            train_dir: Chemin vers les données d'entraînement
            val_dir: Chemin vers les données de validation
            batch_size: Taille des lots
            target_size: Dimensions cibles des images
            augment_train: Si True, applique l'augmentation aux données d'entraînement
            
        Returns:
            Tuple de (train_generator, val_generator)
        """
        # Création des générateurs
        train_datagen = (DataPreprocessor.create_augmented_generator() 
                        if augment_train 
                        else DataPreprocessor.create_basic_generator())
        
        val_datagen = DataPreprocessor.create_basic_generator()
        
        # Génération des données
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator

def plot_examples(
    generator: tf.keras.preprocessing.image.DirectoryIterator,
    num_examples: int = 8,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Affiche des exemples d'images à partir d'un générateur.
    
    Args:
        generator: Générateur d'images
        num_examples: Nombre d'exemples à afficher
        title: Titre du plot
        figsize: Taille de la figure
    """
    # Récupère un batch d'images
    images, labels = next(generator)
    class_names = list(generator.class_indices.keys())
    
    # Limite le nombre d'exemples
    num_examples = min(num_examples, images.shape[0])
    images = images[:num_examples]
    labels = labels[:num_examples]
    
    # Crée la figure
    plt.figure(figsize=figsize)
    if title:
        plt.suptitle(title, fontsize=16)
    
    # Affiche chaque image
    for i in range(num_examples):
        plt.subplot((num_examples // 4) + 1, 4, i + 1)
        plt.imshow(images[i])
        class_idx = np.argmax(labels[i])
        plt.title(f"Class: {class_names[class_idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Configuration
    PATHS = {
        'train': '/workspaces/Project_212/data/train',
        'val': '/workspaces/Project_212/data/test'
    }
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    
    # 1. Avec augmentation des données
    train_gen_aug, val_gen_aug = DataPreprocessor.get_data_generators(
        PATHS['train'], 
        PATHS['val'],
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        augment_train=True
    )
    
    # 2. Sans augmentation des données
    train_gen, val_gen = DataPreprocessor.get_data_generators(
        PATHS['train'], 
        PATHS['val'],
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        augment_train=False
    )
    
    # Affichage des informations
    print("\nInformations sur les données:")
    print(f"Classes: {train_gen.class_indices}")
    print(f"Nombre de classes: {len(train_gen.class_indices)}")
    print(f"Exemples d'entraînement (avec augmentation): {train_gen_aug.samples}")
    print(f"Exemples de validation: {val_gen.samples}")

    # Visualisation des exemples
    print("\nVisualisation des exemples sans augmentation:")
    plot_examples(train_gen, title="Exemples originaux (sans augmentation)")
    
    print("\nVisualisation des exemples avec augmentation:")
    plot_examples(train_gen_aug, title="Exemples avec augmentation")


if __name__ == "__main__":
    main()