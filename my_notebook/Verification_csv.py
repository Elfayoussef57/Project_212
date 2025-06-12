import os
from pathlib import Path
import csv

# Configuration
root_dir = '../data/Verif_data'  # Répertoire principal (pour calculer les chemins relatifs)
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

def collect_images_and_labels():
    """
    Collecte les chemins des images dans train_images et test_images
    à la fois dans ../data et ../data2. Attribue le label 1 si l'image vient de ../data2, sinon 0.
    """
    dataset = []

    # Dossiers à parcourir avec leur label associé
    paths_and_labels = [
        ('../data/Verif_data/Radiographie/train_images', 0),
        ('../data/Verif_data/Radiographie/test_images', 0),
        ('../data/Verif_data/Garbage/train_images', 1),
        ('../data/Verif_data/Garbage/test_images', 1),

    ]

    for path, label in paths_and_labels:
        if not os.path.exists(path):
            print(f"Chemin non trouvé : {path}")
            continue
        for root, _, files in os.walk(path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    full_path = os.path.join(root, file)
                    # Calcul du chemin relatif par rapport à root_dir
                    rel_path = os.path.relpath(full_path, start=root_dir)
                    dataset.append((rel_path, label))

    return dataset

def write_csv(dataset, output_file='../data/Verif_data/all_images_labels.csv'):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])
        for filepath, label in dataset:
            writer.writerow([filepath, label])

if __name__ == "__main__":
    dataset = collect_images_and_labels()
    print(f"Total images collected: {len(dataset)}")
    write_csv(dataset)
    print("CSV file 'all_images_labels.csv' created.")
