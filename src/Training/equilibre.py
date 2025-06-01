from sklearn.model_selection import train_test_split
import os
import shutil

# Dossiers d'origine
base_dir = "../../data/Chest X_ray"
train_dir = os.path.join(base_dir, "train")

# Créer dossiers de destination
new_train = os.path.join(base_dir, "train_new")
new_val   = os.path.join(base_dir, "val_new")
os.makedirs(new_train, exist_ok=True)
os.makedirs(new_val, exist_ok=True)

# Parcours des classes
for cls in ["NORMAL", "PNEUMONIA"]:
    src = os.path.join(train_dir, cls)
    files = [os.path.join(src, f) for f in os.listdir(src)]
    train_files, val_files = train_test_split(
        files, test_size=0.15, stratify=[cls]*len(files), random_state=42
    )
    # Déplacer dans train_new et val_new
    for f in train_files:
        dst = os.path.join(new_train, cls, os.path.basename(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)
    for f in val_files:
        dst = os.path.join(new_val, cls, os.path.basename(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)