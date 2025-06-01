from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
from data_analysis import data_generator

img_path = "../Downloads/Project-212/data/test/NORMAL/IM-0011-0001-0001.jpeg"  # Remplacez par le chemin de votre image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # forme : (1, 224, 224, 3)

model = load_model("../../models/model_vgg.h5")
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Charger le générateur de test pour obtenir les noms de classes
test_generator = data_generator("../../data/Chest X_ray/test", batch_size=1, img_size=224, shuffle=False)

# Afficher le nom de la classe
class_labels = {v: k for k, v in test_generator.class_indices.items()}
print(f"Classe prédite : {class_labels[predicted_class]}")

plt.imshow(img)
plt.title(f"Prédiction : {class_labels[predicted_class]}")
plt.axis('off')
plt.show()
