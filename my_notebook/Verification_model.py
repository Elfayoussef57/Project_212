import os
import numpy as np
import copy
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Logistic Regression Functions ===

def plot_learning_curve(costs, learning_rate):
    plt.figure(figsize=(8, 5))
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title(f'Learning rate = {learning_rate}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Matrice de confusion ===
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Radio", "Radio"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Logistic Regression Implementation ===
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    dw = (np.dot(X, (A - Y).T)) / m
    db = (np.sum(A - Y)) / m
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = (A > 0.5).astype(float)
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    return d

# === Data loading and preprocessing ===

def load_images_from_csv(csv_path, root_dir, image_size=(64, 64), max_images=None):
    import csv
    image_list = []
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_images and i >= max_images:
                break
            img_path = os.path.join(root_dir, row['filepath'])
            label = int(row['label'])
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img)
                image_list.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Warning: skipping {img_path} due to error: {e}")
    X = np.array(image_list)
    Y = np.array(labels)
    return X, Y

def preprocess_data(X, Y):
    X_flatten = X.reshape(X.shape[0], -1).T   # (num_px*num_px*3, m)
    X_norm = X_flatten / 255.0
    Y_reshaped = Y.reshape(1, Y.shape[0])     # (1, m)
    return X_norm, Y_reshaped

# === Main training ===

def main():
    # Configuration
    csv_path = '../data/Verif_data/all_images_labels.csv'
    root_dir = '../data/Verif_data'
    image_size = (64, 64)

    print("Loading images and labels from CSV...")
    X, Y = load_images_from_csv(csv_path, root_dir, image_size)
    print(f"Loaded {X.shape[0]} images of size {X.shape[1:]}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train_prep, Y_train_prep = preprocess_data(X_train, Y_train)
    X_test_prep, Y_test_prep = preprocess_data(X_test, Y_test)

    print(f"Training data shape: {X_train_prep.shape}, labels shape: {Y_train_prep.shape}")
    print(f"Testing data shape: {X_test_prep.shape}, labels shape: {Y_test_prep.shape}")

    # === Entraînement ===
    d = model(X_train_prep, Y_train_prep, X_test_prep, Y_test_prep,
              num_iterations=2000, learning_rate=0.005, print_cost=True)

    print("Training finished!")
    print(f"Final train accuracy: {100 - np.mean(np.abs(d['Y_prediction_train'] - Y_train_prep)) * 100:.2f} %")
    print(f"Final test accuracy: {100 - np.mean(np.abs(d['Y_prediction_test'] - Y_test_prep)) * 100:.2f} %")
    np.savez('../models/logistic_model_weights.npz', w=d['w'], b=d['b'])
    print("Model weights saved to logistic_model_weights.npz")

    # === Visualisations ===
    plot_learning_curve(d["costs"], d["learning_rate"])
    plot_confusion_matrix(Y_test_prep.flatten(), d["Y_prediction_test"].flatten(), title="Test Set Confusion Matrix")

if __name__ == "__main__":
    main()