import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs détectés :", gpus)
if gpus:
    # Active la croissance dynamique de la mémoire
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    raise RuntimeError("Aucun GPU détecté – vérifie ton installation de CUDA/cuDNN et tensorflow-gpu")
