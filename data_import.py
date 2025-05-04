import tensorflow as tf
import os

print(os.listdir('../input'))
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))
