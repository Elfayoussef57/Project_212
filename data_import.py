import tensorflow as tf
import os
import sys

def load_data(data_dir):
        """
        Load the data from the dat folder directory.

        Arguments:   data_dir the path to the data directory (folder: string)
        
        Return values: train_dataset, test_dataset, validation_dataset
        """
        # Define the data transformations
        TEST = os.path.join(data_dir, 'test')
        TRAIN = os.path.join(data_dir, 'train')
        VAL =   os.path.join(data_dir, 'val')
        try:
                BATCH_SIZE = 32
                IMG_SIZE = (160, 160)
                # Load the test dataset
                test_dataset = tf.keras.preprocessing.image_dataset_from_directory(TEST,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)
                train_dataset = tf.keras.preprocessing.image_dataset_from_directory(TRAIN,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE,
                                                        validation_split=0.2,
                                                        subset='training',
                                                        seed=42)
                validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(VAL,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE,
                                                        validation_split=0.2,
                                                        subset='validation',
                                                        seed=42)
        except Exception as e:
                # Handle any errors that occur during loading
                print(f"Error loading dataset: {e}")
                sys.exit(1)

        return train_dataset, test_dataset, validation_dataset

def main():

        data_dir = 'data'
        train_dataset, test_dataset, validation_dataset = load_data(data_dir)
        print("Data loaded successfully.")
        print(f"Train dataset: {train_dataset}")
        print(f"Test dataset: {test_dataset}")
        print(f"Validation dataset: {validation_dataset}")


if __name__ == "__main__":      
        main()

