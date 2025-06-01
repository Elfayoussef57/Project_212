import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


def data_generator(data_path: str, batch_size: int = 32, img_size: int = 128, shuffle: bool=False) -> tf.data.Dataset:
    """
    Function to create a data generator for training and validation datasets.
    Args:
        data_path (str): Path to the directory containing images.
        batch_size (int): Size of the batches of data.
    Returns:
        tf.data.Dataset: A dataset object for training or validation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode="nearest"
    )
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    if data_path.__contains__("train"):
        data_gen = train_datagen.flow_from_directory(data_path,
                                 batch_size = batch_size,
                                 target_size=(img_size,img_size),
                                 class_mode = 'categorical',
                                 shuffle=shuffle,
                                 seed = 42,
                                 color_mode = 'rgb')
    elif data_path.__contains__("val"):
        data_gen = valid_datagen.flow_from_directory(data_path,
                                 batch_size = batch_size,
                                 target_size=(img_size,img_size),
                                 class_mode = 'categorical',
                                 shuffle=shuffle,
                                 seed = 42,
                                 color_mode = 'rgb')
    elif data_path.__contains__("test"):
        data_gen = test_datagen.flow_from_directory(data_path,
                                 batch_size = batch_size,
                                 target_size=(img_size,img_size),
                                 class_mode = 'categorical',
                                 shuffle=shuffle,
                                 seed = 42,
                                 color_mode = 'rgb')
    else:
        print("Invalid data path. Please provide a valid path.")
        return None
    return data_gen

def main():
        # Create data generators
        train_gen = data_generator("../../data/Chest X_ray/train")
        valid_gen = data_generator("../../data/Chest X_ray/val")
        test_gen = data_generator("../../data/Chest X_ray/test")
        
        #print(train_gen.class_indices)
        print("train dataset")
        print("Class indices:", train_gen.class_indices)
        # Get the number of classes
        num_classes = len(train_gen.class_indices)
        print("Number of classes:", num_classes)
        # Get the shape of the input data
        print("Input shape:", train_gen.image_shape)
        # Get the batch size
        print("Batch size:", train_gen.batch_size)

        #print("test dataset")
        print("Class indices:", test_gen.class_indices)
        # Get the number of classes
        num_classes = len(test_gen.class_indices)
        print("Number of classes:", num_classes)
        # Get the shape of the input data
        print("Input shape:", test_gen.image_shape)
        # Get the batch size
        print("Batch size:", test_gen.batch_size)

        #print("valid dataset")
        print("Class indices:", valid_gen.class_indices)
        # Get the number of classes
        num_classes = len(valid_gen.class_indices)
        print("Number of classes:", num_classes)
        # Get the shape of the input data
        print("Input shape:", valid_gen.image_shape)
        # Get the batch size
        print("Batch size:", valid_gen.batch_size)

if __name__ == "__main__":
    main()
