import pickle
import numpy as np
import nvtx
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

class DataLoader:
    """
    This class handles loading the imagenet data.
    """

    @nvtx.annotate("main->load_data->init_data_loader", color="blue")
    def __init__(self):
        """
        Initialization of the DataLoader class.
        """

        # images have the shape [3,64,64]
        self.data_dir = "imagenet"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.val_dir = os.path.join(self.data_dir, "val")

    @nvtx.annotate("main->load_data->load_data", color="blue")
    def load_data(self, problem_size):
        """
        Get the data for training and evaluation.
        """

        # check if tfds exist
        train_ds = None
        val_ds = None

        training_save_path = os.path.join(self.data_dir, "training_data")
        validation_save_path = os.path.join(self.data_dir, "validation_data")

        if os.path.exists(training_save_path) == True and os.path.exists(validation_save_path) == True:
            exists = True
        else:
            exists = False

        if exists == True:

            train_ds = tf.data.experimental.load(training_save_path)
            val_ds = tf.data.experimental.load(validation_save_path)

        else:

            # load the training data and create a tf dataset
            classes = {}
            training_data = []
            labels = []

            counter = 0
            for item in os.listdir(self.train_dir):

                image_class = str(item)
                classes[counter] = image_class

                class_dir = os.path.join(self.train_dir, str(item+"/images/"))

                for pic in os.listdir(class_dir):

                    pic_dir = os.path.join(class_dir, pic)

                    raw = tf.io.read_file(pic_dir)
                    image = tf.image.decode_jpeg(raw, channels=3)
                    training_data.append(image)
                    labels.append(counter)

                counter += 1


            trainY = np.array(labels)
            trainX = np.array(training_data)

            trainY = tf.one_hot(trainY, problem_size)

            trainX = trainX.astype('float32')

            X_train_mean = np.mean(trainX, axis=(0,1,2))
            X_train_std = np.std(trainX, axis=(0,1,2))
            trainX = (trainX - X_train_mean) / X_train_std

            train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))

            # load the validation data and create a tf dataset
            meta_dir = os.path.join(self.val_dir, "val_annotations.txt")
            with open(meta_dir) as f:
                lines = f.readlines()

            linelist = []
            for item in lines:
                linelist.append(item.split("\t"))

            index = {}
            paths = []
            for item in linelist:
                path = item[0]
                label = item[1]
                path = os.path.join(self.val_dir, "images/"+path)
                index[path] = label
                paths.append(path)

            testing_data = []
            labels = []
            for item in paths:

                raw = tf.io.read_file(item)
                image = tf.image.decode_jpeg(raw, channels=3)
                testing_data.append(image)
                path = index[item]
                value = list(classes.keys())[list(classes.values()).index(path)]
                labels.append(value)

            testY = np.array(labels)
            testX = np.array(testing_data)

            testY = tf.one_hot(testY, problem_size)

            testX = testX.astype('float32')

            X_train_mean = np.mean(testX, axis=(0,1,2))
            X_train_std = np.std(testX, axis=(0,1,2))
            testX = (testX - X_train_mean) / X_train_std

            val_ds = tf.data.Dataset.from_tensor_slices((testX, testY))

            size = (224, 224)
            train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
            val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y))

            training_save_path = os.path.join(self.data_dir, "training_data")

            tf.data.experimental.save(train_ds, training_save_path)

            validation_save_path = os.path.join(self.data_dir, "validation_data")

            tf.data.experimental.save(val_ds, validation_save_path)
                
        return train_ds, val_ds

