import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import nvtx

class Preprocessor():
    """
    This class does some necessary data preprocessing for the Cifar-10 benchmark.
    """

    @nvtx.annotate("main->preprocessing->init_preprocessor", color="yellow")
    def __init__(self, trainX, testX, trainY, testY, BATCH_SIZE, EPOCHS, num_workers, worker_index):
        """
        Initialization of the Preprocessor.
        """
        
        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY
        self.train_ds = None
        self.val_ds = None
        self.data_augmentation = None
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.num_workers = num_workers
        self.worker_index = worker_index

    @nvtx.annotate("main->preprocessing->preprocess", color="yellow")
    def preprocess(self):
        """
        This functions runs all necessary preprocessing steps after each other finally it returns the preprocessed training and test data.
        """

        self.define_data_augmentation_layers()
        self.encode_labels()
        self.normalize_data()
        self.convert_to_tfdataset()
        self.create_input_pipeline()
        return self.train_ds, self.val_ds

    @nvtx.annotate("main->preprocessing->preprocess->define_data_augmentation_layers", color="yellow")
    def define_data_augmentation_layers(self):
        """
        This function defines the sequential layers used for data augmentation.
        """

        self.data_augmentation = tf.keras.Sequential([
          layers.experimental.preprocessing.RandomFlip("horizontal"),
        ])

    @nvtx.annotate("main->preprocessing->preprocess->encode_labels", color="yellow")
    def encode_labels(self):
        """
        This function on hot encodes the labels of the data set.
        """

        # one hot encode the labels from one digit integer to e.g. [0 0 1 0 0]
        self.trainY = tf.one_hot(self.trainY, 10)
        self.testY = tf.one_hot(self.testY, 10)


    @nvtx.annotate("main->preprocessing->preprocess->normalize_data", color="yellow")
    def normalize_data(self):
        """
        This function normalizes the image data to the values [0-1]
        """

        # Convert from integers to floats
        self.trainX = self.trainX.astype('float32')
        self.testX = self.testX.astype('float32')

        # Normalize to range 0-1
        X_train_mean = np.mean(self.trainX, axis=(0,1,2))
        X_train_std = np.std(self.trainX, axis=(0,1,2))
        self.trainX = (self.trainX - X_train_mean) / X_train_std
        self.testX = (self.testX - X_train_mean) / X_train_std

    @nvtx.annotate("main->preprocessing->preprocess->convert_to_tfdataset", color="yellow")
    def convert_to_tfdataset(self):
        """
        This function converts the data sets to tf.Datasets.
        """

        # Change to tf dataset
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.trainX, self.trainY))
        self.val_ds = tf.data.Dataset.from_tensor_slices((self.testX, self.testY))

    @nvtx.annotate("main->preprocessing->preprocess->create_input_pipeline", color="yellow")
    def create_input_pipeline(self):
        """
        This function creates the input pipeline for training and testing.
        """
        
        # INFO: buffer size needs to be bigger or equal to sample amount in training data set
        # INFO: we have 5000 samples * 10 classes = 50000
        # INFO: reshuffle will produce different oder for each epoch
        
        # shard data so all workers will use different data
        self.train_ds = self.train_ds.shard(self.num_workers, self.worker_index)
        self.train_ds = self.train_ds.repeat(self.EPOCHS)

        # calculate buffer size for shuffle operation
        bsize = int((50000/self.num_workers)+1000)
        
        self.train_ds = self.train_ds.shuffle(buffer_size=bsize, reshuffle_each_iteration=True)
        
        # batch training data
        self.train_ds = self.train_ds.batch(self.BATCH_SIZE)

        # data augmentation pipeline
        self.train_ds = self.train_ds.map(lambda x, y: (self.data_augmentation(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_ds.cache()
        self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        # batch validation data as well or we will get an error about dimensions
        self.val_ds = self.val_ds.batch(self.BATCH_SIZE)
