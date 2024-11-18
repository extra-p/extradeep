import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import nvtx

class Preprocessor():
    """
    This class does some necessary data preprocessing for the Cifar-10 benchmark.
    """

    @nvtx.annotate("main->preprocessing->init_preprocessor", color="yellow")
    def __init__(self, BATCH_SIZE, PROBLEM_SIZE, EPOCHS, num_workers, worker_index, train_ds, val_ds):
        """
        Initialization of the Preprocessor.
        """

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.data_augmentation = None
        self.BATCH_SIZE = BATCH_SIZE
        self.PROBLEM_SIZE = PROBLEM_SIZE
        self.EPOCHS = EPOCHS
        self.num_workers = num_workers
        self.worker_index = worker_index

    @nvtx.annotate("main->preprocessing->preprocess", color="yellow")
    def preprocess(self):
        """
        This functions runs all necessary preprocessing steps after each other finally it returns the preprocessed training and test data.
        """

        self.scale_problemsize()
        self.create_input_pipeline()
        return self.train_ds, self.val_ds

    @nvtx.annotate("main->preprocessing->scale_problemsize", color="yellow")
    def scale_problemsize(self):
        """
        This functions scales the problem size of the dataset.
        """

        # nothing to be done...
        if self.PROBLEM_SIZE == 200:
            pass
        # scale the data set
        else:
            #TODO: implement logic for scaling here...
            pass

    @nvtx.annotate("main->preprocessing->preprocess->create_input_pipeline", color="yellow")
    def create_input_pipeline(self):
        """
        This function creates the input pipeline for training and testing.
        """

        # INFO: buffer size needs to be bigger or equal to sample amount in training data set
        # INFO: we have 500 samples * 100 classes = 50000
        # INFO: reshuffle will produce different oder for each epoch

        # calculate buffer size depending on the problem size
        #buffer_max_size = self.PROBLEM_SIZE * 500

        # shard data so all workers will use different data
        self.train_ds = self.train_ds.shard(self.num_workers, self.worker_index)
        #self.train_ds = self.train_ds.repeat(self.EPOCHS)

        # calculate buffer size for shuffle operation
        #bsize = int((buffer_max_size/self.num_workers)+1000)

        #self.train_ds = self.train_ds.shuffle(buffer_size=bsize, reshuffle_each_iteration=True)

        # batch training data
        self.train_ds = self.train_ds.batch(self.BATCH_SIZE)

        # data augmentation pipeline
        #self.train_ds = self.train_ds.map(lambda x, y: (self.data_augmentation(x, training=True), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #self.train_ds.cache()
        #self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # batch validation data as well or we will get an error about dimensions
        self.val_ds = self.val_ds.batch(self.BATCH_SIZE)
