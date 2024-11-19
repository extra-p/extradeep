import tensorflow as tf
import os

from create_datasets import DataLoader
from preprocessing import Preprocessor
from resnets import ResNet
from cnns import CNN
from resnets2 import cifar_resnet20, cifar_resnet32, cifar_resnet44, cifar_resnet56, cifar_resnet110, cifar_resnet164, cifar_resnet1001

import horovod
import horovod.tensorflow as hvd

class CIFAR10Benchmark():
    """
    CIFAR-10 Benchmark class. Runs simple distributed deep learning benchmark using tensorflow and horovod on the CIFAR-10 dataset training a specified network architecture.
    This sample code uses a simple user defined training function. For easier readability the code is structured in a object oriented way, and separated into different files.
    """

    def __init__(self, NETWORK, EPOCHS, BATCH_SIZE, VERBOSE, WEAK_SCALING):
        """
        Initialization function of the benchmark. Sets the configurable application parameters and hyperparameters for training.
        """

        # application configuration parameters
        self.NETWORK = NETWORK
        self.VERBOSE = VERBOSE
        self.WEAK_SCALING = WEAK_SCALING

        # set environment variable, prevent tf.data from interfering the threads that launch kernels on the GPU
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

        # Initialize horovod
        hvd.init()

        # hyperparameters for training
        self.BATCH_SIZE = BATCH_SIZE   # use batch size of 32 per gpu
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = 0.001
        self.MOMENTUM = 0.9
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.WEIGHT_DECAY = 0.0001 * self.BATCH_SIZE

        # variable placeholders
        self.testX = None
        self.testY = None
        self.trainX = None
        self.trainY = None
        self.NUM_CLASSES = 10
        self.training_callbacks = None
        self.evaluation_callbacks = None
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.loss = None
        self.opt = None
        self.train_acc_metric = None
        self.val_acc_metric = None

        self.TEST_LOSS = None
        self.TEST_ACC = None

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        # DEBUG CODE, only print on worker 0
        if hvd.rank() == 0:
            if self.VERBOSE == True:
                print("Tensorflow version:", tf.__version__)
                print("Number of GPUs available:", len(tf.config.experimental.list_physical_devices("GPU")))
                print("HVD RANKS:",hvd.size())

    def load_data(self):
        """
        This function loads the CIFAR-10 data for the benchmark from files in the cifar-10-data folder. It also validates the number of image classes.
        """

        # init the data loader
        data_loader = DataLoader()

        # load the test data
        self.testX, self.testY = data_loader.get_evaluation_data()

        # load the training data
        self.trainX, self.trainY = data_loader.get_training_data()

    def preprocessing(self):
        """
        This functions does the preprocessing of the data and builds the input pipeline for the training process.
        """

        # init the preprocessor
        preprocessor = Preprocessor(self.trainX, self.testX, self.trainY, self.testY, self.BATCH_SIZE, self.EPOCHS, hvd.size(), hvd.rank())

        # runs all preprocessing steps
        self.train_ds, self.val_ds = preprocessor.preprocess()

    def define_network(self):
        """
        This function defines the network architecture.
        """

        if self.NETWORK == "resnet18":
            resnet = ResNet(weigt_decay=self.WEIGHT_DECAY)
            self.model = resnet.define_resnet18(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES)

        elif self.NETWORK == "resnet20":
            self.model = cifar_resnet20(l2_reg=self.WEIGHT_DECAY)

        elif self.NETWORK == "resnet32":
            self.model = cifar_resnet32(l2_reg=self.WEIGHT_DECAY)
        
        elif self.NETWORK == "resnet34":
            resnet = ResNet(weigt_decay=self.WEIGHT_DECAY)
            self.model = resnet.define_resnet34(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES)

        elif self.NETWORK == "resnet44":
            self.model = cifar_resnet44(l2_reg=self.WEIGHT_DECAY)

        elif self.NETWORK == "resnet50":
            resnet = ResNet(weigt_decay=self.WEIGHT_DECAY)
            self.model = resnet.define_resnet50(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES)

        elif self.NETWORK == "resnet56":
            self.model = cifar_resnet56(l2_reg=self.WEIGHT_DECAY)

        elif self.NETWORK == "resnet101":
            resnet = ResNet(weigt_decay=self.WEIGHT_DECAY)
            self.model = resnet.define_resnet101(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES)

        elif self.NETWORK == "resnet110":
            self.model = cifar_resnet110('original', shortcut_type='A', l2_reg=self.WEIGHT_DECAY)
            
        elif self.NETWORK == "resnet152":
            resnet = ResNet(weigt_decay=self.WEIGHT_DECAY)
            self.model = resnet.define_resnet152(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES)

        elif self.NETWORK == "resnet164":
            self.model = cifar_resnet164(l2_reg=self.WEIGHT_DECAY)

        elif self.NETWORK == "resnet1001":
            self.model = cifar_resnet1001(l2_reg=self.WEIGHT_DECAY)

        elif self.NETWORK == "cnn5":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=5)

        elif self.NETWORK == "cnn10":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=10)

        elif self.NETWORK == "cnn15":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=15)

        elif self.NETWORK == "cnn20":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=20)

        elif self.NETWORK == "cnn25":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=25)

        elif self.NETWORK == "cnn30":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=30)

        elif self.NETWORK == "cnn35":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=35)

        elif self.NETWORK == "cnn40":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=40)

        elif self.NETWORK == "cnn45":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=45)

        elif self.NETWORK == "cnn50":
            cnn = CNN(weigt_decay=self.WEIGHT_DECAY)
            self.model = cnn.define_cnn(input_shape=(32,32,3), num_outputs=self.NUM_CLASSES, num_layers=50)

        self.loss = tf.losses.SparseCategoricalCrossentropy()
        # Horovod: adjust learning rate based on number of GPUs.
        self.opt = tf.optimizers.Adam(self.LEARNING_RATE * hvd.size())

        # Prepare the metrics.
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def training_step(self, images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = self.model(images, training=True)
            loss_value = self.loss(labels, probs)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.opt.variables(), root_rank=0)

        self.train_acc_metric.update_state(labels, probs)

        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        self.val_acc_metric.update_state(y, val_logits)

    def train(self):
        """
        Training loop.
        """

        # Calculate the number of training and validation steps per epoch
        #train_size = self.trainX.shape[0] / hvd.size()
        #validation_size = self.testX.shape[0]
        #training_steps = train_size // self.BATCH_SIZE
        #validation_steps = validation_size // self.BATCH_SIZE
        #training_steps = 5
        #validation_steps = 5

        # train the model
        for epoch in range(self.EPOCHS):
            if hvd.rank() == 0:
                if self.VERBOSE == True:
                    print("\nStart of epoch %d" % (epoch + 1,))
            if self.WEAK_SCALING == True:
                for batch, (images, labels) in enumerate(self.train_ds.take(10000)):
                    loss_value = self.training_step(images, labels, batch == 0)

                    if batch % 100 == 0 and hvd.rank() == 0:
                        if self.VERBOSE == True:
                            print('Step #%d\tLoss: %.4f' % (batch, loss_value))
                            print("Seen so far: %s samples" % ((batch + 1) * self.BATCH_SIZE))

                # Display metrics at the end of each epoch.
                train_acc = self.train_acc_metric.result()
                if hvd.rank() == 0:
                    if self.VERBOSE == True:
                        print("Training acc over epoch: %.4f" % (float(train_acc),))

                # Reset training metrics at the end of each epoch
                self.train_acc_metric.reset_states()

                # Run a validation loop at the end of each epoch.
                for x_batch_val, y_batch_val in self.val_ds:
                    self.test_step(x_batch_val, y_batch_val)

                val_acc = self.val_acc_metric.result()
                self.val_acc_metric.reset_states()
                if hvd.rank() == 0:
                    if self.VERBOSE == True:
                        print("Validation acc: %.4f" % (float(val_acc),))
            else:
                for batch, (images, labels) in enumerate(self.train_ds.take(10000 // hvd.size())):
                    loss_value = self.training_step(images, labels, batch == 0)

                    if batch % 100 == 0 and hvd.rank() == 0:
                        if self.VERBOSE == True:
                            print('Step #%d\tLoss: %.4f' % (batch, loss_value))
                            print("Seen so far: %s samples" % ((batch + 1) * self.BATCH_SIZE))

                # Display metrics at the end of each epoch.
                train_acc = self.train_acc_metric.result()
                if hvd.rank() == 0:
                    if self.VERBOSE == True:
                        print("Training acc over epoch: %.4f" % (float(train_acc),))

                # Reset training metrics at the end of each epoch
                self.train_acc_metric.reset_states()

                # Run a validation loop at the end of each epoch.
                for x_batch_val, y_batch_val in self.val_ds:
                    self.test_step(x_batch_val, y_batch_val)

                val_acc = self.val_acc_metric.result()
                self.val_acc_metric.reset_states()
                if hvd.rank() == 0:
                    if self.VERBOSE == True:
                        print("Validation acc: %.4f" % (float(val_acc),))

    def evaluation(self):
        """
        Evaluation loop.
        """

        # evaluate the trained model
        for x_batch_val, y_batch_val in self.val_ds:
            self.test_step(x_batch_val, y_batch_val)

        val_acc = self.val_acc_metric.result()
        self.val_acc_metric.reset_states()
        if hvd.rank() == 0:
            if self.VERBOSE == True:
                print("Final validation acc: %.4f" % (float(val_acc),))
