import tensorflow as tf
import sys
import os
import math
import nvtx
import horovod.tensorflow.keras as hvd
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.callbacks import EarlyStopping
from callbacks import PrintLearningRate
from callbacks import IndicateEpochStartEnd
from callbacks import IndicateTestStartEnd
from callbacks import IndicateTrainingStepStartEnd
from callbacks import IndicateTrainingStartEnd
from callbacks import IndicateTestingStepStartEnd
from create_datasets import DataLoader
from preprocessing import Preprocessor
from logger import Logger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

class SpeechBenchmark():
    """
    SpeechCommands Benchmark class. Runs simple distributed deep learning benchmark on the SpeechCommands dataset training a specified CNN network.
    """

    @nvtx.annotate("main->init", color="green")
    def __init__(self, REPETITION, NR_PROCESSES, EPOCHS, BATCH_SIZE, TF_PROFILER, VERBOSE, DETERMINISM, MEMORY_GROWTH, MIXED_PRECISION, TENSOR32_EXECUTION, USE_MINI):
        """
        Initialization function of the benchmark. Sets the configurable application parameters and hyperparameters for training.
        """

        # application configuration parameters
        self.JOBNAME = "SpeechCommands"
        self.REPETITION = REPETITION
        self.TF_PROFILER = TF_PROFILER
        self.VERBOSE = VERBOSE
        self.USE_MINI = USE_MINI

        self.FOLDER_NAME = self.JOBNAME + "_p" + str(NR_PROCESSES) + "_r" + str(REPETITION)

        # set environment variable, prevent tf.data from interfering the threads that launch kernels on the GPU
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

        # Initialize horovod
        hvd.init()

        # hyperparameters for training
        self.BATCH_SIZE = BATCH_SIZE   # use batch size of 32 per gpu
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = 0.01
        # Horovod: adjust learning rate based on number of GPUs.
        self.LEARNING_RATE = self.LEARNING_RATE * math.sqrt(hvd.size())
        self.MOMENTUM = 0.9
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.WEIGHT_DECAY = 0.0001 * self.BATCH_SIZE

        # variable placeholders
        self.testX = None
        self.testY = None
        self.trainX = None
        self.trainY = None
        self.NUM_CLASSES = None
        self.training_callbacks = None
        self.evaluation_callbacks = None
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.history = None
        self.TEST_LOSS = None
        self.TEST_ACC = None

        self.train_files = None
        self.val_files = None
        self.test_files = None
        self.input_shape = None
        self.num_labels = None
        self.commands = None
        self.norm_layer = None

        # create new directory for experiment data
        if hvd.rank() == 0:
            os.mkdir(self.FOLDER_NAME)

        checkpoints_dir = self.FOLDER_NAME + "/checkpoints/"

        # create new directory for tf
        if hvd.rank() == 0:
            os.mkdir(checkpoints_dir)

        # Define the checkpoint directory to store the checkpoints
        self.checkpoint_dir = self.FOLDER_NAME+"/checkpoints/"

        # Name of the checkpoint files
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=30024)])

        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        # set additional flags for tf-code
        if DETERMINISM == True:
            # set op determinism for kernel execution on gpus
            # see https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
            tf.keras.utils.set_random_seed(1)
            tf.config.experimental.enable_op_determinism()
        if MIXED_PRECISION == True:
            # set the mixed precision to use fp16 and fp32 operations
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
        if TENSOR32_EXECUTION == True:
            # enable the tensor float32 hardware support for nvidia ampere gpus
            # makes certain ops like matrix muliplications run much faster on gpus with reduced precision
            tf.config.experimental.enable_tensor_float_32_execution(True)

        # Horovod: write logs on worker 0 only
        if self.VERBOSE == 1 or self.VERBOSE == 2:
            self.VERBOSE = 1 if hvd.rank() == 0 else 0

        # DEBUG CODE, only print on worker 0
        if hvd.rank() == 0:
            print("Tensorflow version:", tf.__version__)
            print("Number of GPUs available:", len(tf.config.experimental.list_physical_devices("GPU")))
            print("HVD RANKS:",hvd.size())

        # Set seed for experiment reproducibility
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

    @nvtx.annotate("main->load_data", color="blue")
    def load_data(self):
        """
        This function loads the CIFAR-10 data for the benchmark from files in the cifar-10-data folder. It also validates the number of image classes.
        """

        # init the data loader
        data_loader = DataLoader(self.USE_MINI)

        data_loader.download_data()
        self.commands = data_loader.get_meta_data()
        self.train_files, self.val_files, self.test_files = data_loader.get_training_data()
        self.NUM_CLASSES = data_loader.get_num_classes()

    @nvtx.annotate("main->preprocessing", color="yellow")
    def preprocessing(self):
        """
        This functions does the preprocessing of the data and builds the input pipeline for the training process.
        """
        
        # init the preprocessor
        preprocessor = Preprocessor(self.train_files, self.val_files, self.test_files, self.BATCH_SIZE, self.EPOCHS, hvd.size(), hvd.rank(), self.NUM_CLASSES, self.commands)

        # runs all preprocessing steps
        self.train_ds, self.val_ds, self.test_ds, self.input_shape, self.num_labels, self.norm_layer = preprocessor.preprocess()

    @nvtx.annotate("main->create_callbacks_training", color="orange")
    def create_callbacks_training(self):
        """
        This function creates the callbacks for the training process.
        """

        # Callbacks for training process
        self.training_callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),

            # Horovod: average metrics among workers at the end of every epoch.
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),

            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
            #hvd.callbacks.LearningRateWarmupCallback(initial_lr=self.LEARNING_RATE, warmup_epochs=3, verbose=1),

            # learning rate adjustment
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, mode="min"),

            #EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True, verbose=(1 if hvd.rank() == 0 else 0)),

            # indicate the start and end of an epoch in the profiling data using nvtx marks
            IndicateEpochStartEnd(),
            # indicate the start and end of a test phase in the profiling data using nvtx marks
            IndicateTestStartEnd(),
            # indicates the start and end of a training step in the profiling using nvtx marks
            IndicateTrainingStepStartEnd(),
            # indicate the start and end of the training in the profiling data using nvtx marks
            IndicateTrainingStartEnd(),
            # indicates the start and end of a testing step in the profiling data using nvtx marks
            IndicateTestingStepStartEnd()
        ]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them
        if hvd.rank() == 0:
            self.training_callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix, save_weights_only=True)),

        # enable tf profile if flag is true
        if self.TF_PROFILER == True and hvd.rank() == 0:
            self.training_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.FOLDER_NAME+"/", update_freq="epoch", profile_batch="100, 1000", histogram_freq=1))

    @nvtx.annotate("main->create_callbacks_evaluation", color="orange")
    def create_callbacks_evaluation(self):
        """
        This function creates the callbacks for the evaluation process.
        """

        # Callbacks for training process
        self.evaluation_callbacks = [
            # indicate the start and end of a test phase in the profiling data using nvtx marks
            IndicateTestStartEnd(),
            # indicates the start and end of a testing step in the profiling data using nvtx marks
            IndicateTestingStepStartEnd()
        ]

    @nvtx.annotate("main->define_network", color="pink")
    def define_network(self):
        """
        This function defines the network.
        """

        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),
            preprocessing.Resizing(32, 32),
            #self.norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_labels),
        ])

        opt = SGD(learning_rate=self.LEARNING_RATE, momentum=self.MOMENTUM)
        # Horovod: add Horovod DistributedOptimizer on top of the standard optimizer
        opt = hvd.DistributedOptimizer(opt)
        # INFO: the next line should help with numerical stability
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Compile the model
        # Horovod: Specify experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        self.model.compile(optimizer=opt, loss=loss, metrics=["accuracy"], experimental_run_tf_function=False)


    @nvtx.annotate("main->train", color="red")
    def train(self):
        """
        Training loop.
        """

        # Calculate the number of training and validation steps per epoch
        #train_size = len(self.train_files) / hvd.size()
        #validation_size = len(self.val_files)
        #training_steps = train_size // self.BATCH_SIZE
        #validation_steps = validation_size // self.BATCH_SIZE
        training_steps = 5
        validation_steps = 5

        # train the model
        self.history = self.model.fit(self.train_ds, steps_per_epoch=training_steps, epochs = self.EPOCHS,
            validation_data = self.val_ds, validation_steps=validation_steps, verbose=self.VERBOSE, callbacks=self.training_callbacks)

    @nvtx.annotate("main->evaluation", color="violet")
    def evaluation(self):
        """
        Evaluation loop.
        """

        # evaluate the trained model
        self.TEST_LOSS, self.TEST_ACC = self.model.evaluate(self.val_ds, verbose=self.VERBOSE, callbacks=self.evaluation_callbacks)
        self.TEST_LOSS = "{:.3f}".format(self.TEST_LOSS)
        self.TEST_ACC = "{:.3f}".format(self.TEST_ACC)

        if hvd.rank() == 0:
            print("Final test loss: ", self.TEST_LOSS)
            print("Final test accuracy: ", self.TEST_ACC)

    @nvtx.annotate("main->log_experiment_data", color="gray")
    def log_experiment_data(self):
        """
        This function logs the experiment data.
        """

        # create a logger instance
        logger = Logger(self.JOBNAME, self.FOLDER_NAME, self.REPETITION)

        # log the configuration of this run including hyperparameters
        logger.log_config(self.JOBNAME, self.BATCH_SIZE, self.EPOCHS,
        self.LEARNING_RATE, self.MOMENTUM, self.WEIGHT_DECAY, len(tf.config.experimental.list_physical_devices("GPU")), tf.__version__, hvd.size())

        # Get the data from the training history and save it to a file
        logger.log_training_history(self.history)

        # Save the final model as SavedModel
        path = self.FOLDER_NAME+"/saved_model"
        self.model.save(path, save_format="tf")
