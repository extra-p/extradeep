import pickle
import numpy as np
import nvtx
import tensorflow as tf
import pathlib

class DataLoader:
    """
    This class handles loading the speech_commands data.
    """

    @nvtx.annotate("main->load_data->init_data_loader", color="blue")
    def __init__(self, USE_MINI):
        """
        Initialization of the DataLoader class.
        """

        self.USE_MINI = USE_MINI
        self.data_dir = None
        self.commands = None
        self.NUM_CLASSES = None
        self.filenames = None
        self.num_samples = None

    @nvtx.annotate("main->load_data->get_num_classes", color="blue")
    def get_num_classes(self):
        """
        Get the number of classes.
        """

        return self.NUM_CLASSES

    @nvtx.annotate("main->load_data->download_data", color="blue")
    def download_data(self):
        """
        Download the dataset.
        """

        if self.USE_MINI == True:
            self.data_dir = pathlib.Path('data/mini_speech_commands')
            if not self.data_dir.exists():
                tf.keras.utils.get_file(
                    'mini_speech_commands.zip',
                    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                    extract=True,
                    cache_dir='.', cache_subdir='data')
        else:
            self.data_dir = pathlib.Path('data/speech_commands')
            if not self.data_dir.exists():
                 tf.keras.utils.get_file(
                     "speech_commands.zip",
                     origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                     extract=True,
                     cache_dir='.', cache_subdir='data')

    @nvtx.annotate("main->load_data->get_meta_data", color="blue")
    def get_meta_data(self):
        """
        Get the meta data, e.g. classes and commands.
        """

        commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
        self.commands = commands[commands != 'README.md']
        self.NUM_CLASSES = len(commands)
        return self.commands

    @nvtx.annotate("main->load_data->get_training_data", color="blue")
    def get_training_data(self):
        """
        Get the training data.
        """

        filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
        self.filenames = tf.random.shuffle(filenames)
        self.num_samples = len(filenames)
        train_files = self.filenames[:6400]
        val_files = self.filenames[6400: 6400 + 800]
        test_files = self.filenames[-800:]
        return train_files, val_files, test_files
