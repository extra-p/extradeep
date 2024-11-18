import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import nvtx
from tensorflow.keras.layers.experimental import preprocessing
import os

class Preprocessor():
    """
    This class does some necessary data preprocessing for the Cifar-10 benchmark.
    """

    @nvtx.annotate("main->preprocessing->init_preprocessor", color="yellow")
    def __init__(self, train_files, val_files, test_files, BATCH_SIZE, EPOCHS, num_workers, worker_index, num_classes, commands):
        """
        Initialization of the Preprocessor.
        """

        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.data_augmentation = None
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.num_workers = num_workers
        self.worker_index = worker_index
        self.num_classes = num_classes
        self.commands = commands
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    @nvtx.annotate("main->preprocessing->decode_audio", color="yellow")
    def decode_audio(self, audio_binary):
        """
        Decode audio.
        """

        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    @nvtx.annotate("main->preprocessing->get_label", color="yellow")
    def get_label(self, file_path):
        """
        Get the label.
        """

        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2]

    @nvtx.annotate("main->preprocessing->get_waveform_and_label", color="yellow")
    def get_waveform_and_label(self, file_path):
        """
        Get the label and the waveform.
        """

        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label

    @nvtx.annotate("main->preprocessing->get_spectrogram", color="yellow")
    def get_spectrogram(self, waveform):
        """
        Get the spectogram of the waveform.
        """

        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
        # Concatenate audio with padding so that all audio clips will be of the
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
          equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        return spectrogram

    @nvtx.annotate("main->preprocessing->get_spectrogram", color="yellow")
    def get_spectrogram_and_label_id(self, audio, label):
        """
        Get the spectogram and label id.
        """

        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == self.commands)
        return spectrogram, label_id

    @nvtx.annotate("main->preprocessing->preprocess_dataset", color="yellow")
    def preprocess_dataset(self, files):
        """
        Preprocess the different data sets.
        """

        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=self.AUTOTUNE)
        output_ds = output_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=self.AUTOTUNE)
        return output_ds


    @nvtx.annotate("main->preprocessing->preprocess", color="yellow")
    def preprocess(self):
        """
        This functions runs all necessary preprocessing steps after each other finally it returns the preprocessed training and test data.
        """

        print("1")

        files_ds = tf.data.Dataset.from_tensor_slices(self.train_files)
        waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=self.AUTOTUNE)
        spectrogram_ds = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=self.AUTOTUNE)
        self.train_ds = spectrogram_ds
        self.val_ds = self.preprocess_dataset(self.val_files)
        self.test_ds = self.preprocess_dataset(self.test_files)
        
        print("2")

        self.train_ds = self.train_ds.shard(self.num_workers, self.worker_index)
        self.train_ds = self.train_ds.repeat(self.EPOCHS)
        bsize = int((25000/self.num_workers)+1000)
        self.train_ds = self.train_ds.shuffle(buffer_size=bsize, reshuffle_each_iteration=True)

        print("3")

        self.train_ds = self.train_ds.batch(self.BATCH_SIZE)
        self.val_ds = self.val_ds.batch(self.BATCH_SIZE)


        print("4")

        self.train_ds = self.train_ds.cache().prefetch(self.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(self.AUTOTUNE)

        print("5")

        for spectrogram, _ in spectrogram_ds.take(1):
            input_shape = spectrogram.shape
        num_labels = len(self.commands)

        print("6")

        #norm_layer = preprocessing.Normalization()
        #norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
        norm_layer = None

        print("7")

        return self.train_ds, self.val_ds, self.test_ds, input_shape, num_labels, norm_layer
