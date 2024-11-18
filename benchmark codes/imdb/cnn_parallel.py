from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import nvtx

"""
Optimized CNN for IMDB data set.
"""
class CNN():

    """
    Initialization of the network builder.
    """
    @nvtx.annotate("main->define_network->init_network_builder", color="pink")
    def __init__(self, trainX):
        self.trainX = trainX

    """
    Defines a simple CNN network that can be trained quickly for CIFAR100.
    """
    @nvtx.annotate("main->define_network->create_layers", color="pink")
    def define(self):

        model = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
        hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
        hub_layer(self.trainX[:3])
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model

    """
    Compiles the model.
    """
    @nvtx.annotate("main->define_network->compile_model", color="pink")
    def compile_model(self, model, learning_rate):
        opt = Adam(learning_rate=learning_rate)
        # Horovod: add Horovod DistributedOptimizer on top of the standard optimizer
        opt = hvd.DistributedOptimizer(opt)
        # INFO: the next line should help with numerical stability
        loss = tf.losses.BinaryCrossentropy(from_logits=True)
        # Compile the model
        # Horovod: Specify experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model.compile(optimizer=opt, loss=loss, metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')], experimental_run_tf_function=False)

        return model
