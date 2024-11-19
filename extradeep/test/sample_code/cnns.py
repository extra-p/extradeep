from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
import nvtx

"""
CNN class. Used to build different cnn architectures.
"""
class CNN():

    """
    Initialization of the network builder.
    """
    @nvtx.annotate("main->define_network->init_network_builder", color="pink")
    def __init__(self, weigt_decay):
        self.weigt_decay = weigt_decay

    """
    Method to initialize cnn network architecture.
    """
    @nvtx.annotate("main->define_network->define_cnn", color="pink")
    def define_cnn(self, input_shape, num_outputs, num_layers):

        num_layers = num_layers - 3

        # f1 and f2 are the width and height, and f3 is the number of channels
        f1, f2, f3 = input_shape
        model = Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(f1, f2, f3)))
        model.add(layers.MaxPooling2D((2, 2)))

        last = 0
        for i in range(num_layers):
            layer_number = i+1+1
            model.add(layers.Conv2D(32*layer_number, (3, 3), activation='relu', padding="same"))
            model.add(layers.MaxPooling2D((2, 2), padding="same"))
            last = layer_number

        model.add(layers.Flatten())
        model.add(layers.Dense(32*last, activation='relu'))
        model.add(layers.Dense(num_outputs, activation="softmax"))

        #model.summary()

        return model
