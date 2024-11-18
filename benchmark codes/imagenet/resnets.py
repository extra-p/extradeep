from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import activations, layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
import nvtx
import horovod.tensorflow.keras as hvd

"""
ResNet class. Used to build different resnet architectures.
"""
class ResNet():

    """
    Initialization of the network builder.
    """
    @nvtx.annotate("main->define_network->init_network_builder", color="pink")
    def __init__(self, weigt_decay):
        self.weigt_decay = weigt_decay

    """
    Resnet block where dimension doesnot change.
    The skip connection is just simple identity conncection
    we will have 3 blocks and then input will be added.
    """
    @nvtx.annotate("main->define_network->res_identity", color="pink")
    def res_identity(self, x, filters):
        x_skip = x # this will be used for addition with the residual block
        f1, f2 = filters

        #first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        # x = Activation(activations.relu)(x)

        # add the input
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x

    """
    Resnet block where dimension doesnot change.
    The skip connection is just simple identity conncection
    we will have 2 blocks and then input will be added.
    """
    @nvtx.annotate("main->define_network->res_identity_2layer", color="pink")
    def res_identity_2layer(self, x, filters):
        x_skip = x # this will be used for addition with the residual block
        f1 = filters

        #first block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding), activation used afeer adding the input
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        #x = Activation(activations.relu)(x)

        # add the input
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x

    """
    Here the input size changes, when it goes via conv blocks
    so the skip connection uses a projection (conv layer) matrix.
    """
    @nvtx.annotate("main->define_network->res_conv", color="pink")
    def res_conv(self, x, s, filters):
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(self.weigt_decay))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)

        # shortcut
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(self.weigt_decay))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x

    """
    Here the input size changes, when it goes via conv blocks
    so the skip connection uses a projection (conv layer) matrix.
    For resnet 18 and 34 where all filters have the same size.
    """
    @nvtx.annotate("main->define_network->res_conv_2layers", color="pink")
    def res_conv_2layers(self, x, s, filters):
        x_skip = x
        f1 = filters

        # first block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(s, s), padding='same', kernel_regularizer=l2(self.weigt_decay))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.weigt_decay))(x)
        x = BatchNormalization()(x)
        #x = Activation(activations.relu)(x)

        # shortcut
        x_skip = Conv2D(f1, kernel_size=(3, 3), strides=(s, s), padding='same', kernel_regularizer=l2(self.weigt_decay))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x

    """
    Method to initialize resnet101 network architecture
    """
    @nvtx.annotate("main->define_network->define_resnet101", color="pink")
    def define_resnet101(self, input_shape, num_outputs):
        # f1 and f2 are the width and height, and f3 is the number of channels

        f1, f2, f3 = input_shape

        input_im = Input(shape=(f1, f2, f3))

        # data augmentation
        #x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(f1, f2, f3))(input_im)
        #x = layers.experimental.preprocessing.RandomRotation(0.1)(x)
        #x = layers.experimental.preprocessing.RandomZoom(0.1)(x)
        #x = ZeroPadding2D(padding=(3, 3))(x)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_im)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage
        # from here on only conv block and identity block, no pooling

        x = self.res_conv(x, s=1, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))

        # 3rd stage

        x = self.res_conv(x, s=2, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))

        # 4th stage

        x = self.res_conv(x, s=2, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))

        # 5th stage

        x = self.res_conv(x, s=2, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(num_outputs, activation="softmax", dtype="float32")(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='ResNet-101')

        return model

    """
    Method to initialize resnet152 network architecture
    """
    @nvtx.annotate("main->define_network->define_resnet152", color="pink")
    def define_resnet152(self, input_shape, num_outputs):
        # f1 and f2 are the width and height, and f3 is the number of channels

        f1, f2, f3 = input_shape

        input_im = Input(shape=(f1, f2, f3))

        # data augmentation
        #x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(f1, f2, f3))(input_im)
        #x = layers.experimental.preprocessing.RandomRotation(0.1)(x)
        #x = layers.experimental.preprocessing.RandomZoom(0.1)(x)
        #x = ZeroPadding2D(padding=(3, 3))(x)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_im)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage
        # from here on only conv block and identity block, no pooling

        x = self.res_conv(x, s=1, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))

        # 3rd stage

        x = self.res_conv(x, s=2, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))

        # 4th stage

        x = self.res_conv(x, s=2, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))

        # 5th stage

        x = self.res_conv(x, s=2, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(num_outputs, activation="softmax", dtype="float32")(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='ResNet-152')

        return model

    """
    Method to initialize resnet50 network architecture.
    """
    @nvtx.annotate("main->define_network->define_resnet50", color="pink")
    def define_resnet50(self, input_shape, num_outputs):

        # f1 and f2 are the width and height, and f3 is the number of channels

        f1, f2, f3 = input_shape

        input_im = Input(shape=(f1, f2, f3))

        # data augmentation
        #x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(f1, f2, f3))(input_im)
        #x = layers.experimental.preprocessing.RandomRotation(0.1)(x)
        #x = layers.experimental.preprocessing.RandomZoom(0.1)(x)
        #x = ZeroPadding2D(padding=(3, 3))(x)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_im)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage
        # from here on only conv block and identity block, no pooling

        x = self.res_conv(x, s=1, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))
        x = self.res_identity(x, filters=(64, 256))

        # 3rd stage

        x = self.res_conv(x, s=2, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))
        x = self.res_identity(x, filters=(128, 512))

        # 4th stage

        x = self.res_conv(x, s=2, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))
        x = self.res_identity(x, filters=(256, 1024))

        # 5th stage

        x = self.res_conv(x, s=2, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))
        x = self.res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(num_outputs, activation="softmax", dtype="float32")(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='ResNet-50')

        return model

    """
    Method to initialize resnet34 network architecture.
    """
    @nvtx.annotate("main->define_network->define_resnet34", color="pink")
    def define_resnet34(self, input_shape, num_outputs):

        # f1 and f2 are the width and height, and f3 is the number of channels

        f1, f2, f3 = input_shape

        input_im = Input(shape=(f1, f2, f3))

        # data augmentation
        #x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(f1, f2, f3))(input_im)
        #x = layers.experimental.preprocessing.RandomRotation(0.1)(x)
        #x = layers.experimental.preprocessing.RandomZoom(0.1)(x)
        #x = ZeroPadding2D(padding=(3, 3))(x)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_im)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage
        # from here on only conv block and identity block, no pooling

        x = self.res_conv_2layers(x, s=1, filters=(64))
        x = self.res_identity_2layer(x, filters=(64))
        x = self.res_identity_2layer(x, filters=(64))

        # 3rd stage

        x = self.res_conv_2layers(x, s=2, filters=(128))
        x = self.res_identity_2layer(x, filters=(128))
        x = self.res_identity_2layer(x, filters=(128))
        x = self.res_identity_2layer(x, filters=(128))

        # 4th stage

        x = self.res_conv_2layers(x, s=2, filters=(256))
        x = self.res_identity_2layer(x, filters=(256))
        x = self.res_identity_2layer(x, filters=(256))
        x = self.res_identity_2layer(x, filters=(256))
        x = self.res_identity_2layer(x, filters=(256))
        x = self.res_identity_2layer(x, filters=(256))

        # 5th stage

        x = self.res_conv_2layers(x, s=2, filters=(512))
        x = self.res_identity_2layer(x, filters=(512))
        x = self.res_identity_2layer(x, filters=(512))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(num_outputs, activation="softmax", dtype="float32")(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='ResNet-34')

        return model

    """
    Method to initialize resnet18 network architecture.
    """
    @nvtx.annotate("main->define_network->define_resnet18", color="pink")
    def define_resnet18(self, input_shape, num_outputs):

        # f1 and f2 are the width and height, and f3 is the number of channels

        f1, f2, f3 = input_shape

        input_im = Input(shape=(f1, f2, f3))

        # data augmentation
        #x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(f1, f2, f3))(input_im)
        #x = layers.experimental.preprocessing.RandomRotation(0.1)(x)
        #x = layers.experimental.preprocessing.RandomZoom(0.1)(x)
        #x = ZeroPadding2D(padding=(3, 3))(x)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_im)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage
        # from here on only conv block and identity block, no pooling

        x = self.res_conv_2layers(x, s=1, filters=(64))
        x = self.res_identity_2layer(x, filters=(64))

        # 3rd stage

        x = self.res_conv_2layers(x, s=2, filters=(128))
        x = self.res_identity_2layer(x, filters=(128))

        # 4th stage

        x = self.res_conv_2layers(x, s=2, filters=(256))
        x = self.res_identity_2layer(x, filters=(256))

        # 5th stage

        x = self.res_conv_2layers(x, s=2, filters=(512))
        x = self.res_identity_2layer(x, filters=(512))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(num_outputs, activation="softmax", dtype="float32")(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='ResNet-18')

        return model

    """
    Compiles the provided model and adds an optimizer and sets hyperparameters
    such as learning rate, weight decay and so on.
    """
    @nvtx.annotate("main->define_network->compile_model", color="pink")
    def compile_model(self, model, learning_rate, momentum):
        opt = SGD(learning_rate=learning_rate, momentum=momentum)
        # Horovod: add Horovod DistributedOptimizer on top of the standard optimizer
        opt = hvd.DistributedOptimizer(opt)
        # INFO: the next line should help with numerical stability
        loss = CategoricalCrossentropy(from_logits=False),
        # Compile the model
        # Horovod: Specify experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"], experimental_run_tf_function=False)
        return model
