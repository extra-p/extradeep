import nvtx
import tensorflow as tf

class PrintLearningRate(tf.keras.callbacks.Callback):
    """
    Callback class for printing the learning rate at the end of each epoch. Based on the tf.keras implementation.
    """

    def __init__(self, model):
        """
        Initializes the instance and overrides the base class constructor so that we have the model here to get the current learning rate from.
        """

        super().__init__()
        self.model = model

    @nvtx.annotate("main->print_learning_rate", color="orange")
    def on_epoch_end(self, epoch, logs=None):
        """
        Print learning rate at the end of the epoch.
        """

        print('\nLearning rate for epoch {} is {:.6f}'.format(epoch + 1, self.model.optimizer.lr.numpy()))

class IndicateEpochStartEnd(tf.keras.callbacks.Callback):
    """
    Callback that is used together with nvtx marks to indicate when an epoch starts and ends. Based on the tf.keras implementation.
    """

    def on_epoch_begin(self, epoch, logs=None):
        """
        Indicate Epoch start using nvtx mark.
        """

        text = "epoch:"+str(epoch)+",type:started"  
        nvtx.mark(message=text, color="orange", category="epoch")
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Indicate Epoch end using nvtx mark.
        """

        text = "epoch:"+str(epoch)+",type:ended"  
        nvtx.mark(message=text, color="orange", category="epoch")

class IndicateTestStartEnd(tf.keras.callbacks.Callback):
    """
    Callback that is used together with nvtx marks to indicate when a validation or evaluation phase starts and ends. Based on the tf.keras implementation.
    """

    def on_test_begin(self, logs=None):
        """
        Indicate test phase start using nvtx mark.
        """

        text = "test started"  
        nvtx.mark(message=text, color="orange", category="test")
        
    def on_test_end(self, logs=None):
        """
        Indicate test phase end using nvtx mark.
        """

        text = "test ended"  
        nvtx.mark(message=text, color="orange", category="test")

class IndicateTrainingStepStartEnd(tf.keras.callbacks.Callback):
    """
    Callback that is used together with nvtx marks to indicate when a training batch / a step starts and ends. Based on the tf.keras implementation.
    """

    def on_train_batch_begin(self, batch, logs=None):
        """
        Indicate training step start using nvtx mark.
        """

        text = "step:"+str(batch)+",type:started"  
        nvtx.mark(message=text, color="orange", category="training step")
        
    def on_train_batch_end(self, batch, logs=None):
        """
        Indicate training step end using nvtx mark.
        """

        text = "step:"+str(batch)+",type:ended"  
        nvtx.mark(message=text, color="orange", category="training step")

class IndicateTrainingStartEnd(tf.keras.callbacks.Callback):
    """
    Callback that is used together with nvtx marks to indicate when training starts and ends. Based on the tf.keras implementation.
    """

    def on_train_begin(self, logs=None):
        """
        Indicate training start using nvtx mark.
        """

        text = "training started"  
        nvtx.mark(message=text, color="orange", category="training")
        
    def on_train_end(self, logs=None):
        """
        Indicate training end using nvtx mark.
        """

        text = "training ended"  
        nvtx.mark(message=text, color="orange", category="training")

class IndicateTestingStepStartEnd(tf.keras.callbacks.Callback):
    """
    Callback that is used together with nvtx marks to indicate when a test batch / a step starts and ends. Based on the tf.keras implementation.
    """

    def on_test_batch_begin(self, batch, logs=None):
        """
        Indicate testing step start using nvtx mark.
        """

        text = "step:"+str(batch)+",type:started"  
        nvtx.mark(message=text, color="orange", category="testing step")
        
    def on_test_batch_end(self, batch, logs=None):
        """
        Indicate testing step end using nvtx mark.
        """

        text = "step:"+str(batch)+",type:ended"  
        nvtx.mark(message=text, color="orange", category="testing step")
