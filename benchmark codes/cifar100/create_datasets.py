import pickle
import numpy as np
import nvtx

class DataLoader:
    """
    This class handles loading the cifar100 data from the files in this directoy. It loads the data and transfers it from dictonary format into numpy arrays. The methods do not do any data preprocessing.
    """

    @nvtx.annotate("main->load_data->init_data_loader", color="blue")
    def __init__(self):
        """
        Initialization of the DataLoader class.
        """

        self.meta_data_path = "meta"
        self.test_data_path = "test"
        self.train_data_path = "train"

    @nvtx.annotate("main->load_data->unpickle_file", color="blue")
    def unpickle(self, file):
        """
        Open the pickeled files containing the training and test data.
        """

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @nvtx.annotate("main->load_data->load_classes", color="blue")
    def get_classes(self):
        """
        Get the classes of the classification data set.
        """

        dict = self.unpickle(self.meta_data_path)
        label_names = dict[b"fine_label_names"]
        classes = []
        for i in range(len(label_names)):
            classes.append(label_names[i].decode("utf-8"))
        return classes

    @nvtx.annotate("main->load_data->load_test_data", color="blue")
    def get_evaluation_data(self):
        """
        Get the evaluation data as numpy arrays.
        """

        dict = self.unpickle(self.test_data_path)
        evaluation_data_labels = dict[b"fine_labels"]
        testY = np.array(evaluation_data_labels)
        data = dict[b"data"]
        evaluation_data = []
        for i in range(len(data)):
            image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            evaluation_data.append(image_data)
        testX = np.array(evaluation_data)
        return testX, testY

    @nvtx.annotate("main->load_data->load_training_data", color="blue")
    def get_training_data(self):
        """
        Get the training data as numpy arrays.
        """

        dict = self.unpickle(self.train_data_path)
        training_data_labels = dict[b"fine_labels"]
        trainY = np.array(training_data_labels)
        data = dict[b"data"]
        training_data = []
        for i in range(len(data)):
            image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)
        trainX = np.array(training_data)
        return trainX, trainY
