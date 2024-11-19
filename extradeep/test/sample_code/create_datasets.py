import pickle
import numpy as np
import nvtx

class DataLoader:
    """
    This class handles loading the cifar10 data from the files in this directoy. It loads the data and transfers it from dictonary format into numpy arrays. The methods do not do any data preprocessing.
    """

    @nvtx.annotate("main->load_data->init_data_loader", color="blue")
    def __init__(self):
        """
        Initialization of the DataLoader class.
        """

        self.meta_data_path = "batches.meta"
        self.test_data_path = "test_batch"
        self.train_data_path_1 = "data_batch_1"
        self.train_data_path_2 = "data_batch_2"
        self.train_data_path_3 = "data_batch_3"
        self.train_data_path_4 = "data_batch_4"
        self.train_data_path_5 = "data_batch_5"
        
    @nvtx.annotate("main->load_data->unpickle_file", color="blue")
    def unpickle(self, file):
        """
        Open the pickeled files containing the training and test data.
        """

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @nvtx.annotate("main->load_data->load_test_data", color="blue")
    def get_evaluation_data(self):
        """
        Get the evaluation data as numpy arrays.
        """

        dict = self.unpickle(self.test_data_path)
        evaluation_data_labels = dict[b"labels"]
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

        dict1 = self.unpickle(self.train_data_path_1)
        dict2 = self.unpickle(self.train_data_path_2)
        dict3 = self.unpickle(self.train_data_path_3)
        dict4 = self.unpickle(self.train_data_path_4)
        dict5 = self.unpickle(self.train_data_path_5)

        training_labels = []
        training_data = []

        temp_labels = dict1[b"labels"]
        for i in range(len(temp_labels)):
            training_labels.append(temp_labels[i])
        temp_labels = dict2[b"labels"]
        for i in range(len(temp_labels)):
            training_labels.append(temp_labels[i])
        temp_labels = dict3[b"labels"]
        for i in range(len(temp_labels)):
            training_labels.append(temp_labels[i])
        temp_labels = dict4[b"labels"]
        for i in range(len(temp_labels)):
            training_labels.append(temp_labels[i])
        temp_labels = dict5[b"labels"]
        for i in range(len(temp_labels)):
            training_labels.append(temp_labels[i])

        trainY = np.array(training_labels)

        temp_data = dict1[b"data"]
        for i in range(len(temp_data)):
            image_data = temp_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)
        temp_data = dict2[b"data"]
        for i in range(len(temp_data)):
            image_data = temp_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)
        temp_data = dict3[b"data"]
        for i in range(len(temp_data)):
            image_data = temp_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)
        temp_data = dict4[b"data"]
        for i in range(len(temp_data)):
            image_data = temp_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)
        temp_data = dict5[b"data"]
        for i in range(len(temp_data)):
            image_data = temp_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            image_data = image_data.astype("float32") / 255.0
            training_data.append(image_data)

        trainX = np.array(training_data)

        return trainX, trainY
