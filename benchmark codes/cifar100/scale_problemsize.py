import numpy as np
import nvtx

class ProblemScaler:
    """
    This class handles the scaling of the problem size based on the configuration specified by the user.
    The class does not do any preprocessing.
    """

    @nvtx.annotate("main->scale_problemsize->init_problem_scaler", color="purple")
    def __init__(self, classes, trainX, trainY, testX, testY):
        """
        Load the data for training and evaluation when initializing the class.
        """

        self.classes = classes
        self.testX = testX
        self.testY = testY
        self.trainX = trainX
        self.trainY = trainY

    @nvtx.annotate("main->scale_problemsize->scale", color="purple")
    def scale_problem(self, PROBLEM_SIZE):
        """
        Scales the problem size of train and test data based on the user specified problem size.
        Problem size must be integer!
        """
        
        # for test data
        if PROBLEM_SIZE != 100:
            new_testX = []
            new_testY = []
            classes = []
            for i in range(PROBLEM_SIZE):
                classes.append(i)
            for i in range(len(self.testY)):
                for j in range(PROBLEM_SIZE):
                    if self.testY[i] == classes[j]:
                        value = self.testY[i]
                        image = self.testX[i]
                        new_testY.append(value)
                        new_testX.append(image)
                        break
            self.testY = new_testY
            self.testX = new_testX
            self.testY = np.asarray(self.testY)
            self.testX = np.asarray(self.testX)

        # for training data
        if PROBLEM_SIZE != 100:
            new_trainY = []
            new_trainX = []
            classes = []
            for i in range(PROBLEM_SIZE):
                classes.append(i)
            for i in range(len(self.trainY)):
                for j in range(PROBLEM_SIZE):
                    if self.trainY[i] == classes[j]:
                        value = self.trainY[i]
                        image = self.trainX[i]
                        new_trainY.append(value)
                        new_trainX.append(image)
                        break
            self.trainY = new_trainY
            self.trainX = new_trainX
            self.trainY = np.asarray(self.trainY)
            self.trainX = np.asarray(self.trainX)

        return self.trainX, self.trainY, self.testX, self.testY
