# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import pandas as pd
import pickle
import os

from extradeep.util.plotting import plot_epochs_runtime
from extradeep.util.util_functions import convert_to_py_function
import math
import matplotlib
import matplotlib.pyplot as plt
from math import log2
import numpy as np
from extradeep.util.util_functions import get_tabulate_string
from rich.tree import Tree
from rich import print as richprint

def evaluate(experiment_eval, experiment, arguments, application_name):
    
    text = "\n"
    text += "Evaluation results:"
    text += "\n"
    text += "\n"
    data = {}

    model_py_function = None
    data_storages = []

    data_storages = {}
    analysistype = experiment.analysistypes
    type = analysistype[0]
    temp_metrics = experiment.metrics[type]
    for i in range(len(temp_metrics)):
        data_storages[i] = []

    counter = 0

    # if loading data from extradeep data set instead from raw data
    # we want to use the latest model generator, otherwise the arguments for the modeler
    # passed to the command line do not influence the output shown in the terminal
    if len(experiment.modelers) > 1:
        modeler = experiment.modelers[len(experiment.modelers)-1]
    else:
        modeler = experiment.modelers[0]

    for j in range(len(experiment.coordinates)):
        analysistype = experiment.analysistypes
        type = analysistype[0]
        temp_callpaths = experiment.callpaths[type]
        temp_metrics = experiment.metrics[type]
        for i in range(len(temp_callpaths)):

            metrics = {}

            for l in range(len(temp_metrics)):

                # use mean or median for evaluation based on the command line argument
                if arguments.median == True:
                    actual = experiment.get_measurement(j, i, l, type).median
                else:
                    actual = experiment.get_measurement(j, i, l, type).mean

                min = experiment.get_measurement(j, i, l, type).minimum
                max = experiment.get_measurement(j, i, l, type).maximum

                div = max - min
                div = div / 2
                if div == 0:
                    noise_percent = 0
                else:
                    if actual == 0:
                        noise_percent = 0
                    else:
                        noise_percent = div / (actual / 100)

                model = modeler.models[temp_callpaths[i], temp_metrics[l], type]
                hypothesis = model.hypothesis
                function = hypothesis.function
                function_string = function.to_string(*experiment.parameters)
                function = convert_to_py_function(function_string)

                parameters = arguments.param
                parameters = parameters.split(",")

                if len(experiment.parameters) == 1:
                    pos = function.find(parameters[0])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x1"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[0])

                    x1 = experiment.coordinates[j][0]
                    prediction = eval(function)

                elif len(experiment.parameters) == 2:
                    pos = function.find(parameters[0])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x1"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[0])

                    pos = function.find(parameters[1])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x2"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[1])

                    x1 = experiment.coordinates[j][0]
                    x2 = experiment.coordinates[j][1]
                    prediction = eval(function)

                elif len(experiment.parameters) == 3:
                    pos = function.find(parameters[0])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x1"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[0])

                    pos = function.find(parameters[1])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x2"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[1])

                    pos = function.find(parameters[2])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x3"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[2])

                    x1 = experiment.coordinates[j][0]
                    x2 = experiment.coordinates[j][1]
                    x3 = experiment.coordinates[j][2]
                    prediction = eval(function)

                if prediction == 0 and actual == 0:
                    relative_error = 0
                    percentage_error = 0
                else:
                    absolute_error = abs(prediction - actual)
                    if actual == 0:
                        relative_error = 0
                        percentage_error = 0
                    else:
                        relative_error = absolute_error / actual
                        relative_error = abs(relative_error)
                        percentage_error = relative_error * 100

                #data_storages.append(DataStorage(int(counter+1), float(experiment.coordinates[j]._values[0]), float(actual), float(prediction), float(absolute_error), float(percentage_error), float(min), float(max), float(noise_percent)))
                data_storages[l].append(DataStorage(int(counter+1), float(experiment.coordinates[j]._values[0]), float(actual), float(prediction), float(absolute_error), float(percentage_error), float(min), float(max), float(noise_percent)))

        counter += 1

    for j in range(len(experiment_eval.coordinates)):

        if len(experiment_eval.coordinates[j]._values) > 1:
            return 1

        else:
            #parameter_values.append(experiment_eval.coordinates[j]._values[0])
            #value_id.append(j+1)
            pass

        callpaths = {}
        # get the default analysis type
        analysistype = experiment_eval.analysistypes
        type = analysistype[0]
        temp_callpaths = experiment_eval.callpaths[type]
        # retrieve the metrics list
        temp_metrics = experiment_eval.metrics[type]
        text += "Coordinate: "+str(experiment_eval.coordinates[j])+"\n"

        for i in range(len(temp_callpaths)):

            text += "\tCallpath: "+str(temp_callpaths[i])+"\n"
            metrics = {}

            for l in range(len(temp_metrics)):

                text += "\t\tMetric: "+str(temp_metrics[l])
                
                # use mean or median for evaluation based on the command line argument
                if arguments.median == True:
                    actual = experiment_eval.get_measurement(j, i, l, type).median
                else:
                    actual = experiment_eval.get_measurement(j, i, l, type).mean

                min = experiment_eval.get_measurement(j, i, l, type).minimum
                max = experiment_eval.get_measurement(j, i, l, type).maximum

                div = max - min
                div = div / 2
                if div == 0:
                    noise_percent = 0
                else:
                    if actual == 0:
                        noise_percent = 0
                    else:
                        noise_percent = div / (actual / 100)

                # get the model and the function
                model = modeler.models[temp_callpaths[i], temp_metrics[l], type]

                hypothesis = model.hypothesis
                function = hypothesis.function
                rss = hypothesis.RSS
                ar2 = hypothesis.AR2
                rrss = hypothesis.rRSS
                smape = hypothesis.SMAPE
                re = hypothesis.RE
                function_string = function.to_string(*experiment.parameters)

                text += "\n"
                text += "\n"
                text += "\t\t\tModel: "+str(convert_to_py_function(function_string))+"\n"
                text += "\t\t\tSMAPE: {:.3f}".format(smape)
                text += " AR2: {:.3f}".format(ar2)
                text += " RE: {:.3f}".format(re)+"\n"

                function = convert_to_py_function(function_string)

                model_py_function = function

                parameters = arguments.param
                parameters = parameters.split(",")

                if len(experiment_eval.parameters) == 1:
                    pos = function.find(parameters[0])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x1"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[0])

                    x1 = experiment_eval.coordinates[j][0]
                    prediction = eval(function)

                elif len(experiment_eval.parameters) == 2:
                    pos = function.find(parameters[0])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x1"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[0])

                    pos = function.find(parameters[1])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x2"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[1])

                    x1 = experiment_eval.coordinates[j][0]
                    x2 = experiment_eval.coordinates[j][1]
                    prediction = eval(function)

                elif len(experiment_eval.parameters) == 3:
                    pos = function.find(parameters[0])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x1"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[0])

                    pos = function.find(parameters[1])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x2"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[1])

                    pos = function.find(parameters[2])
                    while pos != -1:
                        temp = function[:pos]
                        temp += "x3"+function[pos+1:]
                        function = temp
                        pos = function.find(parameters[2])

                    x1 = experiment_eval.coordinates[j][0]
                    x2 = experiment_eval.coordinates[j][1]
                    x3 = experiment_eval.coordinates[j][2]
                    prediction = eval(function)

                text += "\t\t\tActual: {:.9f}".format(actual)+"\n"
                text += "\t\t\tPrediction: {:.9f}".format(prediction)+"\n"

                #actuals.append(actual)
                #predictions.append(prediction)

                if prediction == 0 and actual == 0:
                    relative_error = 0
                    percentage_error = 0
                else:
                    absolute_error = abs(prediction - actual)
                    if actual == 0:
                        relative_error = 0
                        percentage_error = 0
                    else:
                        relative_error = absolute_error / actual
                        relative_error = abs(relative_error)
                        percentage_error = relative_error * 100

                #errors.append(absolute_error)
                #errors_percent.append(percentage_error)

                #text += "\t\t\tMedian Relative Error: {:.3f}".format(relative_error)+"\n"
                text += "\t\t\tMPE: {:.3f}".format(percentage_error)+"%\n"
                text += "\n"
                text += "\n"

                #data_storages.append(DataStorage(int(counter+1), float(experiment_eval.coordinates[j]._values[0]), float(actual), float(prediction), float(absolute_error), float(percentage_error), float(min), float(max), float(noise_percent)))
                data_storages[l].append(DataStorage(int(counter+1), float(experiment_eval.coordinates[j]._values[0]), float(actual), float(prediction), float(absolute_error), float(percentage_error), float(min), float(max), float(noise_percent)))

                metrics[temp_metrics[l]] = percentage_error
            callpaths[temp_callpaths[i]] = metrics
        data[experiment_eval.coordinates[j]] = callpaths

        counter += 1

    print(text)
    
    # make some visualizations from the evaluation data
    #plot_epochs_runtime(data, experiment_eval)


    analysistype = experiment.analysistypes
    type = analysistype[0]
    temp_metrics = experiment.metrics[type]

    # sort the eval data by parameter-values of the cords
    for i in range(len(temp_metrics)):
        data_storages[i] = sort_data_storages(data_storages[i])

    # create pandas data frame from the eval data
    dfs = {}
    for i in range(len(temp_metrics)):
        df = create_data_frame(data_storages[i])
        dfs[temp_metrics[i]] = df

    # create object for saving data from data frame and model string
    data = (model_py_function, dfs)

    # save the data
    save_object(data, "", "training_steps_data")

    for i in range(len(temp_metrics)):
        print("Evaluation results "+str(temp_metrics[i])+":\n")
        print(dfs[temp_metrics[i]].to_markdown())
        print("")



    return text


class DataStorage():

    def __init__(self, value_id, parameter_value, actual, prediction, error, error_percent, min, max, noise) -> None:
        self.value_id = value_id
        self.parameter_value = parameter_value
        self.actual = actual
        self.prediction = prediction
        self.error = error
        self.error_percent = error_percent
        self.min = min
        self.max = max
        self.noise = noise


def sort_data_storages(data_storages):
    """
    sort_data_storages function to sort the storage by the parameter-value.
    """
    data_storages = sorted(data_storages, key=lambda x: x.parameter_value, reverse=False)
    return data_storages


def create_data_frame(data):
    """
    create_data_frame function to create a data frame from pandas using the evaluation data.
    """
    columns = ["id", "parameter_value", "actual", "prediction", "error", "error_percent", "min", "max", "+- noise %"]
    y = []
    for i in range(len(data)):
        x = [data[i].value_id, data[i].parameter_value, data[i].actual, data[i].prediction, data[i].error, data[i].error_percent, data[i].min, data[i].max, data[i].noise]
        y.append(x)
    df = pd.DataFrame(y, columns=columns)
    return df


def save_object(obj, path, name):
    """
    save_object function to save an object using pickle.
    """
    try:
        path = os.path.join(path, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
