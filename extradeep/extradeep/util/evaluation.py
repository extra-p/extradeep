# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

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

def evaluate_epochs(experiment_eval, experiment, arguments, application_name, plot):
    text = "\n"
    text += "Evaluation results:"
    text += "\n"
    text += "\n"
    data = {}
    for j in range(len(experiment_eval.coordinates)):
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
                actual = experiment_eval.get_measurement(j, i, l, type).mean

                # get the model and the function
                modeler = experiment.modelers[0]
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

                #text += "\t\t\tMedian Relative Error: {:.3f}".format(relative_error)+"\n"
                text += "\t\t\tMPE: {:.3f}".format(percentage_error)+"%\n"
                text += "\n"
                text += "\n"

                metrics[temp_metrics[l]] = percentage_error
            callpaths[temp_callpaths[i]] = metrics
        data[experiment_eval.coordinates[j]] = callpaths

    print(text)

    # plot results
    if plot == True:
        # make some visualizations from the evaluation data
        plot_epochs_runtime(data, experiment_eval)

    return text



def evaluate_kernels(experiment_eval, experiment, arguments):

    output_text = ""

    # do the evaluation
    output_text += "\n-------------------\n|    Analysis  |\n--------------------\n"
    print("\n------------------------------\n|    Analysis  |\n------------------------------\n")
    output_text += "\nNr. points used for modeling: "+str(len(experiment.coordinates))
    print("Nr. points used for modeling: "+str(len(experiment.coordinates)))
    output_text += "\nModeling points: "
    print("Modeling points:")
    for i in range(len(experiment.coordinates)):
        output_text += str(experiment.coordinates[i])
        print(experiment.coordinates[i])
    output_text += "\n"
    print("")
    output_text += "\nNr. points used for evaluation : "+str(len(experiment_eval.coordinates))
    print("Nr. points used for evaluation:"+str(len(experiment_eval.coordinates)))
    output_text += "\nEvaluation points: "
    print("Evaluation points:")
    for i in range(len(experiment_eval.coordinates)):
        output_text += str(experiment_eval.coordinates[i])
        print(experiment_eval.coordinates[i])
    print("")
    output_text += "\n"
    metrics = experiment.metrics
    output_text += "\nMetrics: "+str(metrics)
    print("Metrics:",metrics)
    print("")
    output_text += "\n"
    parameters = experiment.parameters
    output_text += "\nParameters: "+str(parameters)
    print("Parameters:",parameters)
    output_text += "\n"
    print("")

    output_text += "\n-------------------------------------\n|    Predictive Power Analysis  |\n-------------------------------------\n"
    print("\n---------------\n|    Predictive Power Analysis  |\n---------------\n")

    type = experiment_eval.analysistypes[0]
    callpaths = experiment_eval.callpaths[type]
    metrics = experiment_eval.metrics[type]

    # create a data container for the evaluation results
    data = []
    for i in range(len(experiment_eval.coordinates)):
        metrics_temp = []
        for j in range(len(metrics)):
            metrics_temp.append([])
        data.append(metrics_temp)

    # compute the evaluation results
    for i in range(len(callpaths)):

        output_text += "\n"
        print("")

        callpath = callpaths[i]
        output_text += "Callpath: "+str(callpath)+"\n"
        print("Callpath:",callpath)

        for k in range(2):
            output_text += "\n"
            print("")
            output_text += "\tMetric: "+str(metrics[k])+"\n"
            print("\tMetric:",metrics[k])
            for j in range(len(experiment_eval.coordinates)):
                output_text += "\t\tCoordinate: "+str(experiment_eval.coordinates[j])+"\n"
                print("\t\tCoordinate:",experiment_eval.coordinates[j])

                measurement = experiment_eval.get_measurement(j, i, k, type)

                actual = measurement.mean

                # get the model and the function
                modeler = experiment.modelers[0]

                callpath_exists = True

                try:

                    model = modeler.models[callpath, metrics[k], type]

                except KeyError:
                    callpath_exists = False

                if callpath_exists == True:

                    hypothesis = model.hypothesis
                    function = hypothesis.function
                    rss = hypothesis.RSS
                    ar2 = hypothesis.AR2
                    rrss = hypothesis.rRSS
                    smape = hypothesis.SMAPE
                    re = hypothesis.RE
                    function_string = function.to_string(*experiment.parameters)

                    output_text += "\t\t\tModel: "+str(convert_to_py_function(function_string))+"\n"
                    print("\t\t\tModel:",convert_to_py_function(function_string))
                    text = "\t\t\tSMAPE: {:.3f}".format(smape)
                    text += " AR2: {:.3f}".format(ar2)
                    text += " RE: {:.3f}".format(re)
                    output_text += text+"\n"
                    print(text)

                    function = convert_to_py_function(function_string)

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

                    output_text += "\t\t\tActual: {:.9f}".format(actual)+"\n"
                    print("\t\t\tActual: {:.9f}".format(actual))
                    output_text += "\t\t\tPrediction: {:.9f}".format(prediction)+"\n"
                    print("\t\t\tPrediction: {:.9f}".format(prediction))

                    absolute_error = abs(prediction - actual)
                    output_text += "\t\t\tDivergence: {:.9f}".format(absolute_error)+"\n"
                    print("\t\t\tDivergence: {:.9f}".format(absolute_error))
                    relative_error = absolute_error / actual
                    relative_error = abs(relative_error)
                    percentage_error = relative_error * 100

                    output_text += "\t\t\tMedian Relative Error: {:.3f}".format(relative_error)+"\n"
                    print("\t\t\tMedian Relative Error: {:.3f}".format(relative_error))
                    output_text += "\t\t\tMedian Relative Error Percent: {:.3f}".format(percentage_error)+"\n"
                    print("\t\t\tMedian Relative Error Percent: {:.3f}".format(percentage_error))

                    data[j][k].append(percentage_error)

                    output_text += "\n"
                    print("")

    output_text += "------------------------------------------------------\n"
    output_text += "Statistics:\n"
    output_text += "\n"
    print("------------------------------------------------------\n")
    print("Statistics:")
    print("")
    output_text += "Number of analyzed kernels: "+str(len(callpaths))+"\n"
    print("Number of analyzed kernels: "+str(len(callpaths)))

    #print(data)

    for i in range(len(data)):
        print(experiment_eval.coordinates[i])
        for j in range(len(data[i])):
            #print(metrics[j])
            if str(metrics[j]) == "runtime":
                errors = data[i][j]
                for k in range(len(callpaths)):
                    print(callpaths[k],":",errors[k])

    x = {
        "coordinate": [],
        "metric": [],
        "error": []
    }

    import pandas as pd
    for i in range(len(data)):
        for j in range(len(data[i])):
            errors = data[i][j]
            for k in range(len(errors)):
                error = errors[k]
                temp = x["error"]
                temp.append(error)
                x["error"] = temp
                cord_string = str(experiment_eval.coordinates[i])
                temp = x["coordinate"]
                temp.append(cord_string)
                x["class"] = temp
                metric_str = str(metrics[j])
                temp = x["metric"]
                temp.append(metric_str)
                x["metric"] = temp

    df = pd.DataFrame(x)

    print(df)

    import seaborn
    import matplotlib.pyplot as plt
 
    # use to set style of background of plot
    seaborn.set(style="whitegrid")
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10,6))
    
    seaborn.violinplot(x="coordinate", y="error", hue="metric",
                        data=df, palette="Set2", split=True, bw=.15, cut=0, linewidth=1)

    plt.show()

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = np.median(data[i][j])

    for i in range(2):
        text = "MPE "+str(metrics[i].__str__())
        for j in range(len(experiment_eval.coordinates)):
            cord = "Coordinate: "+str(experiment_eval.coordinates[j])+" = "
            value = str(data[j][i])
            output_text += str(text)+" "+str(cord)+" "+str(value)+" \n"
            print(text, cord, value)

    p = []
    for i in range(len(metrics)):
        p.append([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            p[j].append(data[i][j])

    output_text += "\n"
    output_text += "------------------------------------------------------\n"
    print("")
    print("------------------------------------------------------\n")


    # plot results

    labels = []
    for i in range(len(experiment_eval.coordinates)):
        labels.append(str(experiment_eval.coordinates[i]))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    legend = []
    for i in range(len(metrics)):
        legend.append(metrics[i].__str__())

    fig, ax = plt.subplots()

    rects = []

    rect = ax.bar(x-width/2, p[0], width, label=legend[0])
    rects.append(rect)

    rect2 = ax.bar(x+width/2, p[1], width, label=legend[1])
    rects.append(rect2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MPE [%]')
    ax.set_xlabel('Evaluation points')
    ax.set_title('Prediction error at evaluation points per metric')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects, i):
        #Attach a text label above each bar in *rects*, displaying its height
        if i==0:
            offset = 15
        else:
            offset = 3
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for i in range(len(rects)):
        autolabel(rects[i], i)

    fig.tight_layout()

    plt.show()

    return output_text


def evaluate_phases(experiment_eval, experiment, arguments, application_name, plot):

    temp = []
    nr_dimensions = -1
    for coordinate_id in range(len(experiment.coordinates)):
        coordinate = experiment.coordinates[coordinate_id]
        dimensions = coordinate.dimensions
        nr_dimensions = dimensions
        if dimensions == 1:
            x = [int(coordinate_id), float(coordinate[0])]
            temp.append(x)
        elif dimensions == 2:
            x = [int(coordinate_id), float(coordinate[0]), float(coordinate[1])]
            temp.append(x)
        elif dimensions == 3:
            x = [int(coordinate_id), float(coordinate[0]), float(coordinate[1]), float(coordinate[2])]
            temp.append(x)

    # sort this list by the parameter values
    if dimensions == 1:
        temp = sorted(temp, key=lambda x: (x[1]))
    elif dimensions == 2:
        temp = sorted(temp, key=lambda x: (x[1], x[2]))
    elif dimensions == 3:
        temp = sorted(temp, key=lambda x: (x[1], x[2], x[3]))

    temp2 = []
    nr_dimensions = -1
    for coordinate_id in range(len(experiment_eval.coordinates)):
        coordinate = experiment_eval.coordinates[coordinate_id]
        dimensions = coordinate.dimensions
        nr_dimensions = dimensions
        if dimensions == 1:
            x = [int(coordinate_id), float(coordinate[0])]
            temp2.append(x)
        elif dimensions == 2:
            x = [int(coordinate_id), float(coordinate[0]), float(coordinate[1])]
            temp2.append(x)
        elif dimensions == 3:
            x = [int(coordinate_id), float(coordinate[0]), float(coordinate[1]), float(coordinate[2])]
            temp2.append(x)

    # sort this list by the parameter values
    if dimensions == 1:
        temp2 = sorted(temp2, key=lambda x: (x[1]))
    elif dimensions == 2:
        temp2 = sorted(temp2, key=lambda x: (x[1], x[2]))
    elif dimensions == 3:
        temp2 = sorted(temp2, key=lambda x: (x[1], x[2], x[3]))

    output_text = ""

    # do the evaluation

    print("\n")
    tab_string = get_tabulate_string("Measurement Data")
    print(tab_string+"\n")
    output_text += tab_string

    output_text += "\nNr. points used for modeling: "+str(len(experiment.coordinates))
    print("Nr. points used for modeling: "+str(len(experiment.coordinates)))
    output_text += "\nModeling points: "
    #print("\nModeling points:")
    points_string = ""
    for i in range(len(temp)):
        output_text += str(experiment.coordinates[temp[i][0]])
        #output_text += str(experiment.coordinates[i])
        #print(experiment.coordinates[temp[i][0]])
        if i == 0:
            points_string += str(experiment.coordinates[temp[i][0]])
        else:
            points_string += ", "+str(experiment.coordinates[temp[i][0]])
    print("\nModeling points: "+points_string)

    output_text += "\n"
    print("")
    output_text += "\nNr. points used for evaluation: "+str(len(experiment_eval.coordinates))
    print("Nr. points used for evaluation: "+str(len(experiment_eval.coordinates)))
    output_text += "\nEvaluation points: "
    points_string = ""
    for i in range(len(temp2)):
        #output_text += str(experiment_eval.coordinates[i])
        output_text += str(experiment_eval.coordinates[temp2[i][0]])
        #print(experiment_eval.coordinates[i])
        #print(experiment_eval.coordinates[temp2[i][0]])
        if i == 0:
            points_string += str(experiment_eval.coordinates[temp2[i][0]])
        else:
            points_string += ", "+str(experiment_eval.coordinates[temp2[i][0]])
    print("\nEvaluation points: "+points_string)

    print("")
    output_text += "\n"
    metrics = experiment.metrics
    output_text += "\nMetrics: "+str(metrics)
    print("Metrics:",metrics)
    print("")
    output_text += "\n"
    parameters = experiment.parameters
    output_text += "\nParameters: "+str(parameters)
    print("Parameters:",parameters)
    output_text += "\n"
    print("")

    tab_string = get_tabulate_string("Predictive Power Analysis")
    print(tab_string+"\n")
    output_text += tab_string

    # create a data container for the evaluation results
    data = []
    for i in range(len(experiment_eval.coordinates)):
        metrics = []
        for j in range(len(experiment_eval.metrics)):
            metrics.append([])
        data.append(metrics)

    # compute the evaluation results

    tree = Tree("Application: "+application_name)

    for i in range(len(experiment_eval.callpaths)):

        callpath = experiment_eval.callpaths[i]

        node = tree.add("Callpath: "+str(callpath))

        output_text += "Callpath: "+str(callpath)+"\n"
        #print("Callpath:",callpath)

        for k in range(len(experiment_eval.metrics)):

            output_text += "\n"
            #print("")
            output_text += "\tMetric: "+str(experiment_eval.metrics[k])+"\n"
            #print("\tMetric:",experiment_eval.metrics[k])

            measurement_found = False
            nodes_to_add = []

            for j in range(len(temp2)):

                cord_id = temp2[j][0]

                output_text += "\t\tCoordinate: "+str(experiment_eval.coordinates[cord_id])+"\n"
                #print("\t\tCoordinate:",experiment_eval.coordinates[cord_id])

                measurement = experiment_eval.get_measurement(cord_id, i, k)

                if measurement != None:
                    actual = measurement.mean

                    measurement_found = True

                    # get the model and the function
                    modeler = experiment.modelers[0]

                    callpath_exists = True

                    try:

                        model = modeler.models[callpath, experiment.metrics[k]]

                    except KeyError:
                        callpath_exists = False

                    if callpath_exists == True:

                        hypothesis = model.hypothesis
                        function = hypothesis.function
                        rss = hypothesis.RSS
                        ar2 = hypothesis.AR2
                        rrss = hypothesis.rRSS
                        smape = hypothesis.SMAPE
                        re = hypothesis.RE
                        function_string = function.to_string(*experiment.parameters)

                        output_text += "\t\t\tModel: "+str(convert_to_py_function(function_string))+"\n"
                        #print("\t\t\tModel:",convert_to_py_function(function_string))
                        text = "SMAPE: {:.3f}".format(smape)
                        text += " AR2: {:.3f}".format(ar2)
                        text += " RE: {:.3f}".format(re)
                        output_text += text+"\n"
                        #print(text)

                        function = convert_to_py_function(function_string)

                        parameters = arguments.param
                        parameters = parameters.split(",")

                        if len(experiment_eval.parameters) == 1:
                            pos = function.find(parameters[0])
                            found = False
                            if pos != -1:
                                found = True
                            while pos != -1:
                                temp = function[:pos]
                                temp += "x1"+function[pos+1:]
                                function = temp
                                pos = function.find(parameters[0])

                            if found == True:
                                x1 = experiment_eval.coordinates[cord_id][0]
                            prediction = eval(function)

                        elif len(experiment_eval.parameters) == 2:
                            pos = function.find(parameters[0])
                            found = False
                            if pos != -1:
                                found = True
                            while pos != -1:
                                temp = function[:pos]
                                temp += "x1"+function[pos+1:]
                                function = temp
                                pos = function.find(parameters[0])

                            pos = function.find(parameters[1])
                            found2 = False
                            if pos != -1:
                                found2 = True
                            while pos != -1:
                                temp = function[:pos]
                                temp += "x2"+function[pos+1:]
                                function = temp
                                pos = function.find(parameters[1])

                            if found == True:
                                x1 = experiment_eval.coordinates[cord_id][0]
                            if found2 == True:
                                x2 = experiment_eval.coordinates[cord_id][1]
                            prediction = eval(function)

                        elif len(experiment_eval.parameters) == 3:
                            pos = function.find(parameters[0])
                            found = False
                            if pos != -1:
                                found = True
                            while pos != -1:
                                temp = function[:pos]
                                temp += "x1"+function[pos+1:]
                                function = temp
                                pos = function.find(parameters[0])

                            pos = function.find(parameters[1])
                            found2 = False
                            if pos != -1:
                                found2 = True
                            while pos != -1:
                                temp = function[:pos]
                                temp += "x2"+function[pos+1:]
                                function = temp
                                pos = function.find(parameters[1])

                            pos = function.find(parameters[2])
                            found3 = False
                            if pos != -1:
                                found3 = True
                            while pos != -1:
                                temp = function[:pos]
                                temp += "x3"+function[pos+1:]
                                function = temp
                                pos = function.find(parameters[2])

                            if found == True:
                                x1 = experiment_eval.coordinates[cord_id][0]
                            if found2 == True:
                                x2 = experiment_eval.coordinates[cord_id][1]
                            if found3 == True:
                                x3 = experiment_eval.coordinates[cord_id][2]
                            prediction = eval(function)

                        output_text += "\t\t\tActual: {:.9f}".format(actual)+"\n"
                        #print("\t\t\tActual: {:.9f}".format(actual))
                        output_text += "\t\t\tPrediction: {:.9f}".format(prediction)+"\n"
                        #print("\t\t\tPrediction: {:.9f}".format(prediction))

                        absolute_error = abs(prediction - actual)
                        output_text += "\t\t\tDivergence: {:.9f}".format(absolute_error)+"\n"
                        #print("\t\t\tDivergence: {:.9f}".format(absolute_error))
                        relative_error = absolute_error / actual
                        relative_error = abs(relative_error)
                        percentage_error = relative_error * 100

                        output_text += "\t\t\tMedian Relative Error: {:.3f}".format(relative_error)+"\n"
                        #print("\t\t\tMedian Relative Error: {:.3f}".format(relative_error))
                        output_text += "\t\t\tMedian Relative Error Percent: {:.3f}".format(percentage_error)+"\n"
                        #print("\t\t\tMPE: {:.3f}".format(percentage_error)+"%")

                        data[cord_id][k].append(percentage_error)

                        output_text += "\n"
                        #print("")

                    else:
                        measurement_found = False

                if measurement != None:
                    model_string = "\n\tModel:"+convert_to_py_function(function_string)
                    actual_string = "\n\tActual: {:.9f}".format(actual)
                    prediction_string = "\n\tPrediction: {:.9f}".format(prediction)
                    diverg_string = "\n\tDivergence: {:.9f}".format(absolute_error)
                    mpe_string = "\n\tMPE: {:.3f}".format(percentage_error)+"%"
                    #subsubnode = subnode.add("Coordinate: "+str(experiment_eval.coordinates[cord_id])+model_string+"\n\t"+text+actual_string+prediction_string+diverg_string+mpe_string+"\n")
                    nodes_to_add.append("Coordinate: "+str(experiment_eval.coordinates[cord_id])+model_string+"\n\t"+text+actual_string+prediction_string+diverg_string+mpe_string+"\n")

            if measurement_found == True:
                subnode = node.add("Metric: "+str(experiment_eval.metrics[k]))

                for o in range(len(nodes_to_add)):
                    subsubnode = subnode.add(nodes_to_add[o])


    # print the results
    richprint(tree)

    tab_string = get_tabulate_string("Statistics")
    print(tab_string+"\n")

    output_text += tab_string
    output_text += "\n"

    comp = []
    com = []
    mem = []
    mem2 = []

    for i in range(len(data)):
        comp.append(data[i][0][0])
        com.append(data[i][0][1])
        mem.append(data[i][0][2])
        mem2.append(data[i][1][0])

    for i in range(len(experiment_eval.coordinates)):
        text = "MPE computation "+str(experiment.metrics[0].__str__())
        cord = str(experiment_eval.coordinates[i])+" ="
        value = comp[i]
        value = "{:.3f}".format(value)+"%"
        output_text += str(text)+" "+str(cord)+" "+value+" \n"
        print(text, cord, value)

        text = "MPE communication "+str(experiment.metrics[0].__str__())
        cord = str(experiment_eval.coordinates[i])+" ="
        value = com[i]
        value = "{:.3f}".format(value)+"%"
        output_text += str(text)+" "+str(cord)+" "+value+" \n"
        print(text, cord, value)

        text = "MPE memory operations "+str(experiment.metrics[0].__str__())
        cord = str(experiment_eval.coordinates[i])+" ="
        value = mem[i]
        value = "{:.3f}".format(value)+"%"
        output_text += str(text)+" "+str(cord)+" "+value+" \n"
        print(text, cord, value)

    for i in range(len(experiment_eval.coordinates)):
        text = "MPE memory operations "+str(experiment.metrics[1].__str__())
        cord = str(experiment_eval.coordinates[i])+" ="
        value = mem2[i]
        value = "{:.3f}".format(value)+"%"
        output_text += str(text)+" "+str(cord)+" "+value+" \n"
        print(text, cord, value)

    p = []
    for i in range(len(experiment.metrics)):
        p.append([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            p[j].append(data[i][j])

    output_text += "\n"
    print("")

    # plot results
    if plot == True:

        labels = []
        for i in range(len(experiment_eval.coordinates)):
            labels.append(str(experiment_eval.coordinates[i]))

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        legend = []
        for i in range(len(experiment.callpaths)):
            legend.append(experiment.callpaths[i].__str__())

        fig, ax = plt.subplots()

        rects = []

        rect = ax.bar(x-width, comp, width, label=legend[0])
        rects.append(rect)

        rect = ax.bar(x, com, width, label=legend[1])
        rects.append(rect)

        rect = ax.bar(x+width, mem, width, label=legend[2])
        rects.append(rect)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MPE [%]')
        ax.set_xlabel('Evaluation points')
        ax.set_title('Prediction error at evaluation points for application phases and the runtime metric.')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects, i):
            #Attach a text label above each bar in *rects*, displaying its height
            if i==0:
                offset = 15
            else:
                offset = 3
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        for i in range(len(rects)):
            autolabel(rects[i], i)

        fig.tight_layout()

        plt.show()


        ###############################################

        # plot results

        labels = []
        for i in range(len(experiment_eval.coordinates)):
            labels.append(str(experiment_eval.coordinates[i]))

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        legend = []
        for i in range(len(experiment.callpaths)):
            legend.append(experiment.callpaths[i].__str__())

        fig, ax = plt.subplots()

        rects = []

        rect = ax.bar(x, mem2, width, label=legend[2])
        rects.append(rect)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MPE [%]')
        ax.set_xlabel('Evaluation points')
        ax.set_title('Prediction error at evaluation points for the number of transferred bytes for the application phases.')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects, i):
            #Attach a text label above each bar in *rects*, displaying its height
            if i==0:
                offset = 15
            else:
                offset = 3
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        for i in range(len(rects)):
            autolabel(rects[i], i)

        fig.tight_layout()

        plt.show()

    return output_text


def evaluate_memory():

    pass



def evaluate_gpu_util(experiment_eval, experiment, arguments):
    output_text = ""

    # do the evaluation
    output_text += "\n-------------------\n|    Analysis  |\n--------------------\n"
    print("\n------------------------------\n|    Analysis  |\n------------------------------\n")
    output_text += "\nNr. points used for modeling: "+str(len(experiment.coordinates))
    print("Nr. points used for modeling: "+str(len(experiment.coordinates)))
    output_text += "\nModeling points: "
    print("Modeling points:")
    for i in range(len(experiment.coordinates)):
        output_text += str(experiment.coordinates[i])
        print(experiment.coordinates[i])
    output_text += "\n"
    print("")
    output_text += "\nNr. points used for evaluation : "+str(len(experiment_eval.coordinates))
    print("Nr. points used for evaluation:"+str(len(experiment_eval.coordinates)))
    output_text += "\nEvaluation points: "
    print("Evaluation points:")
    for i in range(len(experiment_eval.coordinates)):
        output_text += str(experiment_eval.coordinates[i])
        print(experiment_eval.coordinates[i])
    print("")
    output_text += "\n"
    metrics = experiment.metrics
    output_text += "\nMetrics: "+str(metrics)
    print("Metrics:",metrics)
    print("")
    output_text += "\n"
    parameters = experiment.parameters
    output_text += "\nParameters: "+str(parameters)
    print("Parameters:",parameters)
    output_text += "\n"
    print("")

    output_text += "\n-------------------------------------\n|    Predictive Power Analysis  |\n-------------------------------------\n"
    print("\n---------------\n|    Predictive Power Analysis  |\n---------------\n")

    # create a data container for the evaluation results
    data = []
    for i in range(len(experiment_eval.coordinates)):
        metrics = []
        for j in range(len(experiment_eval.metrics)):
            callpaths = []
            for k in range(len(experiment_eval.callpaths)):
                callpaths.append([])
            metrics.append(callpaths)
        data.append(metrics)

    # compute the evaluation results
    for i in range(len(experiment_eval.callpaths)):

        output_text += "\n"
        print("")

        callpath = experiment_eval.callpaths[i]
        output_text += "Callpath: "+str(callpath)+"\n"
        print("Callpath:",callpath)

        for k in range(len(experiment_eval.metrics)):
            output_text += "\n"
            print("")
            output_text += "\tMetric: "+str(experiment_eval.metrics[k])+"\n"
            print("\tMetric:",experiment_eval.metrics[k])
            for j in range(len(experiment_eval.coordinates)):
                output_text += "\t\tCoordinate: "+str(experiment_eval.coordinates[j])+"\n"
                print("\t\tCoordinate:",experiment_eval.coordinates[j])

                measurement = experiment_eval.get_measurement(j, i, k)

                actual = measurement.mean

                # get the model and the function
                modeler = experiment.modelers[0]
                model = modeler.models[callpath, experiment.metrics[k]]

                hypothesis = model.hypothesis
                function = hypothesis.function
                rss = hypothesis.RSS
                ar2 = hypothesis.AR2
                rrss = hypothesis.rRSS
                smape = hypothesis.SMAPE
                re = hypothesis.RE
                function_string = function.to_string(*experiment.parameters)

                output_text += "\t\t\tModel: "+str(convert_to_py_function(function_string))+"\n"
                print("\t\t\tModel:",convert_to_py_function(function_string))
                text = "\t\t\tSMAPE: {:.3f}".format(smape)
                text += " AR2: {:.3f}".format(ar2)
                text += " RE: {:.3f}".format(re)
                output_text += text+"\n"
                print(text)

                function = convert_to_py_function(function_string)

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

                output_text += "\t\t\tActual: {:.9f}".format(actual)+"\n"
                print("\t\t\tActual: {:.9f}".format(actual))
                output_text += "\t\t\tPrediction: {:.9f}".format(prediction)+"\n"
                print("\t\t\tPrediction: {:.9f}".format(prediction))

                absolute_error = abs(prediction - actual)
                output_text += "\t\t\tDivergence: {:.9f}".format(absolute_error)+"\n"
                print("\t\t\tDivergence: {:.9f}".format(absolute_error))
                if absolute_error == 0:
                    relative_error = 0
                else:
                    relative_error = absolute_error / actual
                    relative_error = abs(relative_error)
                percentage_error = relative_error * 100

                output_text += "\t\t\tMedian Relative Error: {:.3f}".format(relative_error)+"\n"
                print("\t\t\tMedian Relative Error: {:.3f}".format(relative_error))
                output_text += "\t\t\tMedian Relative Error Percent: {:.3f}".format(percentage_error)+"\n"
                print("\t\t\tMedian Relative Error Percent: {:.3f}".format(percentage_error))

                data[j][k][i].append(percentage_error)

                output_text += "\n"
                print("")

    print("------------------------------------------------------\n")
    print("Statistics:")
    print("")
    output_text += "------------------------------------------------------\n"
    output_text += "Statistics:\n"
    output_text += "\n"

    for k in range(len(experiment.callpaths)):
        for i in range(len(experiment.metrics)):
            text = ""+str(experiment.callpaths[k].__str__())+", "+str(experiment.metrics[i].__str__())
            for j in range(len(experiment_eval.coordinates)):
                cord = "Coordinate: "+str(experiment_eval.coordinates[j])+" = "
                value = str(data[j][i][k][0])
                output_text += str(text)+" "+str(cord)+" "+str(value)+"\n"
                print(text, cord, value)

    p2 = []
    for i in range(len(experiment.metrics)):
        p2.append([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            p2[j].append(data[i][j][0][0])

    p3 = []
    for i in range(len(experiment.metrics)):
        p3.append([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            p3[j].append(data[i][j][1][0])

    output_text += "\n"

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = np.median(data[i][j])

    for i in range(len(experiment.metrics)):
        text = "MPE "+str(experiment.metrics[i].__str__())
        for j in range(len(experiment_eval.coordinates)):
            cord = "Coordinate: "+str(experiment_eval.coordinates[j])+" = "
            value = str(data[j][i])
            output_text += str(text)+" "+str(cord)+" "+str(value)+"\n"
            print(text, cord, value)

    p = []
    for i in range(len(experiment.metrics)):
        p.append([])
    for i in range(len(data)):
        for j in range(len(data[i])):
            p[j].append(data[i][j])

    print("")
    print("------------------------------------------------------\n")
    output_text += "\n"
    output_text += "------------------------------------------------------\n"


    # plot results

    labels = []
    for i in range(len(experiment_eval.coordinates)):
        labels.append(str(experiment_eval.coordinates[i]))

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    legend = []
    for i in range(len(experiment.metrics)):
        legend.append("average "+experiment.metrics[i].__str__())
        legend.append(experiment.metrics[i].__str__()+" per step")
        legend.append(experiment.metrics[i].__str__()+" between steps")

    fig, ax = plt.subplots()

    rects = []
    for i in range(len(experiment.metrics)):
            rect = ax.bar(x-width, p[i], width, label=legend[0])
            rects.append(rect)
            rect = ax.bar(x, p2[i], width, label=legend[1])
            rects.append(rect)
            rect = ax.bar(x+width, p3[i], width, label=legend[2])
            rects.append(rect)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Median Percentage Error [%]')
    ax.set_xlabel('Number of MPI ranks p')
    ax.set_title('Prediction error at evaluation points per metric')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        #Attach a text label above each bar in *rects*, displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for i in range(len(rects)):
        autolabel(rects[i])

    fig.tight_layout()

    plt.show()

    return output_text
