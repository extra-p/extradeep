# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List

from extradeep.entities.callpath import Callpath
from extradeep.entities.calltree import CallTree
from extradeep.entities.calltree import Node
from extradeep.entities.measurement import Measurement
from extradeep.util.exceptions import InvalidExperimentError
from extradeep.util.progress_bar import DUMMY_PROGRESS

if TYPE_CHECKING:
    from extradeep.entities.experiment import Experiment


def get_current_analysis_type(experiment, type_name):
    analysis_type_id = None
    for i in range(len(experiment.analysistypes)):
        if experiment.analysistypes[i].name == type_name:
            analysis_type_id = i
    current_analysis_type = experiment.analysistypes[analysis_type_id]
    return current_analysis_type


def format_callpaths(experiment, analysistype_name):
    """
    This method formats the ouput so that only the callpaths are shown.
    """
    current_analysis_type = get_current_analysis_type(experiment, analysistype_name)
    callpaths = experiment.callpaths[current_analysis_type]
    text = ""
    for callpath_id in range(len(callpaths)):
        callpath = callpaths[callpath_id]
        callpath_string = callpath.name
        text += callpath_string + "\n"
    return text


def format_metrics(experiment, analysistype_name):
    """
    This method formats the ouput so that only the metrics are shown.
    """
    current_analysis_type = get_current_analysis_type(experiment, analysistype_name)
    metrics = experiment.metrics[current_analysis_type]
    text = ""
    for metric_id in range(len(metrics)):
        metric = metrics[metric_id]
        metric_string = metric.name
        text += metric_string + "\n"
    return text


def format_parameters(experiment):
    """
    This method formats the ouput so that only the parameters are shown.
    """
    parameters = experiment.parameters
    text = ""
    for parameters_id in range(len(parameters)):
        parameter = parameters[parameters_id]
        parameter_string = parameter.name
        text += parameter_string + "\n"
    return text


def format_functions(experiment):
    """
    This method formats the ouput so that only the functions are shown.
    """

    # if loading data from extradeep data set instead from raw data
    # we want to use the latest model generator, otherwise the arguments for the modeler
    # passed to the command line do not influence the output shown in the terminal
    if len(experiment.modelers) > 1:
        modeler = experiment.modelers[len(experiment.modelers)-1]
    else:
        modeler = experiment.modelers[0]
    models = modeler.models
    text = ""
    for model in models.values():
        hypothesis = model.hypothesis
        function = hypothesis.function
        function_string = function.to_string(*experiment.parameters)
        text += function_string + "\n"
    return text


def format_all(experiment):
    """
    This method formats the ouput so that all information is shown.
    """
    coordinates = experiment.coordinates
    callpaths = experiment.callpaths
    metrics = experiment.metrics
    analysistype = experiment.analysistypes

    # if loading data from extradeep data set instead from raw data
    # we want to use the latest model generator, otherwise the arguments for the modeler
    # passed to the command line do not influence the output shown in the terminal
    if len(experiment.modelers) > 1:
        modeler = experiment.modelers[len(experiment.modelers)-1]
    else:
        modeler = experiment.modelers[0]

    # get a list of the coordinates with their parameter values and corresponding ids
    temp = []
    nr_dimensions = -1
    for coordinate_id in range(len(coordinates)):
        coordinate = coordinates[coordinate_id]
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

    # get the default analysis type
    type = analysistype[0]
    # retrive the callpath list
    callpaths = callpaths[type]
    # retrieve the metrics list
    metrics = metrics[type]

    # prepare the output according to the ordered point list
    text = ""
    for callpath_id in range(len(callpaths)):
        callpath = callpaths[callpath_id]
        callpath_string = callpath.name
        text += "Callpath: " + callpath_string + "\n"
        for metric_id in range(len(metrics)):
            metric = metrics[metric_id]
            metric_string = metric.name
            text += "\tMetric: " + metric_string + "\n"
            for i in range(len(coordinates)):
                coordinate_id = temp[i][0]
                coordinate = coordinates[coordinate_id]
                dimensions = coordinate.dimensions
                coordinate_text = "Measurement point: ("
                for dimension in range(dimensions):
                    value = coordinate[dimension]
                    #value_string = "{:.2E}".format(value)
                    #coordinate_text += value_string + ","
                    coordinate_text += str(value) + ","
                coordinate_text = coordinate_text[:-1]
                coordinate_text += ")"
                measurement = experiment.get_measurement(coordinate_id, callpath_id, metric_id, type)
                if measurement == None:
                    value_mean = 0
                    value_median = 0
                else:
                    value_mean = measurement.mean
                    value_median = measurement.median
                #text += f"\t\t{coordinate_text} Mean: {value_mean:.2E} Median: {value_median:.2E}\n"
                text += f"\t\t{coordinate_text} Mean: {value_mean} Median: {value_median}\n"
            try:
                model = modeler.models[callpath, metric, type]
            except KeyError as e:
                model = None
            if model != None:
                hypothesis = model.hypothesis
                function = hypothesis.function
                rss = hypothesis.RSS
                ar2 = hypothesis.AR2
                function_string = function.to_string(*experiment.parameters)
            else:
                rss = 0
                ar2 = 0
                function_string = "None"
            text += "\t\tModel: " + function_string + "\n"
            #text += "\t\tRSS: {:.2E}\n".format(rss)
            #text += "\t\tAdjusted R^2: {:.2E}\n".format(ar2)
            text += "\t\tRSS: {}\n".format(rss)
            text += "\t\tAdjusted R^2: {}\n".format(ar2)
    return text


def format_output(experiment, printtype, analysistype):
    """
    This method formats the ouput of the modeler to a string that can be printed in the console
    or to a file. Depending on the given options only parts of the modelers output get printed.
    """

    analysistype_name = ""
    if analysistype == "nvtx":
        analysistype_name = "NVTX user instrumentation"
    if analysistype == "cuda-kernel":
        analysistype_name = "CUDA kernel"
    if analysistype == "os":
        analysistype_name = "OS events"
    if analysistype == "mpi":
        analysistype_name = "MPI events"
    if analysistype == "cublas":
        analysistype_name = "CUBLAS events"
    if analysistype == "cudnn":
        analysistype_name = "CUDNN events"
    if analysistype == "memory":
        analysistype_name = "Memory events"
    if analysistype == "cuda-api":
        analysistype_name = "CUDA API events"
    if analysistype == "epochs":
        analysistype_name = "Epochs"
    if analysistype == "training-steps":
        analysistype_name = "Training steps"
    if analysistype == "validation-steps":
        analysistype_name = "Validation steps"
    if analysistype == "phases":
        analysistype_name = "Application phases"
    if analysistype == "gpu-util":
        analysistype_name = "GPU utilization"

    if printtype == "ALL":
        text = format_all(experiment)
    elif printtype == "CALLPATHS":
        text = format_callpaths(experiment, analysistype_name)
    elif printtype == "METRICS":
        text = format_metrics(experiment, analysistype_name)
    elif printtype == "PARAMETERS":
        text = format_parameters(experiment)
    elif printtype == "FUNCTIONS":
        text = format_functions(experiment)
    else:
        raise ValueError('printtype does not exist')
    return text


def save_output(text, path):
    """
    This method saves the output of the modeler, i.e. it's results to a text file at the given path.
    """
    with open(path, "w+") as out:
        out.write(text)


def append_to_repetition_dict(complete_data, key, coordinate, value, progress_bar=DUMMY_PROGRESS):
    if isinstance(value, list):
        if key in complete_data:
            if coordinate in complete_data[key]:
                complete_data[key][coordinate].extend(value)
            else:
                complete_data[key][coordinate] = value
                progress_bar.total += 1
        else:
            complete_data[key] = {
                coordinate: value
            }
            progress_bar.total += 1
    else:
        if key in complete_data:
            if coordinate in complete_data[key]:
                complete_data[key][coordinate].append(value)
            else:
                complete_data[key][coordinate] = [value]
                progress_bar.total += 1
        else:
            complete_data[key] = {
                coordinate: [value]
            }
            progress_bar.total += 1


def repetition_dict_to_experiment(complete_data, experiment, progress_bar=DUMMY_PROGRESS):
    progress_bar.step('Creating experiment')
    for mi, key in enumerate(complete_data):
        progress_bar.update()
        callpath, metric = key
        measurementset = complete_data[key]
        experiment.add_callpath(callpath)
        experiment.add_metric(metric)
        for coordinate in measurementset:
            values = measurementset[coordinate]
            experiment.add_coordinate(coordinate)
            experiment.add_measurement(Measurement(coordinate, callpath, metric, values))


def create_call_tree(callpaths: List[Callpath], progress_bar=DUMMY_PROGRESS, progress_total_added=False,
                     progress_scale=1):
    """
    This method creates the call tree object from the callpaths read.
    It builds a structure with a root node and child nodes.
    It can be used to display the callpaths in a tree structure.
    However, this method only works if the read callpaths are in
    the correct order, as they would appear in the real program.
    """
    tree = CallTree()
    progress_bar.step('Creating calltree')
    # create a two dimensional array of the callpath elements as strings
    callpaths2 = []
    max_length = 0

    if not progress_total_added:
        progress_bar.total += len(callpaths) * progress_scale

    for splitted_callpath in callpaths:
        callpath_string = splitted_callpath.name
        elems = callpath_string.split("->")
        callpaths2.append(elems)
        progress_bar.total += len(elems) * progress_scale
        progress_bar.update(progress_scale)
        if len(elems) > max_length:
            max_length = len(elems)

    # iterate over the elements of one call path
    for i in range(max_length):
        # iterate over all callpaths
        for callpath, splitted_callpath in zip(callpaths, callpaths2):
            # check that we do not try to access an element that does not exist
            if i >= len(splitted_callpath):
                continue
            # if the element does exist
            progress_bar.update(progress_scale)
            callpath_string = splitted_callpath[i]

            # when at root level
            if i == 0:
                root_node = tree
            # when not at root level, the previous nodes of the elements have to be checked
            else:
                # find the root node of the element that we want to add currently
                root_node = find_root_node(splitted_callpath, tree, i)

            # check if that child node is already existing
            child_node = root_node.find_child(callpath_string)
            is_leaf = i == len(splitted_callpath) - 1
            if child_node:
                if is_leaf:
                    if child_node.path == Callpath.EMPTY:
                        child_node.path = callpath
                    else:
                        warnings.warn("Duplicate callpath encountered, only first occurence is retained.")

            else:
                # add a new child node to the root node
                if is_leaf:
                    child_node = Node(callpath_string, callpath)
                else:
                    child_node = Node(callpath_string, Callpath.EMPTY)
                root_node.add_child_node(child_node)

    return tree


def find_root_node(callpath_elements, tree, loop_id):
    """
    This method finds the root node of a element in the callpath tree.
    Therefore, it searches iterativels through the tree.
    """
    level = 0
    root_element_string = callpath_elements[level]
    root_node = tree.get_node(root_element_string)

    # root node already found
    if loop_id == level + 1:
        return root_node

    # need to search deeper in the tree for the root node
    else:
        return find_child_node(root_node, level, callpath_elements, loop_id)


def find_child_node(root_node, level, callpath_elements, loop_id):
    """
    This method searches for a child node in the tree. Searches iteratively
    into the three and each nodes child nodes. Returns the root node of the
    child.
    """
    level = level + 1
    root_element_string = callpath_elements[level]
    childs = root_node.childs

    for i in range(len(childs)):
        child_name = childs[i].name

        if child_name == root_element_string:
            new_root_node = childs[i]

            # root node already found
            if loop_id == level + 1:
                return new_root_node

            # need to search deeper in the tree for the root node
            else:
                return find_child_node(new_root_node, level, callpath_elements, loop_id)


def validate_experiment(experiment: Experiment, progress_bar=DUMMY_PROGRESS):
    def require(cond, message):
        if not cond:
            raise InvalidExperimentError(message)

    progress_bar.step('Validating experiment')

    length_parameters = len(experiment.parameters)
    require(length_parameters > 0, "Parameters are missing.")
    length_coordinates = len(experiment.coordinates)
    require(length_coordinates > 0, "Coordinates are missing.")
    require(len(experiment.metrics) > 0, "Metrics are missing.")
    require(len(experiment.analysistypes) > 0, "Analysis types are missing.")
    require(len(experiment.callpaths) > 0, "Callpaths are missing.")
    require(len(experiment.call_tree.childs) > 0, "Calltree is missing.")
    for c in experiment.coordinates:
        require(len(c) == length_parameters,
                f'The number of coordinate units of {c} does not match the number of '
                f'parameters ({length_parameters}).')

    for k, m in progress_bar(experiment.measurements.items(), len(experiment.measurements)):
        require(len(m) == length_coordinates,
                f'The number of measurements ({len(m)}) for {k} does not match the number of coordinates '
                f'({length_coordinates}).')
