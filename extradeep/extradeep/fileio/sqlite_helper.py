# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from tkinter import filedialog
from tkinter import *
from os import listdir
from os.path import isfile, join
import sqlite3
import re
import numpy as np

#from extradeep.entities.dl.epoch import Epoch
from extradeep.entities.nvtx.nvtx_event import NVTX_Event
#from extradeep.entities.nvtx.nvtx_mark import NVTX_Mark
#from extradeep.entities.dl.testing_step import TestingStep
#from extradeep.entities.dl.testing import Testing
#from extradeep.entities.dl.training_step import TrainingStep
#from extradeep.entities.dl.training import Training
#from extradeep.fileio.io_helper import *
#from extradeep.util.util_functions import *
from extradeep.entities.dl.experiment_config import ExperimentConfig
from extradeep.entities.cupti.cupti_kernel import CuptiKernel
#from extradeep.util.data_explorer_view_model import DataExplorerViewModel
#from extradeep.util.time_line_plot_model import TimeLinePlotModel
from extradeep.entities.cupti.cupti_runtime import CuptiRuntime
from extradeep.entities.cudnn.cudnn_events import CudnnEvent
from extradeep.entities.cublas.cublas_events import CublasEvent
from extradeep.entities.cupti.cupti_memset import CuptiMemset
from extradeep.entities.cupti.cupti_memcopy import CuptiMemcopy
from extradeep.entities.cupti.cupti_synchronization import CuptiSynchronization
from extradeep.entities.mpi.mpi_events import MPIEvent
from extradeep.entities.os.os_event import OSEvent
from extradeep.entities.cupti.cupti_kernel_launch import CuptiKernelLaunch
#from extradeep.entities.horovod.horovod_events import HorovodEvent

def populate_data_explorer_view_model(path, start, end):
    """
    populate_data_explorer_view_model function to load the data from the database and populate a data_explorer_view_model with it

    :param path: the path of the experiment as a string value
    :param start: the start time of the selected time interval in the selection as long
    :param end:  the end time of the selected time interval in the selection as long
    :return data_explorer_view_model, timelineplot_model: the DataExplorerViewModel containing the data to display in the event list and TimeLinePlot
    """

    # get the cupti kernels list
    cupti_kernels_list, cupti_kernels = get_cupti_kernels_list(path, start, end)
    cupti_runtimes_list, cupti_runtimes = get_cupti_runtime_list(path, start, end)
    cudnn_events_list, cudnn_events = get_cudnn_events_list(path, start, end)
    cublas_events_list, cublas_events = get_cublas_events_list(path, start, end)

    event_types = []
    event_types.append("CUDA Kernel")
    event_types.append("CUDA API Call")
    event_types.append("CUDNN Events")
    event_types.append("CUBLAS Events")

    data_explorer_view_model = DataExplorerViewModel(event_types, cupti_kernels_list, cupti_runtimes_list, cudnn_events_list, cublas_events_list)

    time_line_plot_model = TimeLinePlotModel(cupti_kernels, cupti_runtimes, cudnn_events, cublas_events)

    return data_explorer_view_model, time_line_plot_model

def get_nvtx_events_list(nvtx_events):
    """
    get_nvtx_events_list functionn to compute and create a list of nvtx events in the selected time interval with the interesting metrics

    :param nvtx_events: a list of nvtx event objects
    :return nvtx_events_list: the list of nvtx objects containing the metrics for the view model
    """

    # get a list of unique nvtx events in this time interval
    unique = []
    for i in range(len(nvtx_events)):
        unique.append(nvtx_events[i].callpath_name)
    short_names = get_unique_numbers(unique)

    # sum up the duration of each unique cupti kernel in seconds
    runtime_dict = {}
    for i in range(len(short_names)):
        runtime_dict[short_names[i]] = 0.0
    for i in range(len(nvtx_events)):
        runtime_dict[nvtx_events[i].callpath_name] += nvtx_events[i].run_time_seconds

    # sum up the number of visits of each unique cupti kernel
    visits_dict = {}
    for i in range(len(short_names)):
        visits_dict[short_names[i]] = 0
    for i in range(len(nvtx_events)):
        visits_dict[nvtx_events[i].callpath_name] += 1

    # compute the median runtime in seconds for each unique cupti kernel
    median_dict = {}
    for i in range(len(short_names)):
        median_dict[short_names[i]] = []
    for i in range(len(nvtx_events)):
        median_dict[nvtx_events[i].callpath_name].append(nvtx_events[i].run_time_seconds)
    for i in range(len(nvtx_events)):
        median_dict[nvtx_events[i].callpath_name] = np.median(median_dict[nvtx_events[i].callpath_name])

    # assemble a list with arrays containing the information for each unique event and its metrics
    nvtx_events_list = []
    for i in range(len(short_names)):
        nvtx_events_list.append([
        str(short_names[i]),
        str(visits_dict[short_names[i]]),
        str(runtime_dict[short_names[i]]),
        str(median_dict[short_names[i]])
        ])

    return nvtx_events_list

def get_cupti_kernels_list(path, start, end):
    """
    get_cupti_kernels_list function to get a list of cupti kernel events and their metrics

    :param path: the path as string for the experiment database to be read
    :param start: the start time of the selected time interval in the selection as long
    :param end:  the end time of the selected time interval in the selection as long
    :return cupti_kernels_list: the list of events and their metrics of the cupti kernels for the time line plot view
    :return cupti_kernels: the list for the event list widget containing the computet metrics
    """

    # open database in path and create connection
    db = sqlite3.connect(path)
    cursor = db.cursor()

    # get the unique cupti kernel events in this time interval
    query = "SELECT Distinct strings.value, strings2.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel LEFT JOIN StringIds AS strings ON kernel.demangledName = strings.id LEFT JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE kernel.start >="+str(start)+" AND kernel.end <="+str(end)+";"
    unique_cupti_events = get_data_from_db(cursor, query)

    # load cupti events with interesting infors from database
    query = "SELECT kernel.start, kernel.end, (end - start) AS duration, (1.0 * kernel.start / 1000000000) AS start_seconds, (1.0 * kernel.end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, strings.value AS demangledName, strings2.value AS shortName, ('(' || gridX || ',' || gridY || ',' || gridZ || ')') AS grid, ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block, staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted, registersPerThread, localMemoryTotal, localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel INNER JOIN StringIds AS strings ON kernel.demangledName = strings.id INNER JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
    cupti_events = get_data_from_db(cursor, query)
    cupti_kernels = convert_cupti_kernels_to_objects(cupti_events)

    # get a list of unique cupti kernel events in this time interval
    #unique = []
    #for i in range(len(cupti_kernels)):
    #    unique.append(cupti_kernels[i].shortName)
    #short_names = get_unique_numbers(unique)

    # sum up the duration of each unique cupti kernel in seconds
    runtime_dict = {}
    for i in range(len(unique_cupti_events)):
        runtime_dict[unique_cupti_events[i][0]] = 0.0
    for i in range(len(cupti_kernels)):
        runtime_dict[cupti_kernels[i].demangledName] += cupti_kernels[i].duration_seconds

    # sum up the number of visits of each unique cupti kernel
    visits_dict = {}
    for i in range(len(unique_cupti_events)):
        visits_dict[unique_cupti_events[i][0]] = 0
    for i in range(len(cupti_kernels)):
        visits_dict[cupti_kernels[i].demangledName] += 1

    # compute the median runtime in seconds for each unique cupti kernel
    median_dict = {}
    for i in range(len(unique_cupti_events)):
        median_dict[unique_cupti_events[i][0]] = []
    for i in range(len(cupti_kernels)):
        median_dict[cupti_kernels[i].demangledName].append(cupti_kernels[i].duration_seconds)
    for i in range(len(cupti_kernels)):
        median_dict[cupti_kernels[i].demangledName] = np.median(median_dict[cupti_kernels[i].demangledName])

    # assemble a list with arrays containing the information for each unique event and its metrics
    cupti_kernels_list = []
    for i in range(len(unique_cupti_events)):
        cupti_kernels_list.append([
        str(unique_cupti_events[i][1]),
        str(visits_dict[unique_cupti_events[i][0]]),
        str(runtime_dict[unique_cupti_events[i][0]]),
        str(median_dict[unique_cupti_events[i][0]]),
        str(unique_cupti_events[i][0])
        ])

    return cupti_kernels_list, cupti_kernels

def get_cupti_runtime_list(path, start, end):
    """
    get_cupti_runtime_list function to get a list of cupti runtime events and their metrics

    :param path: the path as string for the experiment database to be read
    :param start: the start time of the selected time interval in the selection as long
    :param end:  the end time of the selected time interval in the selection as long
    :return cupti_runtime_list: the list of events and their metrics of the cupti kernels
    """

    # open database in path and create connection
    db = sqlite3.connect(path)
    cursor = db.cursor()

    # load cupti runtime events from database
    query = "SELECT runtime.start, runtime.end, runtime.eventClass, runtime.globalTid, runtime.correlationId, runtime.returnValue, runtime.callchainId, strings.value FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime INNER JOIN StringIds AS strings ON runtime.nameId = strings.id WHERE runtime.start>="+str(start)+" AND runtime.end<="+str(end)+";"
    cupti_events = get_data_from_db(cursor, query)
    cupti_runtimes = convert_cupti_runtimes_to_objects(cupti_events)

    # get a list of unique cupti kernel events in this time interval
    unique = []
    for i in range(len(cupti_runtimes)):
        unique.append(cupti_runtimes[i].shortName)
    short_names = get_unique_numbers(unique)

    # sum up the duration of each unique cupti kernel in seconds
    runtime_dict = {}
    for i in range(len(short_names)):
        runtime_dict[short_names[i]] = 0.0
    for i in range(len(cupti_runtimes)):
        runtime_dict[cupti_runtimes[i].shortName] += cupti_runtimes[i].duration_seconds

    # sum up the number of visits of each unique cupti kernel
    visits_dict = {}
    for i in range(len(short_names)):
        visits_dict[short_names[i]] = 0
    for i in range(len(cupti_runtimes)):
        visits_dict[cupti_runtimes[i].shortName] += 1

    # compute the median runtime in seconds for each unique cupti kernel
    median_dict = {}
    for i in range(len(short_names)):
        median_dict[short_names[i]] = []
    for i in range(len(cupti_runtimes)):
        median_dict[cupti_runtimes[i].shortName].append(cupti_runtimes[i].duration_seconds)
    for i in range(len(cupti_runtimes)):
        median_dict[cupti_runtimes[i].shortName] = np.median(median_dict[cupti_runtimes[i].shortName])

    # assemble a list with arrays containing the information for each unique event and its metrics
    cupti_runtimes_list = []
    for i in range(len(short_names)):
        cupti_runtimes_list.append([
        str(short_names[i]),
        str(visits_dict[short_names[i]]),
        str(runtime_dict[short_names[i]]),
        str(median_dict[short_names[i]])
        ])

    return cupti_runtimes_list, cupti_runtimes

def get_cudnn_events_list(path, start, end):
    """
    get_cudnn_events_list function to get a list of cudnn events and their metrics

    :param path: the path as string for the experiment database to be read
    :param start: the start time of the selected time interval in the selection as long
    :param end:  the end time of the selected time interval in the selection as long
    :return cudnn_events_list: the list of events and their metrics of the cudnn events
    """

    # open database in path and create connection
    db = sqlite3.connect(path)
    cursor = db.cursor()

    # load cupti runtime events from database
    query = "SELECT events.start, events.end, events.eventClass, events.globalTid, strings.value FROM CUDNN_EVENTS AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE events.start>="+str(start)+" AND events.end<="+str(end)+";"
    cudnn_events = get_data_from_db(cursor, query)
    cudnn_event_objects = convert_cudnn_events_to_objects(cudnn_events)

    # get a list of unique cupti kernel events in this time interval
    unique = []
    for i in range(len(cudnn_event_objects)):
        unique.append(cudnn_event_objects[i].name)
    short_names = get_unique_numbers(unique)

    # sum up the duration of each unique cupti kernel in seconds
    runtime_dict = {}
    for i in range(len(short_names)):
        runtime_dict[short_names[i]] = 0.0
    for i in range(len(cudnn_event_objects)):
        runtime_dict[cudnn_event_objects[i].name] += cudnn_event_objects[i].duration_seconds

    # sum up the number of visits of each unique cupti kernel
    visits_dict = {}
    for i in range(len(short_names)):
        visits_dict[short_names[i]] = 0
    for i in range(len(cudnn_event_objects)):
        visits_dict[cudnn_event_objects[i].name] += 1

    # compute the median runtime in seconds for each unique cupti kernel
    median_dict = {}
    for i in range(len(short_names)):
        median_dict[short_names[i]] = []
    for i in range(len(cudnn_event_objects)):
        median_dict[cudnn_event_objects[i].name].append(cudnn_event_objects[i].duration_seconds)
    for i in range(len(cudnn_event_objects)):
        median_dict[cudnn_event_objects[i].name] = np.median(median_dict[cudnn_event_objects[i].name])

    # assemble a list with arrays containing the information for each unique event and its metrics
    cudnn_event_objects_list = []
    for i in range(len(short_names)):
        cudnn_event_objects_list.append([
        str(short_names[i]),
        str(visits_dict[short_names[i]]),
        str(runtime_dict[short_names[i]]),
        str(median_dict[short_names[i]])
        ])

    return cudnn_event_objects_list, cudnn_event_objects

def get_cublas_events_list(path, start, end):
    """
    get_cublas_events_list function to get a list of cublas events and their metrics

    :param path: the path as string for the experiment database to be read
    :param start: the start time of the selected time interval in the selection as long
    :param end:  the end time of the selected time interval in the selection as long
    :return cublas_events_list: the list of events and their metrics of the cublas events
    """

    # open database in path and create connection
    db = sqlite3.connect(path)
    cursor = db.cursor()

    # load cupti runtime events from database
    query = "SELECT events.start, events.end, events.eventClass, events.globalTid, strings.value FROM CUBLAS_EVENTS AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE events.start>="+str(start)+" AND events.end<="+str(end)+";"
    cublas_events = get_data_from_db(cursor, query)
    cublas_event_objects = convert_cublas_events_to_objects(cublas_events)

    # get a list of unique cupti kernel events in this time interval
    unique = []
    for i in range(len(cublas_event_objects)):
        unique.append(cublas_event_objects[i].name)
    short_names = get_unique_numbers(unique)

    # sum up the duration of each unique cupti kernel in seconds
    runtime_dict = {}
    for i in range(len(short_names)):
        runtime_dict[short_names[i]] = 0.0
    for i in range(len(cublas_event_objects)):
        runtime_dict[cublas_event_objects[i].name] += cublas_event_objects[i].duration_seconds

    # sum up the number of visits of each unique cupti kernel
    visits_dict = {}
    for i in range(len(short_names)):
        visits_dict[short_names[i]] = 0
    for i in range(len(cublas_event_objects)):
        visits_dict[cublas_event_objects[i].name] += 1

    # compute the median runtime in seconds for each unique cupti kernel
    median_dict = {}
    for i in range(len(short_names)):
        median_dict[short_names[i]] = []
    for i in range(len(cublas_event_objects)):
        median_dict[cublas_event_objects[i].name].append(cublas_event_objects[i].duration_seconds)
    for i in range(len(cublas_event_objects)):
        median_dict[cublas_event_objects[i].name] = np.median(median_dict[cublas_event_objects[i].name])

    # assemble a list with arrays containing the information for each unique event and its metrics
    cublas_event_objects_list = []
    for i in range(len(short_names)):
        cublas_event_objects_list.append([
        str(short_names[i]),
        str(visits_dict[short_names[i]]),
        str(runtime_dict[short_names[i]]),
        str(median_dict[short_names[i]])
        ])

    return cublas_event_objects_list, cublas_event_objects

def choose_folder():
    """
    choose_folder function chooses a folder from the file system

    :return path: the path in which the experiment files lie
    """

    root = Tk()
    w = 800 # width for the Tk root
    h = 650 # height for the Tk root
    # get screen width and height
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen
    if ws > 1920:
        ws /= 2
    ws /= 2
    ws -= w/2
    x = ws
    y = (hs/2)-h/2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    path = filedialog.askdirectory(parent=root, initialdir="/home/marcus/Schreibtisch/experiments2")
    root.withdraw()
    root.destroy()
    if path == "":
        return None
    else:
        return path

def get_data_from_db(cursor, query):
    """
    get_data_from_db function to query data from a database

    :param query: the database query to be executed
    :return list: the list of elements that were found for the query in the database
    """
    try:
        x = list(cursor.execute(query))
        return x
    except Exception as e:
        return None

"""
This class identifies the training steps start and end using nvtx marks.
"""
def identify_training_steps(nvtx_marks):
    step_start_marker = []
    step_end_marker = []
    steps = []
    for i in range(len(nvtx_marks)):
        if nvtx_marks[i].category == 3:
            text = nvtx_marks[i].callpath_name
            text = text.split(",")
            nr = text[0]
            nr = nr.split(":")
            nr = nr[1]
            nr = int(nr)
            text = text[1]
            text = text.split(":")
            text = text[1]
            steps.append(nr)
            if text == "started":
                step_start_marker.append(nvtx_marks[i])
            else:
                step_end_marker.append(nvtx_marks[i])
    max = np.max(steps)
    counter1 = 0
    counter2 = 0
    training_steps = []
    for i in range(len(step_end_marker)):
        start = step_start_marker[i].time_stamp_long
        color = step_start_marker[i].color
        end = step_end_marker[i].time_stamp_long
        step = step_start_marker[i].callpath_name
        step = step.split(",")
        nr = step[0]
        nr = nr.split(":")
        nr = nr[1]
        nr = int(nr)+1
        if counter1 > max:
            counter2 += 1
            counter1 = 0
        counter1 += 1
        training_steps.append(TrainingStep(start, end, color, nr, counter2+1))
    return training_steps

"""
This class identifies the testing steps start and end using nvtx marks.
"""
def identify_testing_steps(nvtx_marks, training):
    step_start_marker = []
    step_end_marker = []
    steps = []
    for i in range(len(nvtx_marks)):
        if nvtx_marks[i].category == 5:
            text = nvtx_marks[i].callpath_name
            text = text.split(",")
            nr = text[0]
            nr = nr.split(":")
            nr = nr[1]
            nr = int(nr)
            text = text[1]
            text = text.split(":")
            text = text[1]
            steps.append(nr)
            if text == "started":
                step_start_marker.append(nvtx_marks[i])
            else:
                step_end_marker.append(nvtx_marks[i])
    max = np.max(steps)
    counter1 = 0
    counter2 = 0
    testing_steps = []
    for i in range(len(step_end_marker)):
        start = step_start_marker[i].time_stamp_long
        color = step_start_marker[i].color
        end = step_end_marker[i].time_stamp_long
        step = step_start_marker[i].callpath_name
        step = step.split(",")
        nr = step[0]
        nr = nr.split(":")
        nr = nr[1]
        nr = int(nr)+1
        if counter1 > max:
            counter2 += 1
            counter1 = 0
        counter1 += 1
        epoch = counter2+1
        if start >= training.end_time_long:
            final = True
            epoch = -1
        else:
            final = False
        testing_steps.append(TestingStep(start, end, color, nr, epoch, final))
    return testing_steps


"""
This functions identifies the start and end of the training epochs using nvtx marks.
"""
def identify_training_epochs(nvtx_marks):
    epochs = []
    temp1 = []
    temp2 = []
    for i in range(len(nvtx_marks)):
        if nvtx_marks[i].category == 2:
            text = nvtx_marks[i].callpath_name
            text = text.split(",")
            text = text[1]
            if text == "type:started":
                temp1.append(nvtx_marks[i])
            elif text == "type:ended":
                temp2.append(nvtx_marks[i])
    for i in range(len(temp1)):
        text = temp1[i].callpath_name
        text = text.split(",")
        text = text[0]
        text = text.split(":")
        text = text[1]
        text = int(text)
        epoch_nr = text + 1
        color = temp1[i].color
        start = temp1[i].time_stamp_long
        end = temp2[i].time_stamp_long
        epochs.append(Epoch(start, end, color, epoch_nr))
    return epochs

"""
This functions identifies the start and end of the testing process using the nvtx marks.
"""
def identify_testing_process(nvtx_marks, training):
    test_start_marker = []
    test_end_marker = []
    tests = []
    for i in range(len(nvtx_marks)):
        if nvtx_marks[i].category == 4:
            text = nvtx_marks[i].callpath_name
            if text == "test started":
                test_start_marker.append(nvtx_marks[i])
            if text == "test ended":
                test_end_marker.append(nvtx_marks[i])
    for i in range(len(test_end_marker)):
        start = test_start_marker[i].time_stamp_long
        color = test_start_marker[i].color
        end = test_end_marker[i].time_stamp_long
        epoch = i+1
        if start >= training.end_time_long:
            final = True
            epoch = -1
        else:
            final = False
        tests.append(Testing(start, end, color, epoch, final))
    return tests

def convert_memset_events_to_objects(list):
    """
    convert_memset_events_to_objects function converts the database memset events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return memset_events: a list of CuptiMemset objects
    """

    cupti_memsets = []
    for i in range(len(list)):
        cupti_memsets.append(CuptiMemset(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7], list[i][8], list[i][9], list[i][10], list[i][11], list[i][12], list[i][13], list[i][14]))
    return cupti_memsets

def convert_cublas_events_to_objects(list):
    """
    convert_cublas_events_to_objects function converts the database cublas events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return memset_events: a list of CublasEvent objects
    """

    cublas_events = []
    for i in range(len(list)):
        cublas_events.append(CublasEvent(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4],
        list[i][5], list[i][6], list[i][7], list[i][8]))
    return cublas_events

def convert_os_events_to_objects(list):
    """
    convert_os_events_to_objects function converts the database os events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return memset_events: a list of OSEvent objects
    """

    os_events = []
    for i in range(len(list)):
        os_events.append(OSEvent(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4],
        list[i][5], list[i][6], list[i][7], list[i][8], list[i][9], list[i][10], list[i][11], list[i][12]))
    return os_events

def convert_memcopy_events_to_objects(list):
    """
    convert_memcopy_events_to_objects function converts the database memcopy events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return memset_events: a list of CuptiMemcopy objects
    """

    cupti_memcopy = []
    for i in range(len(list)):
        cupti_memcopy.append(CuptiMemcopy(list[i][0], list[i][1], list[i][18], list[i][19], list[i][20],
        list[i][21], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7],
        list[i][8], list[i][9], list[i][10], list[i][11], list[i][12], list[i][13], list[i][14], list[i][15],
        list[i][16], list[i][17]))
    return cupti_memcopy

def convert_cupti_kernels_to_objects(list):
    """
    convert_cupti_kernels_to_objects function converts the database cupti kernels in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return cupti_kernels: a list of CuptiKernel objects
    """

    cupti_kernels = []
    for i in range(len(list)):
        cupti_kernels.append(CuptiKernel(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7], list[i][8], list[i][9], list[i][10], list[i][11], list[i][12], list[i][13], list[i][14], list[i][15]))
    return cupti_kernels

def convert_mpi_events_to_objects(list):
    """
    convert_mpi_events_to_objects function converts the database mpi events from nvtx events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return mpi_events: a list of MPI Event objects
    """

    mpi_events = []
    for i in range(len(list)):
        mpi_events.append(MPIEvent(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6]))
    return mpi_events

def convert_horovod_events_to_objects(list):
    """
    convert_horovod_events_to_objects function converts the database mpi events from nvtx events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return horovod_events: a list of Horovod Event objects
    """

    horovod_events = []
    for i in range(len(list)):
        horovod_events.append(HorovodEvent(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7]))
    return horovod_events

def convert_cupti_runtimes_to_objects(list):
    """
    convert_cupti_runtimes_to_objects function converts the database cupti runtimes in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return cupti_runtimes: a list of CuptiRuntime objects
    """

    cupti_runtimes = []
    for i in range(len(list)):
        cupti_runtimes.append(CuptiRuntime(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7],
        list[i][8], list[i][9], list[i][10], list[i][11]))
    return cupti_runtimes

def convert_cupti_kernel_launch_to_objects(list):
    """
    convert_cupti_kernel_launch_to_objects function converts the database cupti kernel launch objects in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return cupti_kernel_launches: a list of CuptiKernelLaunch objects
    """

    cupti_kernel_launches = []
    for i in range(len(list)):
        cupti_kernel_launches.append(CuptiKernelLaunch(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7],
        list[i][8], list[i][9], list[i][10], list[i][11]))
    return cupti_kernel_launches

def convert_cudnn_events_to_objects(list):
    """
    convert_cudnn_events_to_objects function converts the database cudnn events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return cudnn_events: a list of CudnnEvents objects
    """

    cudnn_events = []
    for i in range(len(list)):
        cudnn_events.append(CudnnEvent(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5],
        list[i][6], list[i][7], list[i][8]))
    return cudnn_events

def convert_cupti_synchronize_to_objects(list):
    """
    convert_cupti_synchronize_to_objects function converts the database cupti synchronization events in the list into Objects to handle them easier.

    :param list: the list containing all elements retrieved from the database
    :return cupti_synchronize_events: a list of CuptiSynchronization objects
    """

    cupti_synchronize_events = []
    for i in range(len(list)):
        cupti_synchronize_events.append(CuptiSynchronization(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5],
        list[i][6], list[i][7], list[i][8], list[i][9], list[i][10], list[i][11], list[i][12]))
    return cupti_synchronize_events

"""
Converts the database nvtx elements in the list into Objects to handle them easier.
"""
def convert_nvtx_events_to_objects(list):
    nvtx_events = []
    for i in range(len(list)):
        if len(list[0]) == 11:
            nvtx_events.append(NVTX_Event(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7], list[i][8], list[i][9], list[i][10]))
        else:
            nvtx_events.append(NVTX_Event(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7], list[i][8], list[i][9]))
    return nvtx_events

"""
This functions identifies the start and end of the training process using the nvtx marks.
"""
def identify_training_process(nvtx_marks):
    training_start_marker = None
    training_end_marker = None
    for i in range(len(nvtx_marks)):
        if nvtx_marks[i].category == 1:
            text = nvtx_marks[i].callpath_name
            if text == "training started":
                training_start_marker = nvtx_marks[i]
            if text == "training ended":
                training_end_marker = nvtx_marks[i]

    if training_start_marker == None or training_end_marker == None:
        return False, None
    else:
        training = Training(training_start_marker.time_stamp_long, training_end_marker.time_stamp_long, training_start_marker.color)
        return True, training

"""
Converts the database nvtx markers in the list into Objects to handle them easier.
"""
def convert_nvtx_marks_to_objects(list):
    nvtx_markers = []
    for i in range(len(list)):
        if len(list[0]) == 10:
            nvtx_markers.append(NVTX_Mark(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7], list[i][8], list[i][9]))
        else:
            nvtx_markers.append(NVTX_Mark(list[i][0], list[i][1], list[i][2], list[i][3], list[i][4], list[i][5], list[i][6], list[i][7], list[i][8]))
    return nvtx_markers

"""
Function to read the performance experiment configuration of
the current file using the file name.
"""
def read_experiment_configurations(id, filename):
    string_parts = filename.split(".")
    app_name = string_parts[0]
    string_parts.pop(0)
    string_parts.pop(len(string_parts)-1)
    mpi_process_nr = -1
    for j in range(len(string_parts)):
        if string_parts[j].find("mpi") != -1:
            mpi_process_nr = string_parts[j]
            mpi_process_nr = mpi_process_nr.replace("mpi", "")
            string_parts.pop(j)
            break
    repetition_nr = -1
    for j in range(len(string_parts)):
        if string_parts[j].find("r") != -1:
            repetition_nr = string_parts[j]
            repetition_nr = repetition_nr.replace("r", "")
            string_parts.pop(j)
            break
    parameter_names = []
    parameter_values = []
    for j in range(len(string_parts)):
        strings = string_parts[j]
        parameter_values.append(int(re.findall(r"\d+", string_parts[j])[0]))
        parameter_names.append(re.findall("[a-zA-Z]+", string_parts[j])[0])
    return ExperimentConfig(id, app_name, repetition_nr, mpi_process_nr, parameter_names, parameter_values, filename)

def get_events_in_interval(start, end, events):
    """
    get_events_in_interval function retrieves the nvtx events in a given interval from a given set of events

    :param start: the start time as long of the interval
    :param end: the end time as long of the interval
    :param events: the list of events to search in
    :return events_in_interval: the list of events in the interval
    """

    events_in_interval = []
    for i in range(len(events)):
        if events[i].start_time_long >= start and events[i].end_time_long <= end:
            events_in_interval.append(events[i])
        elif events[i].start_time_long <= start and events[i].end_time_long >= end:
            new_event = NVTX_Event(start, end, events[i].callpath_name, None, events[i].text_id, events[i].domain_id, events[i].event_type, events[i].range_id, events[i].category, events[i].global_tid)
            events_in_interval.append(new_event)

    return events_in_interval

def get_marks_in_interval(start, end, marks):
    """
    get_marks_in_interval function retrieves the nvtx marks in a given interval from a given set of marks

    :param start: the start time as long of the interval
    :param end: the end time as long of the interval
    :param marks: the list of marks to search in
    :return marks_in_interval: the list of marks in the interval
    """

    marks_in_interval = []
    for i in range(len(marks)):
        if marks[i].time_stamp_long > start and marks[i].time_stamp_long < end:
            marks_in_interval.append(marks[i])

    return marks_in_interval
