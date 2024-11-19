from extradeep.entities.parameter import Parameter
from extradeep.entities.experiment import Experiment

from extradeep.util.progress_bar import ProgressBar
from extradeep.fileio.sqlite_helper import read_experiment_configurations
from extradeep.util.util_functions import get_sqlite_files_in_path

import sqlite3
import numpy as np
import logging
import sys

from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

from extradeep.fileio.nsight_systems_file_reader import find_unique_experiment_configurations
from extradeep.fileio.nsight_systems_file_reader import create_storage_objects_for_exp_configs
from extradeep.fileio.nsight_systems_file_reader import find_reps_for_unique_exps
from extradeep.fileio.nsight_systems_file_reader import remove_outliers
from extradeep.fileio.nsight_systems_file_reader import get_experiment_config_ids
from extradeep.fileio.nsight_systems_file_reader import create_measurement_list

from extradeep.fileio.nsight_systems_file_reader import read_app_phases, add_app_phases_to_experiment

from extradeep.fileio.sqlite_helper import get_data_from_db
from extradeep.entities.system.system_info import SystemInfo


def parallel_read_app_phase(inputs):
    """
    parallel_read_app_phase function...
    """

    # unpack input params from shared dict
    count = inputs[0]
    path = inputs[1]
    file = inputs[2]

    # get the performance experiment configuration of this file
    config = read_experiment_configurations(count, file)

    # open database in path and create connection
    db = sqlite3.connect(path)
    cursor = db.cursor()

    # load the nvtx epoch events for epoch analysis
    communication_time, memory_time, computation_time, communication_time_epochs, memory_time_epochs, computation_time_epochs = read_app_phases(cursor)

    com_time = None
    comp_time = None
    mem_time = None
    com_time_epochs = None
    comp_time_epochs = None
    mem_time_epochs = None

    # if epochs exist
    if communication_time != None and memory_time != None and computation_time != None:
        com_time = communication_time
        comp_time = computation_time
        mem_time = memory_time
        com_time_epochs = communication_time_epochs
        comp_time_epochs = computation_time_epochs
        mem_time_epochs = memory_time_epochs

    data = (config, com_time, comp_time, mem_time, com_time_epochs, comp_time_epochs, mem_time_epochs)

    return data


def read_application_phases(folder, arguments):

    paths, files = get_sqlite_files_in_path(folder)

    #system info
    system_info = None

    # read system info from first sqlite file
    for i in range(len(paths)):
        db = sqlite3.connect(paths[i])
        cursor = db.cursor()
        query = "SELECT value AS cpu_cores FROM TARGET_INFO_SYSTEM_ENV WHERE name LIKE \"CpuCores\";"
        result = get_data_from_db(cursor, query)
        if result != None:
            if len(result) != 0:
                cpu_cores = float(result[0][0])
                system_info = SystemInfo(cpu_cores)
                break

    if len(paths) != 0:

        max_processes = multiprocessing.cpu_count()

        # variable to save all seen experiment configurations
        configs = []

        communication_runtimes = {}
        computation_runtimes = {}
        memory_runtimes = {}
        communication_runtimes_epochs = {}
        computation_runtimes_epochs = {}
        memory_runtimes_epochs = {}

        # create input array for parallel testing step i/o
        inputs = []
        for i in range(len(paths)):
            inputs.append([i, paths[i], files[i]])

        # create process pool for parallel testing step i/o
        with Pool(processes=max_processes) as pool:
            parallel_result = list(tqdm(pool.imap(parallel_read_app_phase, inputs), total=len(paths), desc="Reading Nsight Systems .sqlite files"))

        # construct the measurement dict from the results of the different processes
        for i in range(len(parallel_result)):
            config = parallel_result[i][0]
            com_time = parallel_result[i][1]
            comp_time = parallel_result[i][2]
            mem_time = parallel_result[i][3]
            com_time_epochs = parallel_result[i][4]
            comp_time_epochs = parallel_result[i][5]
            mem_time_epochs = parallel_result[i][6]
            if com_time != None:
                communication_runtimes[config.id] = com_time
            if comp_time != None:
                computation_runtimes[config.id] = comp_time
            if mem_time != None:
                memory_runtimes[config.id] = mem_time
            if com_time_epochs != None:
                communication_runtimes_epochs[config.id] = com_time_epochs
            if comp_time_epochs != None:
                computation_runtimes_epochs[config.id] = comp_time_epochs
            if mem_time_epochs != None:
                memory_runtimes_epochs[config.id] = mem_time_epochs
            configs.append(config)

        # data processing
        with ProgressBar(desc='Data preprocessing') as pbar:

            pbar.total += 2

            # do some data preprocessing
            pbar.step("Data preprocessing")
            pbar.update(1)
            
            # find unique experiment configurations
            nr_experiment_configs, unique_configs = find_unique_experiment_configurations(configs)

            # create a storage object for each unique experiment config
            storages = create_storage_objects_for_exp_configs(unique_configs)

            # find the repetitions for each unique experiment config
            find_reps_for_unique_exps(storages, configs)

            # for each unique experiment config get a list containing its repetitions and different mpi ranks to average the metrics over
            experiment_config_ids = get_experiment_config_ids(storages, configs)

            measurement_communication_runtime = None
            measurement_computation_runtime = None
            measurement_memory_runtime = None
            measurement_communication_runtime_epochs = None
            measurement_computation_runtime_epochs = None
            measurement_memory_runtime_epochs = None

            # if data for testing step analysis exists do the preprocessing for it
            if bool(communication_runtimes) != False and bool(computation_runtimes) != False and bool(memory_runtimes) != False:

                measurement_communication_runtime = create_measurement_list(experiment_config_ids, communication_runtimes)
                measurement_computation_runtime = create_measurement_list(experiment_config_ids, computation_runtimes)
                measurement_memory_runtime = create_measurement_list(experiment_config_ids, memory_runtimes)
                measurement_communication_runtime_epochs = create_measurement_list(experiment_config_ids, communication_runtimes_epochs)
                measurement_computation_runtime_epochs = create_measurement_list(experiment_config_ids, computation_runtimes_epochs)
                measurement_memory_runtime_epochs = create_measurement_list(experiment_config_ids, memory_runtimes_epochs)

            # if measurements exits create experiment
            if measurement_communication_runtime != None and measurement_computation_runtime != None and measurement_memory_runtime != None:

                # create an experiment
                pbar.step("Create experiment")
                pbar.update(1)

                # create new empty extrap experiment
                extrap_experiment = Experiment()

                # add scaling type to experiment
                if arguments.strong_scaling == True:
                    extrap_experiment.scaling = "strong"
                else:
                    extrap_experiment.scaling = "weak"

                # create and add parameters to experiment
                for x in unique_configs[0].parameter_names:
                    extrap_parameter = Parameter(x)
                    extrap_experiment.add_parameter(extrap_parameter)

                # add data for testing step analysis to experiment
                extrap_experiment = add_app_phases_to_experiment(extrap_experiment, pbar, unique_configs, 
                measurement_communication_runtime, measurement_computation_runtime, measurement_memory_runtime, 
                measurement_communication_runtime_epochs, measurement_computation_runtime_epochs, measurement_memory_runtime_epochs,
                arguments, system_info)

                if extrap_experiment != None:
                    return True, extrap_experiment
                else:
                    return False, None

            else:
                return False, None

    else:
        logging.error("No .sqlite files found in provided folder.")
        sys.exit(1)
