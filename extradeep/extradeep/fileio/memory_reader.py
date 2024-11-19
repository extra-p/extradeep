from extradeep.entities.parameter import Parameter
from extradeep.entities.experiment import Experiment

from extradeep.util.progress_bar import ProgressBar
from extradeep.fileio.io_helper import *
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
from extradeep.fileio.nsight_systems_file_reader import get_experiment_config_ids
from extradeep.fileio.nsight_systems_file_reader import create_measurement_list

from extradeep.fileio.nsight_systems_file_reader import read_memory_events, add_memory_events_to_experiment

from extradeep.fileio.sqlite_helper import get_data_from_db
from extradeep.entities.system.system_info import SystemInfo


def parallel_read_memory_events(inputs):
    """
    parallel_read_memory_events function...
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

    # memory events
    memset_event_visits = {}
    memset_event_runtime = {}
    memset_event_bytes = {}
    memcopy_event_visits = {}
    memcopy_event_runtime = {}
    memcopy_event_bytes = {}
    memcopy_htod_event_visits = {}
    memcopy_dtoh_event_visits = {}
    memcopy_dtod_event_visits = {}
    memcopy_htoh_event_visits = {}
    memcopy_htod_event_runtime = {}
    memcopy_dtoh_event_runtime = {}
    memcopy_dtod_event_runtime = {}
    memcopy_htoh_event_runtime = {}
    memcopy_htod_event_bytes = {}
    memcopy_dtoh_event_bytes = {} 
    memcopy_dtod_event_bytes = {}
    memcopy_htoh_event_bytes = {}

    # load the memory events data
    memset_visits, memset_runtime, memcopy_visits, memcopy_runtime, memset_bytes, memcopy_bytes, memcopy_htod_visits, memcopy_dtoh_visits, memcopy_dtod_visits, memcopy_htoh_visits, memcopy_htod_runtime, memcopy_dtoh_runtime, memcopy_dtod_runtime, memcopy_htoh_runtime, memcopy_htod_bytes, memcopy_dtoh_bytes, memcopy_dtod_bytes, memcopy_htoh_bytes = read_memory_events(cursor)

    if memset_visits != None and memset_runtime != None and memcopy_visits != None and memcopy_runtime != None:
            
        memset_event_visits = memset_visits
        memset_event_runtime = memset_runtime
        memset_event_bytes = memset_bytes
        memcopy_event_visits = memcopy_visits
        memcopy_event_runtime = memcopy_runtime
        memcopy_event_bytes = memcopy_bytes
        memcopy_htod_event_visits = memcopy_htod_visits
        memcopy_dtoh_event_visits = memcopy_dtoh_visits
        memcopy_dtod_event_visits = memcopy_dtod_visits
        memcopy_htoh_event_visits = memcopy_htoh_visits
        memcopy_htod_event_runtime = memcopy_htod_runtime
        memcopy_dtoh_event_runtime = memcopy_dtoh_runtime
        memcopy_dtod_event_runtime = memcopy_dtod_runtime
        memcopy_htoh_event_runtime = memcopy_htoh_runtime
        memcopy_htod_event_bytes = memcopy_htod_bytes
        memcopy_dtoh_event_bytes = memcopy_dtoh_bytes
        memcopy_dtod_event_bytes = memcopy_dtod_bytes
        memcopy_htoh_event_bytes = memcopy_htoh_bytes

        # remove outliers, should remove first epoch with 
        # high initialization and optimization overhead and others
        #runtimes = remove_outliers(runtimes)

    data = (
        config, memset_event_visits, memset_event_runtime, memset_event_bytes, memcopy_event_visits, memcopy_event_runtime,
    memcopy_event_bytes, memcopy_htod_event_visits, memcopy_dtoh_event_visits, memcopy_dtod_event_visits, memcopy_htoh_event_visits,
    memcopy_htod_event_runtime, memcopy_dtoh_event_runtime, memcopy_dtod_event_runtime, memcopy_htoh_event_runtime, memcopy_htod_event_bytes,
    memcopy_dtoh_event_bytes, memcopy_dtod_event_bytes, memcopy_htoh_event_bytes
    )

    return data


def read_memory(dir_name, arguments):

    paths, files = get_sqlite_files_in_path(dir_name)

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

        # memory events
        memset_event_visits = {}
        memset_event_runtime = {}
        memset_event_bytes = {}
        memcopy_event_visits = {}
        memcopy_event_runtime = {}
        memcopy_event_bytes = {}
        memcopy_htod_event_visits = {}
        memcopy_dtoh_event_visits = {}
        memcopy_dtod_event_visits = {}
        memcopy_htoh_event_visits = {}
        memcopy_htod_event_runtime = {}
        memcopy_dtoh_event_runtime = {}
        memcopy_dtod_event_runtime = {}
        memcopy_htoh_event_runtime = {}
        memcopy_htod_event_bytes = {}
        memcopy_dtoh_event_bytes = {} 
        memcopy_dtod_event_bytes = {}
        memcopy_htoh_event_bytes = {}

        # create input array for parallel epoch i/o
        inputs = []
        for i in range(len(paths)):
            inputs.append([i, paths[i], files[i]])

        # create process pool for parallel epoch i/o
        with Pool(processes=max_processes) as pool:
            parallel_result = list(tqdm(pool.imap(parallel_read_memory_events, inputs), total=len(paths), desc="Reading Nsight Systems .sqlite files"))

        # construct the measurement dict from the results of the different processes
        for i in range(len(parallel_result)):
            
            config = parallel_result[i][0]
            memset_visits = parallel_result[i][1]
            memset_runtime = parallel_result[i][2]
            memset_bytes = parallel_result[i][3]
            memcopy_visits = parallel_result[i][4]
            memcopy_runtime = parallel_result[i][5]
            memcopy_bytes = parallel_result[i][6]
            memcopy_htod_visits = parallel_result[i][7]
            memcopy_dtoh_visits = parallel_result[i][8]
            memcopy_dtod_visits = parallel_result[i][9]
            memcopy_htoh_visits = parallel_result[i][10]
            memcopy_htod_runtime = parallel_result[i][11]
            memcopy_dtoh_runtime = parallel_result[i][12]
            memcopy_dtod_runtime = parallel_result[i][13]
            memcopy_htoh_runtime = parallel_result[i][14]
            memcopy_htod_bytes = parallel_result[i][15]
            memcopy_dtoh_bytes = parallel_result[i][16]
            memcopy_dtod_bytes = parallel_result[i][17]
            memcopy_htoh_bytes = parallel_result[i][18]

            if memset_visits != None:
                memset_event_visits[config.id] = memset_visits
            if memset_runtime != None:
                memset_event_runtime[config.id] = memset_runtime
            if memset_bytes != None:
                memset_event_bytes[config.id] = memset_bytes
            if memcopy_visits != None:
                memcopy_event_visits[config.id] = memcopy_visits
            if memcopy_runtime != None:
                memcopy_event_runtime[config.id] = memcopy_runtime
            if memcopy_bytes != None:
                memcopy_event_bytes[config.id] = memcopy_bytes
            if memcopy_htod_visits != None:
                memcopy_htod_event_visits[config.id] = memcopy_htod_visits
            if memcopy_dtoh_visits != None:
                memcopy_dtoh_event_visits[config.id] = memcopy_dtoh_visits
            if memcopy_dtod_visits != None:
                memcopy_dtod_event_visits[config.id] = memcopy_dtod_visits
            if memcopy_htoh_visits != None:
                memcopy_htoh_event_visits[config.id] = memcopy_htoh_visits
            if memcopy_htod_runtime != None:
                memcopy_htod_event_runtime[config.id] = memcopy_htod_runtime
            if memcopy_dtoh_runtime != None:
                memcopy_dtoh_event_runtime[config.id] = memcopy_dtoh_runtime
            if memcopy_dtod_runtime != None:
                memcopy_dtod_event_runtime[config.id] = memcopy_dtod_runtime
            if memcopy_htoh_runtime != None:
                memcopy_htoh_event_runtime[config.id] = memcopy_htoh_runtime
            if memcopy_htod_bytes != None:
                memcopy_htod_event_bytes[config.id] = memcopy_htod_bytes
            if memcopy_dtoh_bytes != None:
                memcopy_dtoh_event_bytes[config.id] = memcopy_dtoh_bytes
            if memcopy_dtod_bytes != None:
                memcopy_dtod_event_bytes[config.id] = memcopy_dtod_bytes
            if memcopy_htoh_bytes != None:
                memcopy_htoh_event_bytes[config.id] = memcopy_htoh_bytes
            
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

            measurement_memset_runtime = None
            measurement_memset_visits = None
            measurement_memcopy_runtime = None
            measurement_memcopy_visits = None
            measurement_memset_bytes = None
            measurement_memcopy_bytes = None

            measurement_memcopy_htod_visits = None
            measurement_memcopy_dtoh_visits = None
            measurement_memcopy_dtod_visits = None
            measurement_memcopy_htoh_visits = None

            measurement_memcopy_htod_runtime = None
            measurement_memcopy_dtoh_runtime = None
            measurement_memcopy_dtod_runtime = None
            measurement_memcopy_htoh_runtime = None

            measurement_memcopy_htod_bytes = None
            measurement_memcopy_dtoh_bytes = None
            measurement_memcopy_dtod_bytes = None
            measurement_memcopy_htoh_bytes = None

            # if data for memory operations analysis exists do the preprocessing for it
            if bool(memset_event_visits) != False and bool(memcopy_event_visits) != False:

                measurement_memset_runtime = create_measurement_list(experiment_config_ids, memset_event_runtime)
                measurement_memset_visits = create_measurement_list(experiment_config_ids, memset_event_visits)
                measurement_memcopy_runtime = create_measurement_list(experiment_config_ids, memcopy_event_runtime)
                measurement_memcopy_visits = create_measurement_list(experiment_config_ids, memcopy_event_visits)
                measurement_memset_bytes = create_measurement_list(experiment_config_ids, memset_event_bytes)
                measurement_memcopy_bytes = create_measurement_list(experiment_config_ids, memcopy_event_bytes)

                measurement_memcopy_htod_visits = create_measurement_list(experiment_config_ids, memcopy_htod_event_visits)
                measurement_memcopy_dtoh_visits = create_measurement_list(experiment_config_ids, memcopy_dtoh_event_visits)
                measurement_memcopy_dtod_visits = create_measurement_list(experiment_config_ids, memcopy_dtod_event_visits)
                measurement_memcopy_htoh_visits = create_measurement_list(experiment_config_ids, memcopy_htoh_event_visits)

                measurement_memcopy_htod_runtime = create_measurement_list(experiment_config_ids, memcopy_htod_event_runtime)
                measurement_memcopy_dtoh_runtime = create_measurement_list(experiment_config_ids, memcopy_dtoh_event_runtime)
                measurement_memcopy_dtod_runtime = create_measurement_list(experiment_config_ids, memcopy_dtod_event_runtime)
                measurement_memcopy_htoh_runtime = create_measurement_list(experiment_config_ids, memcopy_htoh_event_runtime)

                measurement_memcopy_htod_bytes = create_measurement_list(experiment_config_ids, memcopy_htod_event_bytes)
                measurement_memcopy_dtoh_bytes = create_measurement_list(experiment_config_ids, memcopy_dtoh_event_bytes)
                measurement_memcopy_dtod_bytes = create_measurement_list(experiment_config_ids, memcopy_dtod_event_bytes)
                measurement_memcopy_htoh_bytes = create_measurement_list(experiment_config_ids, memcopy_htoh_event_bytes)

            if measurement_memset_visits != None and measurement_memcopy_visits != None:

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

                # add data for memory events to experiment
                extrap_experiment = add_memory_events_to_experiment(extrap_experiment, measurement_memset_runtime, measurement_memset_visits, measurement_memcopy_runtime, measurement_memcopy_visits, measurement_memset_bytes, measurement_memcopy_bytes, measurement_memcopy_htod_visits, measurement_memcopy_dtoh_visits, measurement_memcopy_dtod_visits, measurement_memcopy_htoh_visits, measurement_memcopy_htod_runtime, measurement_memcopy_dtoh_runtime, measurement_memcopy_dtod_runtime, measurement_memcopy_htoh_runtime, measurement_memcopy_htod_bytes, measurement_memcopy_dtoh_bytes, measurement_memcopy_dtod_bytes, measurement_memcopy_htoh_bytes, pbar, unique_configs, arguments, system_info)

                if extrap_experiment != None:
                    return True, extrap_experiment
                else:
                    return False, None

            else:
                return False, None
    
    else:
        logging.error("No .sqlite files found in provided folder.")
        sys.exit(1)
