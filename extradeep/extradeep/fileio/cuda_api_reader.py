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
from extradeep.fileio.nsight_systems_file_reader import concat_unique_kernel_list
from extradeep.fileio.nsight_systems_file_reader import get_experiment_config_ids
from extradeep.fileio.nsight_systems_file_reader import create_metric_dict, create_measurement_list

from extradeep.fileio.nsight_systems_file_reader import read_cuda_api_events, populate_measurement_callpath, add_cuda_api_events_to_experiment

from extradeep.fileio.sqlite_helper import get_data_from_db
from extradeep.entities.system.system_info import SystemInfo


def parallel_read_cuda_api_events(inputs):
    """
    parallel_read_os_events function...
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

    # CUDA API events
    cuda_api_event_visits = {}
    cuda_api_event_runtime = {}
    callpaths_cuda_api = []
    callpaths_cuda_api_kernel_launches = []
    cuda_api_kernel_launch_visits = {}
    cuda_api_kernel_launch_runtime = {}
    cuda_api_kernel_launch_visits_sum = {}
    cuda_api_kernel_launch_runtime_sum = {}

    # load the cuda_api events data
    cuda_api_callpaths, cuda_api_visits, cuda_api_runtime, unique_cuda_kernel_launches, cuda_api_kernel_launches_runtime, cuda_api_kernel_launches_visits, cuda_api_kernel_launches_visits_sum, cuda_api_kernel_launches_runtime_sum = read_cuda_api_events(cursor)

    if cuda_api_callpaths != None and cuda_api_visits != None and cuda_api_runtime != None and unique_cuda_kernel_launches != None and cuda_api_kernel_launches_runtime != None and cuda_api_kernel_launches_visits != None and cuda_api_kernel_launches_visits_sum != None and cuda_api_kernel_launches_runtime_sum != None:
            
        cuda_api_event_visits = cuda_api_visits
        cuda_api_event_runtime = cuda_api_runtime
        callpaths_cuda_api = cuda_api_callpaths
        callpaths_cuda_api_kernel_launches = unique_cuda_kernel_launches
        cuda_api_kernel_launch_visits = cuda_api_kernel_launches_visits
        cuda_api_kernel_launch_runtime = cuda_api_kernel_launches_runtime
        cuda_api_kernel_launch_visits_sum = cuda_api_kernel_launches_visits_sum
        cuda_api_kernel_launch_runtime_sum = cuda_api_kernel_launches_runtime_sum

        # remove outliers, should remove first epoch with 
        # high initialization and optimization overhead and others
        #runtimes = remove_outliers(runtimes)

    data = (config, cuda_api_event_visits, cuda_api_event_runtime, callpaths_cuda_api, callpaths_cuda_api_kernel_launches, cuda_api_kernel_launch_visits, cuda_api_kernel_launch_runtime, cuda_api_kernel_launch_visits_sum, cuda_api_kernel_launch_runtime_sum)

    return data


def read_cuda_api(dir_name, arguments):
    
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

        # CUDA API events
        cuda_api_event_visits = {}
        cuda_api_event_runtime = {}
        callpaths_cuda_api = []
        callpaths_cuda_api_kernel_launches = []
        cuda_api_kernel_launch_visits = {}
        cuda_api_kernel_launch_runtime = {}
        cuda_api_kernel_launch_visits_sum = {}
        cuda_api_kernel_launch_runtime_sum = {}

        # create input array for parallel epoch i/o
        inputs = []
        for i in range(len(paths)):
            inputs.append([i, paths[i], files[i]])

        # create process pool for parallel epoch i/o
        with Pool(processes=max_processes) as pool:
            parallel_result = list(tqdm(pool.imap(parallel_read_cuda_api_events, inputs), total=len(paths), desc="Reading Nsight Systems .sqlite files"))

        # construct the measurement dict from the results of the different processes
        for i in range(len(parallel_result)):

            config = parallel_result[i][0]
            cuda_api_visits = parallel_result[i][1]
            cuda_api_runtime = parallel_result[i][2]
            callpaths_api = parallel_result[i][3]
            callpaths_api_kernel_launches = parallel_result[i][4]
            api_kernel_launch_visits = parallel_result[i][5]
            api_kernel_launch_runtime = parallel_result[i][6]
            api_kernel_launch_visits_sum = parallel_result[i][7]
            api_kernel_launch_runtime_sum = parallel_result[i][8]

            if cuda_api_visits != None:
                cuda_api_event_visits[config.id] = cuda_api_visits
            if cuda_api_runtime != None:
                cuda_api_event_runtime[config.id] = cuda_api_runtime
            if callpaths_api != None:
                callpaths_cuda_api.append(callpaths_api)
            if callpaths_api_kernel_launches != None:
                callpaths_cuda_api_kernel_launches.append(callpaths_api_kernel_launches)
            if api_kernel_launch_visits != None:
                cuda_api_kernel_launch_visits[config.id] = api_kernel_launch_visits
            if api_kernel_launch_runtime != None:
                cuda_api_kernel_launch_runtime[config.id] = api_kernel_launch_runtime
            if api_kernel_launch_visits_sum != None:
                cuda_api_kernel_launch_visits_sum[config.id] = api_kernel_launch_visits_sum
            if api_kernel_launch_runtime_sum != None:
                cuda_api_kernel_launch_runtime_sum[config.id] = api_kernel_launch_runtime_sum
            
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

            callpath_list_cuda_api_events = None
            measurement_visits_cuda_api_events = None
            measurement_runtime_cuda_api_events = None
            measurement_visits_cuda_api_events = None
            measurement_runtime_cuda_api_events = None

            # if data for cuda api analysis exists do the preprocessing for it
            if bool(cuda_api_event_visits) != False and len(callpaths_cuda_api) != 0:
                
                concat_kernel_list_cuda_api_kernel_launches = concat_unique_kernel_list(callpaths_cuda_api_kernel_launches)
                measurement_visits_cuda_api_kernel_launches = create_metric_dict(experiment_config_ids, concat_kernel_list_cuda_api_kernel_launches)
                measurement_runtime_cuda_api_kernel_launches = create_metric_dict(experiment_config_ids, concat_kernel_list_cuda_api_kernel_launches)
                callpath_list_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches, measurement_runtime_cuda_api_kernel_launches = populate_measurement_callpath(concat_kernel_list_cuda_api_kernel_launches, experiment_config_ids, cuda_api_kernel_launch_visits, cuda_api_kernel_launch_runtime, measurement_visits_cuda_api_kernel_launches, measurement_runtime_cuda_api_kernel_launches)
                measurement_runtime_cuda_api_kernel_launches_sum = create_measurement_list(experiment_config_ids, cuda_api_kernel_launch_runtime_sum)
                measurement_visits_cuda_api_kernel_launches_sum = create_measurement_list(experiment_config_ids, cuda_api_kernel_launch_visits_sum)

                # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
                concat_kernel_list_cuda_api_events = concat_unique_kernel_list(callpaths_cuda_api)
            
                # create new free metric dict for visits
                measurement_visits_cuda_api_events = create_metric_dict(experiment_config_ids, concat_kernel_list_cuda_api_events)
                
                # create new free metric dict for runtime sum
                measurement_runtime_cuda_api_events = create_metric_dict(experiment_config_ids, concat_kernel_list_cuda_api_events)

                # populate the metric dicts and get the callpaths
                callpath_list_cuda_api_events, measurement_visits_cuda_api_events, measurement_runtime_cuda_api_events = populate_measurement_callpath(concat_kernel_list_cuda_api_events, experiment_config_ids, cuda_api_event_visits, cuda_api_event_runtime, measurement_visits_cuda_api_events, measurement_runtime_cuda_api_events)

            # if measurements exits create experiment
            if callpath_list_cuda_api_events != None or measurement_visits_cuda_api_events != None or measurement_runtime_cuda_api_events != None or measurement_runtime_cuda_api_events != None or measurement_visits_cuda_api_events != None:

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

                # add data for cuda api events to experiment
                extrap_experiment = add_cuda_api_events_to_experiment(extrap_experiment, callpath_list_cuda_api_events, measurement_visits_cuda_api_events, measurement_runtime_cuda_api_events, pbar, unique_configs, callpath_list_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches, measurement_runtime_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches_sum, measurement_runtime_cuda_api_kernel_launches_sum, arguments, system_info)

                if extrap_experiment != None:
                    return True, extrap_experiment
                else:
                    return False, None

            else:
                return False, None
    
    else:
        logging.error("No .sqlite files found in provided folder.")
        sys.exit(1)
