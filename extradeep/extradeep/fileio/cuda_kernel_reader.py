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
from extradeep.fileio.nsight_systems_file_reader import create_metric_dict

from extradeep.fileio.nsight_systems_file_reader import read_cuda_kernel, populate_measurement_callpath, add_cuda_events_to_experiment

from extradeep.fileio.sqlite_helper import get_data_from_db
from extradeep.entities.system.system_info import SystemInfo


def parallel_read_cuda_kernel(inputs):
    """
    parallel_read_cuda_kernel function...
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

    # CUDA kernels
    cuda_kernel_runtime = {}
    cuda_kernel_visits = {}
    callpaths_kernel = []

    # load the cuda_kernel events data
    kernel_callpaths, kernel_visits, kernel_runtimes = read_cuda_kernel(cursor)

    if kernel_callpaths != None and kernel_visits != None and kernel_runtimes != None:
            
        cuda_kernel_visits = kernel_visits
        cuda_kernel_runtime = kernel_runtimes
        callpaths_kernel = kernel_callpaths

        # remove outliers, should remove first epoch with 
        # high initialization and optimization overhead and others
        #runtimes = remove_outliers(runtimes)

    data = (config, cuda_kernel_visits, cuda_kernel_runtime, callpaths_kernel)

    return data


def read_kernel(dir_name, arguments):
    
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

        # CUDA kernels
        cuda_kernel_runtime = {}
        cuda_kernel_visits = {}
        callpaths_kernel = []

        # create input array for parallel epoch i/o
        inputs = []
        for i in range(len(paths)):
            inputs.append([i, paths[i], files[i]])

        # create process pool for parallel epoch i/o
        with Pool(processes=max_processes) as pool:
            parallel_result = list(tqdm(pool.imap(parallel_read_cuda_kernel, inputs), total=len(paths), desc="Reading Nsight Systems .sqlite files"))

        # construct the measurement dict from the results of the different processes
        for i in range(len(parallel_result)):
            config = parallel_result[i][0]
            visits = parallel_result[i][1]
            runtime = parallel_result[i][2]
            callpaths = parallel_result[i][3]
            if visits != None:
                cuda_kernel_visits[config.id] = visits
            if runtime != None:
                cuda_kernel_runtime[config.id] = runtime
            if callpaths != None:
                callpaths_kernel.append(callpaths)
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

            callpath_list_cuda_kernel = None
            measurement_visits_cuda_kernel = None
            measurement_runtime_cuda_kernel = None

            # if data for cuda kernel analysis exists do the preprocessing for it
            if bool(cuda_kernel_visits) != False and len(callpaths_kernel) != 0:

                # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
                concat_kernel_list_cuda_kernel = concat_unique_kernel_list(callpaths_kernel)
            
                # create new free metric dict for visits
                measurement_visits_cuda_kernel = create_metric_dict(experiment_config_ids, concat_kernel_list_cuda_kernel)
                
                # create new free metric dict for runtime sum
                measurement_runtime_cuda_kernel = create_metric_dict(experiment_config_ids, concat_kernel_list_cuda_kernel)

                # populate the metric dicts and get the callpaths
                callpath_list_cuda_kernel, measurement_visits_cuda_kernel, measurement_runtime_cuda_kernel = populate_measurement_callpath(concat_kernel_list_cuda_kernel, experiment_config_ids, cuda_kernel_visits, cuda_kernel_runtime, measurement_visits_cuda_kernel, measurement_runtime_cuda_kernel)

            # if measurements exits create experiment
            if measurement_runtime_cuda_kernel != None or measurement_visits_cuda_kernel != None or callpath_list_cuda_kernel != None:

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

                # add data for cuda kernels to experiment
                extrap_experiment = add_cuda_events_to_experiment(extrap_experiment, pbar, callpath_list_cuda_kernel, measurement_visits_cuda_kernel, unique_configs, measurement_runtime_cuda_kernel, arguments, system_info)

                if extrap_experiment != None:
                    return True, extrap_experiment
                else:
                    return False, None

            else:
                return False, None
    
    else:
        logging.error("No .sqlite files found in provided folder.")
        sys.exit(1)
