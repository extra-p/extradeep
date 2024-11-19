# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import numpy as np
import sqlite3
import math

from pathlib import Path

from extradeep.entities.experiment import Experiment
from extradeep.util.exceptions import FileFormatError
from extradeep.util.progress_bar import DUMMY_PROGRESS
from extradeep.entities.parameter import Parameter
from extradeep.entities.metric import Metric
from extradeep.entities.measurement import Measurement
from extradeep.entities.coordinate import Coordinate
from extradeep.entities.callpath import Callpath
from extradeep.entities.experiment import Experiment
from extradeep.entities.analysistype import AnalysisType
from extradeep.fileio.io_helper import create_call_tree
from extradeep.util.util_functions import get_sqlite_files_in_path
from extradeep.fileio.sqlite_helper import read_experiment_configurations 
from extradeep.fileio.sqlite_helper import get_data_from_db
from extradeep.fileio.sqlite_helper import convert_nvtx_events_to_objects
from extradeep.fileio.sqlite_helper import convert_cupti_kernels_to_objects
from extradeep.fileio.sqlite_helper import convert_mpi_events_to_objects
from extradeep.fileio.sqlite_helper import convert_cublas_events_to_objects
from extradeep.fileio.sqlite_helper import convert_cudnn_events_to_objects
from extradeep.fileio.sqlite_helper import convert_cupti_runtimes_to_objects
from extradeep.fileio.sqlite_helper import convert_cupti_synchronize_to_objects
from extradeep.fileio.sqlite_helper import convert_os_events_to_objects
from extradeep.fileio.sqlite_helper import convert_memset_events_to_objects
from extradeep.fileio.sqlite_helper import convert_memcopy_events_to_objects
from extradeep.fileio.sqlite_helper import convert_cupti_kernel_launch_to_objects
from extradeep.util.util_functions import remove_outliers, get_outliers


def read_nsight_systems_files(dir_name, scaling_type, pbar=DUMMY_PROGRESS):

    # read the nsight systems .sqlite files in the given directory with dir_name
    path = Path(dir_name)
    if not path.is_dir():
        raise FileFormatError(f'Nsight Systems file path must point to a directory: {dir_name}')
    nisght_systems_files = list(path.glob('*[!.]*.sqlite'))
    if not nisght_systems_files:
        raise FileFormatError(f'No Nsight Systems .sqlite files were found in: {dir_name}')

    paths, files = get_sqlite_files_in_path(dir_name)

    pbar.total += len(paths) + 2
    pbar.step("Reading Nsight Systems .sqlite files")

    # iterate through all files and then get the data back
    data = read_files(paths, pbar, files)

    configs = data[0]
    callpaths_nvtx = data[1]
    nvtx_visits = data[2] 
    nvtx_runtime = data[3] 
    nvtx_training_step_runtimes = data[4] 
    cuda_kernel_runtime = data[5] 
    cuda_kernel_visits = data[6] 
    callpaths_kernel = data[7]
    mpi_event_visits = data[8]
    mpi_event_runtime = data[9]
    callpaths_mpi = data[10]
    analysis_types = data[11]
    cublas_event_visits = data[12]
    cublas_event_runtime = data[13]
    callpaths_cublas = data[14]
    cudnn_event_visits = data[15]
    cudnn_event_runtime = data[16]
    callpaths_cudnn = data[17]
    cuda_api_event_visits = data[18]
    cuda_api_event_runtime = data[19]
    callpaths_cuda_api = data[20]
    os_event_visits = data[21]
    os_event_runtime = data[22]
    callpaths_os = data[23]
    memset_event_visits = data[24]
    memset_event_runtime = data[25]
    memcopy_event_visits = data[26]
    memcopy_event_runtime = data[27]
    memset_event_bytes = data[28]
    memcopy_event_bytes = data[29]
    memcopy_htod_event_visits = data[30]
    memcopy_dtoh_event_visits = data[31]
    memcopy_dtod_event_visits = data[32]
    memcopy_htoh_event_visits = data[33]
    memcopy_htod_event_runtime = data[34]
    memcopy_dtoh_event_runtime = data[35]
    memcopy_dtod_event_runtime = data[36]
    memcopy_htoh_event_runtime = data[37]
    memcopy_htod_event_bytes = data[38]
    memcopy_dtoh_event_bytes = data[39]
    memcopy_dtod_event_bytes = data[40]
    memcopy_htoh_event_bytes = data[41]
    callpaths_cuda_api_kernel_launches = data[42]
    cuda_api_kernel_launch_visits = data[43]
    cuda_api_kernel_launch_runtime = data[44]
    cuda_api_kernel_launch_visits_sum = data[45]
    cuda_api_kernel_launch_runtime_sum = data[46]
    nvtx_epoch_runtimes = data[47]
    nvtx_testing_step_runtimes = data[48] 
    
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

    # if data for nvtx analysis exists do the preprocessing for it
    if bool(nvtx_visits) != False and len(callpaths_nvtx) != 0:

        # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
        concat_kernel_list_nvtx = concat_unique_kernel_list(callpaths_nvtx)
    
        # create new free metric dict for visits
        measurement_visits_nvtx = create_metric_dict(experiment_config_ids, concat_kernel_list_nvtx)
        
        # create new free metric dict for runtime sum
        measurement_runtime_nvtx = create_metric_dict(experiment_config_ids, concat_kernel_list_nvtx)

        # populate the metric dicts and get the callpaths
        callpath_list_nvtx, measurement_visits_nvtx, measurement_runtime_nvtx = populate_measurement_callpath(concat_kernel_list_nvtx, experiment_config_ids, nvtx_visits, nvtx_runtime, measurement_visits_nvtx, measurement_runtime_nvtx)

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

    # if data for mpi analysis exists do the preprocessing for it
    if bool(mpi_event_visits) != False and len(callpaths_mpi) != 0:

        # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
        concat_kernel_list_mpi_events = concat_unique_kernel_list(callpaths_mpi)
    
        # create new free metric dict for visits
        measurement_visits_mpi_events = create_metric_dict(experiment_config_ids, concat_kernel_list_mpi_events)
        
        # create new free metric dict for runtime sum
        measurement_runtime_mpi_events = create_metric_dict(experiment_config_ids, concat_kernel_list_mpi_events)

        # populate the metric dicts and get the callpaths
        callpath_list_mpi_events, measurement_visits_mpi_events, measurement_runtime_mpi_events = populate_measurement_callpath(concat_kernel_list_mpi_events, experiment_config_ids, mpi_event_visits, mpi_event_runtime, measurement_visits_mpi_events, measurement_runtime_mpi_events)

    # if data for cublas analysis exists do the preprocessing for it
    if bool(cublas_event_visits) != False and len(callpaths_cublas) != 0:
        
        # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
        concat_kernel_list_cublas_events = concat_unique_kernel_list(callpaths_cublas)
    
        # create new free metric dict for visits
        measurement_visits_cublas_events = create_metric_dict(experiment_config_ids, concat_kernel_list_cublas_events)
        
        # create new free metric dict for runtime sum
        measurement_runtime_cublas_events = create_metric_dict(experiment_config_ids, concat_kernel_list_cublas_events)

        # populate the metric dicts and get the callpaths
        callpath_list_cublas_events, measurement_visits_cublas_events, measurement_runtime_cublas_events = populate_measurement_callpath(concat_kernel_list_cublas_events, experiment_config_ids, cublas_event_visits, cublas_event_runtime, measurement_visits_cublas_events, measurement_runtime_cublas_events)

    # if data for cudnn analysis exists do the preprocessing for it
    if bool(cudnn_event_visits) != False and len(callpaths_cudnn) != 0:
        
        # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
        concat_kernel_list_cudnn_events = concat_unique_kernel_list(callpaths_cudnn)
    
        # create new free metric dict for visits
        measurement_visits_cudnn_events = create_metric_dict(experiment_config_ids, concat_kernel_list_cudnn_events)
        
        # create new free metric dict for runtime sum
        measurement_runtime_cudnn_events = create_metric_dict(experiment_config_ids, concat_kernel_list_cudnn_events)

        # populate the metric dicts and get the callpaths
        callpath_list_cudnn_events, measurement_visits_cudnn_events, measurement_runtime_cudnn_events = populate_measurement_callpath(concat_kernel_list_cudnn_events, experiment_config_ids, cudnn_event_visits, cudnn_event_runtime, measurement_visits_cudnn_events, measurement_runtime_cudnn_events)

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

    # if data for os analysis exists do the preprocessing for it
    if bool(os_event_visits) != False and len(callpaths_os) != 0:
        
        # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
        concat_kernel_list_os_events = concat_unique_kernel_list(callpaths_os)
    
        # create new free metric dict for visits
        measurement_visits_os_events = create_metric_dict(experiment_config_ids, concat_kernel_list_os_events)
        
        # create new free metric dict for runtime sum
        measurement_runtime_os_events = create_metric_dict(experiment_config_ids, concat_kernel_list_os_events)

        # populate the metric dicts and get the callpaths
        callpath_list_os_events, measurement_visits_os_events, measurement_runtime_os_events = populate_measurement_callpath(concat_kernel_list_os_events, experiment_config_ids, os_event_visits, os_event_runtime, measurement_visits_os_events, measurement_runtime_os_events)

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

    # if data for training step analysis exists do the preprocessing for it
    if bool(nvtx_training_step_runtimes) != False:

        measurement_training_step_runtime = create_measurement_list(experiment_config_ids, nvtx_training_step_runtimes)

    # if data for epoch analysis exists do the preprocessing for it
    if bool(nvtx_epoch_runtimes) != False:

        measurement_epoch_runtime = create_measurement_list(experiment_config_ids, nvtx_epoch_runtimes)

    # if data for testing step analysis exists do the preprocessing for it
    if bool(nvtx_testing_step_runtimes) != False:

        measurement_testing_step_runtime = create_measurement_list(experiment_config_ids, nvtx_testing_step_runtimes)

    #TODO: Add the other advanced analysis types here...

    # create an experiment
    pbar.step("Create experiment")
    pbar.update(1)

    # pack data for creating the experiments
    exp_data = (
        unique_configs,
        callpath_list_nvtx,
        measurement_visits_nvtx,
        measurement_runtime_nvtx,
        measurement_training_step_runtime, 
        callpath_list_cuda_kernel, 
        measurement_visits_cuda_kernel, 
        measurement_runtime_cuda_kernel,
        callpath_list_mpi_events,
        measurement_visits_mpi_events,
        measurement_runtime_mpi_events,
        analysis_types, 
        pbar, 
        scaling_type,
        callpath_list_cublas_events,
        measurement_visits_cublas_events,
        measurement_runtime_cublas_events,
        callpath_list_cudnn_events,
        measurement_visits_cudnn_events,
        measurement_runtime_cudnn_events,
        callpath_list_cuda_api_events,
        measurement_visits_cuda_api_events,
        measurement_runtime_cuda_api_events,
        callpath_list_os_events,
        measurement_visits_os_events,
        measurement_runtime_os_events,
        measurement_memset_runtime,
        measurement_memset_visits,
        measurement_memcopy_runtime,
        measurement_memcopy_visits,
        measurement_memset_bytes,
        measurement_memcopy_bytes,
        measurement_memcopy_htod_visits,
        measurement_memcopy_dtoh_visits,
        measurement_memcopy_dtod_visits,
        measurement_memcopy_htoh_visits,
        measurement_memcopy_htod_runtime,
        measurement_memcopy_dtoh_runtime,
        measurement_memcopy_dtod_runtime,
        measurement_memcopy_htoh_runtime,
        measurement_memcopy_htod_bytes,
        measurement_memcopy_dtoh_bytes,
        measurement_memcopy_dtod_bytes,
        measurement_memcopy_htoh_bytes,
        callpath_list_cuda_api_kernel_launches,
        measurement_visits_cuda_api_kernel_launches,
        measurement_runtime_cuda_api_kernel_launches,
        measurement_visits_cuda_api_kernel_launches_sum,
        measurement_runtime_cuda_api_kernel_launches_sum,
        measurement_epoch_runtime,
        measurement_testing_step_runtime
    )

    # convert the dicts and other info to Extra-P objects to prepare for modeling
    extrap_experiment = create_experiment(exp_data)

    return extrap_experiment


def create_experiment(exp_data):
    
    # unpack the experiment data
    unique_configs = exp_data[0]
    callpath_list_nvtx = exp_data[1]
    measurement_visits_nvtx = exp_data[2]
    measurement_runtime_nvtx = exp_data[3]
    measurement_training_step_runtime = exp_data[4]
    callpath_list_cuda_kernel = exp_data[5]
    measurement_visits_cuda_kernel = exp_data[6]
    measurement_runtime_cuda_kernel = exp_data[7]
    callpath_list_mpi_events = exp_data[8]
    measurement_visits_mpi_events = exp_data[9]
    measurement_runtime_mpi_events = exp_data[10]
    analysis_types = exp_data[11]
    pbar = exp_data[12]
    scaling_type = exp_data[13]
    callpath_list_cublas_events = exp_data[14]
    measurement_visits_cublas_events = exp_data[15]
    measurement_runtime_cublas_events = exp_data[16]
    callpath_list_cudnn_events = exp_data[17]
    measurement_visits_cudnn_events = exp_data[18]
    measurement_runtime_cudnn_events = exp_data[19]
    callpath_list_cuda_api_events = exp_data[20]
    measurement_visits_cuda_api_events = exp_data[21]
    measurement_runtime_cuda_api_events = exp_data[22]
    callpath_list_os_events = exp_data[23]
    measurement_visits_os_events = exp_data[24]
    measurement_runtime_os_events = exp_data[25]
    measurement_memset_runtime = exp_data[26]
    measurement_memset_visits = exp_data[27]
    measurement_memcopy_runtime = exp_data[28]
    measurement_memcopy_visits = exp_data[29]
    measurement_memset_bytes = exp_data[30]
    measurement_memcopy_bytes = exp_data[31]
    measurement_memcopy_htod_visits = exp_data[32]
    measurement_memcopy_dtoh_visits = exp_data[33]
    measurement_memcopy_dtod_visits = exp_data[34]
    measurement_memcopy_htoh_visits = exp_data[35]
    measurement_memcopy_htod_runtime = exp_data[36]
    measurement_memcopy_dtoh_runtime = exp_data[37]
    measurement_memcopy_dtod_runtime = exp_data[38]
    measurement_memcopy_htoh_runtime = exp_data[39]
    measurement_memcopy_htod_bytes = exp_data[40]
    measurement_memcopy_dtoh_bytes = exp_data[41]
    measurement_memcopy_dtod_bytes = exp_data[42]
    measurement_memcopy_htoh_bytes = exp_data[43]
    callpath_list_cuda_api_kernel_launches = exp_data[44]
    measurement_visits_cuda_api_kernel_launches = exp_data[45]
    measurement_runtime_cuda_api_kernel_launches = exp_data[46]
    measurement_visits_cuda_api_kernel_launches_sum = exp_data[47]
    measurement_runtime_cuda_api_kernel_launches_sum = exp_data[48]
    measurement_epoch_runtime = exp_data[49]
    measurement_testing_step_runtime = exp_data[50]
    
    # create new empty extrap experiment
    extrap_experiment = Experiment()

    # add scaling type to experiment
    if scaling_type == "weak" or scaling_type == "strong":
        extrap_experiment.scaling = scaling_type

    # create and add parameters to experiment
    for x in unique_configs[0].parameter_names:
        extrap_parameter = Parameter(x)
        extrap_experiment.add_parameter(extrap_parameter)

    # add the analysis types to the experiment for switching between them in the GUI
    #for i in range(len(analysis_types)):
    #    extrap_experiment.add_analysistype(AnalysisType(str(analysis_types[i])))

    #TODO: acutally use the analysis type here...
    #print(analysis_types)

    # add data for nvtx user events to experiment
    extrap_experiment = add_nvtx_events_to_experiment(extrap_experiment, pbar, callpath_list_nvtx, unique_configs, measurement_visits_nvtx, measurement_runtime_nvtx)

    # add data for cuda kernels to experiment
    extrap_experiment = add_cuda_events_to_experiment(extrap_experiment, pbar, callpath_list_cuda_kernel, measurement_visits_cuda_kernel, unique_configs, measurement_runtime_cuda_kernel)

    # add data for mpi events to experiment
    extrap_experiment = add_mpi_events_to_experiment(extrap_experiment, callpath_list_mpi_events, measurement_visits_mpi_events, unique_configs, measurement_runtime_mpi_events, pbar)

    # add the data for cublas events to experiment
    extrap_experiment = add_cublas_events_to_experiment(extrap_experiment, callpath_list_cublas_events, measurement_visits_cublas_events, measurement_runtime_cublas_events, pbar, unique_configs)

    # add data for cudnn events to experiment
    extrap_experiment = add_cudnn_events_to_experiment(extrap_experiment, callpath_list_cudnn_events, measurement_visits_cudnn_events, measurement_runtime_cudnn_events, pbar, unique_configs)

    # add data for cuda api events to experiment
    extrap_experiment = add_cuda_api_events_to_experiment(extrap_experiment, callpath_list_cuda_api_events, measurement_visits_cuda_api_events, measurement_runtime_cuda_api_events, pbar, unique_configs, callpath_list_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches, measurement_runtime_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches_sum, measurement_runtime_cuda_api_kernel_launches_sum)

    # add data for os events to experiment
    extrap_experiment = add_os_events_to_experiment(extrap_experiment, callpath_list_os_events, measurement_visits_os_events, measurement_runtime_os_events, pbar, unique_configs)

    # add data for memory events to experiment
    extrap_experiment = add_memory_events_to_experiment(extrap_experiment, measurement_memset_runtime, measurement_memset_visits, measurement_memcopy_runtime, measurement_memcopy_visits, measurement_memset_bytes, measurement_memcopy_bytes, measurement_memcopy_htod_visits, measurement_memcopy_dtoh_visits, measurement_memcopy_dtod_visits, measurement_memcopy_htoh_visits, measurement_memcopy_htod_runtime, measurement_memcopy_dtoh_runtime, measurement_memcopy_dtod_runtime, measurement_memcopy_htoh_runtime, measurement_memcopy_htod_bytes, measurement_memcopy_dtoh_bytes, measurement_memcopy_dtod_bytes, measurement_memcopy_htoh_bytes, pbar, unique_configs)

    # add data for training step analysis to experiment
    extrap_experiment = add_training_steps_to_experiment(extrap_experiment, pbar, unique_configs, measurement_training_step_runtime)

    # add data for epoch analysis to experiment
    extrap_experiment = add_epochs_to_experiment(extrap_experiment, pbar, unique_configs, measurement_epoch_runtime)

    # add data for testing step analysis to experiment
    extrap_experiment = add_testing_steps_to_experiment(extrap_experiment, pbar, unique_configs, measurement_testing_step_runtime)

    return extrap_experiment


def add_cudnn_events_to_experiment(extrap_experiment, callpath_list_cudnn_events, measurement_visits_cudnn_events, measurement_runtime_cudnn_events, pbar, unique_configs, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("CUDNN events"))

    # create and add measurements to experiment for mpi event analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "CUDNN events")
    
    # create and add callpaths to experiment
    callpaths = []
    for i in range(len(callpath_list_cudnn_events)):
        callpaths.append(Callpath(callpath_list_cudnn_events[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_visits, measurement_visits_cudnn_events, callpath_list_cudnn_events)
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_runtime, measurement_runtime_cudnn_events, callpath_list_cudnn_events)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_cuda_api_events_to_experiment(extrap_experiment, callpath_list_cuda_api_events, measurement_visits_cuda_api_events, measurement_runtime_cuda_api_events, pbar, unique_configs, callpath_list_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches, measurement_runtime_cuda_api_kernel_launches, measurement_visits_cuda_api_kernel_launches_sum, measurement_runtime_cuda_api_kernel_launches_sum, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("CUDA API events"))

    # create and add measurements to experiment for cuda api event analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "CUDA API events")
    
    # create callpaths for kernel launches
    callpaths = []
    kernel_launcher_callpath = Callpath("Kernel launcher")
    callpaths.append(kernel_launcher_callpath)
    for i in range(len(callpath_list_cuda_api_kernel_launches)):
        callpaths.append(Callpath("Kernel launcher->"+callpath_list_cuda_api_kernel_launches[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    for l in range(len(measurement_visits_cuda_api_kernel_launches_sum)):

        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        extrap_experiment.add_coordinate(extrap_coordinate)

        callpath = kernel_launcher_callpath

        if extrap_experiment.scaling == "strong":
            # get the number of mpi ranks
            # this assumes that the mpi rank is the parameter with index 0
            mpi_ranks = float(unique_configs[l].parameter_values[0])
        
            # multiply the runtime values with the number of mpi ranks
            strong_scaling_values = measurement_runtime_cuda_api_kernel_launches_sum[l]
            for k in range(len(strong_scaling_values)):
                strong_scaling_values[k] = strong_scaling_values[k] * mpi_ranks

            # multiply the runtime values with the number of mpi ranks
            strong_scaling_values_visits = measurement_visits_cuda_api_kernel_launches_sum[l]
            for k in range(len(strong_scaling_values_visits)):
                strong_scaling_values_visits[k] = strong_scaling_values_visits[k] * mpi_ranks

        if extrap_experiment.scaling == "strong":
            extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, strong_scaling_values_visits)
            extrap_experiment.add_measurement(extrap_measurement)

            extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_runtime, current_analysis_type, strong_scaling_values)
            extrap_experiment.add_measurement(extrap_measurement)

        else:
            extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, measurement_visits_cuda_api_kernel_launches_sum[l])
            extrap_experiment.add_measurement(extrap_measurement)

            extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_runtime, current_analysis_type, measurement_runtime_cuda_api_kernel_launches_sum[l])
            extrap_experiment.add_measurement(extrap_measurement)

    for l in range(len(measurement_visits_cuda_api_kernel_launches)):

        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        extrap_experiment.add_coordinate(extrap_coordinate)

        for i in range(len(callpath_list_cuda_api_kernel_launches)):

            # get the callpath from the experiment
            callpaths = extrap_experiment.callpaths[current_analysis_type]
            id = -1
            for j in range(len(callpaths)):
                callpath = callpaths[j]
                if callpath.name == "Kernel launcher->"+callpath_list_cuda_api_kernel_launches[i]:
                    id = j
            callpath = callpaths[id]

            if extrap_experiment.scaling == "strong":
                # get the number of mpi ranks
                # this assumes that the mpi rank is the parameter with index 0
                mpi_ranks = float(unique_configs[l].parameter_values[0])
            
                # multiply the runtime values with the number of mpi ranks
                strong_scaling_values = measurement_runtime_cuda_api_kernel_launches[l][callpath_list_cuda_api_kernel_launches[i]]
                for k in range(len(strong_scaling_values)):
                    strong_scaling_values[k] = strong_scaling_values[k] * mpi_ranks

                # multiply the runtime values with the number of mpi ranks
                strong_scaling_values_visits = measurement_visits_cuda_api_kernel_launches[l][callpath_list_cuda_api_kernel_launches[i]]
                for k in range(len(strong_scaling_values_visits)):
                    strong_scaling_values_visits[k] = strong_scaling_values_visits[k] * mpi_ranks

            if extrap_experiment.scaling == "strong":
                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, strong_scaling_values_visits)
                extrap_experiment.add_measurement(extrap_measurement)
                
                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_runtime, current_analysis_type, strong_scaling_values)
                extrap_experiment.add_measurement(extrap_measurement)

            else:
                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, measurement_visits_cuda_api_kernel_launches[l][callpath_list_cuda_api_kernel_launches[i]])
                extrap_experiment.add_measurement(extrap_measurement)

                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_runtime, current_analysis_type, measurement_runtime_cuda_api_kernel_launches[l][callpath_list_cuda_api_kernel_launches[i]])
                extrap_experiment.add_measurement(extrap_measurement)

    # create and add callpaths to experiment
    for i in range(len(callpath_list_cuda_api_events)):
        callpaths.append(Callpath(callpath_list_cuda_api_events[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    for l in range(len(measurement_visits_cuda_api_events)):

        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        extrap_experiment.add_coordinate(extrap_coordinate)

        for i in range(len(callpath_list_cuda_api_events)):

            # get the callpath from the experiment
            callpaths = extrap_experiment.callpaths[current_analysis_type]
            id = -1
            for j in range(len(callpaths)):
                callpath = callpaths[j]
                if callpath.name == callpath_list_cuda_api_events[i]:
                    id = j
            callpath = callpaths[id]

            if extrap_experiment.scaling == "strong":
                # get the number of mpi ranks
                # this assumes that the mpi rank is the parameter with index 0
                mpi_ranks = float(unique_configs[l].parameter_values[0])
            
                # multiply the runtime values with the number of mpi ranks
                strong_scaling_values = measurement_runtime_cuda_api_events[l][callpath_list_cuda_api_events[i]]
                for k in range(len(strong_scaling_values)):
                    strong_scaling_values[k] = strong_scaling_values[k] * mpi_ranks

                # multiply the runtime values with the number of mpi ranks
                strong_scaling_values_visits = measurement_visits_cuda_api_events[l][callpath_list_cuda_api_events[i]]
                for k in range(len(strong_scaling_values_visits)):
                    strong_scaling_values_visits[k] = strong_scaling_values_visits[k] * mpi_ranks
            
            extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, measurement_visits_cuda_api_events[l][callpath_list_cuda_api_events[i]])
            extrap_experiment.add_measurement(extrap_measurement)

            if extrap_experiment.scaling == "strong":
                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, strong_scaling_values_visits)
                extrap_experiment.add_measurement(extrap_measurement)

                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_runtime, current_analysis_type, strong_scaling_values)
                extrap_experiment.add_measurement(extrap_measurement)

            else:
                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_visits, current_analysis_type, measurement_visits_cuda_api_events[l][callpath_list_cuda_api_events[i]])
                extrap_experiment.add_measurement(extrap_measurement)

                extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric_runtime, current_analysis_type, measurement_runtime_cuda_api_events[l][callpath_list_cuda_api_events[i]])
                extrap_experiment.add_measurement(extrap_measurement)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_os_events_to_experiment(extrap_experiment, callpath_list_os_events, measurement_visits_os_events, measurement_runtime_os_events, pbar, unique_configs, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("OS events"))

    # create and add measurements to experiment for mpi event analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "OS events")
    
    # create and add callpaths to experiment
    callpaths = []
    for i in range(len(callpath_list_os_events)):
        callpaths.append(Callpath(callpath_list_os_events[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_visits, measurement_visits_os_events, callpath_list_os_events)
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_runtime, measurement_runtime_os_events, callpath_list_os_events)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_memory_events_to_experiment(extrap_experiment, measurement_memset_runtime, measurement_memset_visits, measurement_memcopy_runtime, measurement_memcopy_visits, measurement_memset_bytes, measurement_memcopy_bytes, measurement_memcopy_htod_visits, measurement_memcopy_dtoh_visits, measurement_memcopy_dtod_visits, measurement_memcopy_htoh_visits, measurement_memcopy_htod_runtime, measurement_memcopy_dtoh_runtime, measurement_memcopy_dtod_runtime, measurement_memcopy_htoh_runtime, measurement_memcopy_htod_bytes, measurement_memcopy_dtoh_bytes, measurement_memcopy_dtod_bytes, measurement_memcopy_htoh_bytes, pbar, unique_configs, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("Memory events"))

    # create and add measurements to experiment for memory event analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "Memory events")
    
    # create and add callpaths to experiment
    callpaths = []
    memset_callpath = Callpath("Memset")
    callpaths.append(memset_callpath)
    memcopy_callpath = Callpath("Memcopy")
    callpaths.append(memcopy_callpath)
    memcopy_callpath_htod = Callpath("Memcopy->HtoD")
    memcopy_callpath_dtoh = Callpath("Memcopy->DtoH")
    memcopy_callpath_dtod = Callpath("Memcopy->DtoD")
    memcopy_callpath_htoh = Callpath("Memcopy->HtoH")
    callpaths.append(memcopy_callpath_htod)
    callpaths.append(memcopy_callpath_dtoh)
    callpaths.append(memcopy_callpath_dtod)
    callpaths.append(memcopy_callpath_htoh)
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    temp_metrics.append(extrap_metric_visits)
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_runtime)
    extrap_metric_bytes = Metric("bytes")
    temp_metrics.append(extrap_metric_bytes)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memset_callpath, extrap_metric_visits, current_analysis_type, measurement_memset_visits)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memset_callpath, extrap_metric_runtime, current_analysis_type, measurement_memset_runtime)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memset_callpath, extrap_metric_bytes, current_analysis_type, measurement_memset_bytes)

    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath, extrap_metric_visits, current_analysis_type, measurement_memcopy_visits)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath, extrap_metric_runtime, current_analysis_type, measurement_memcopy_runtime)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath, extrap_metric_bytes, current_analysis_type, measurement_memcopy_bytes)

    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_htod, extrap_metric_visits, current_analysis_type, measurement_memcopy_htod_visits)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_dtoh, extrap_metric_visits, current_analysis_type, measurement_memcopy_dtoh_visits)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_dtod, extrap_metric_visits, current_analysis_type, measurement_memcopy_dtod_visits)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_htoh, extrap_metric_visits, current_analysis_type, measurement_memcopy_htoh_visits)

    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_htod, extrap_metric_runtime, current_analysis_type, measurement_memcopy_htod_runtime)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_dtoh, extrap_metric_runtime, current_analysis_type, measurement_memcopy_dtoh_runtime)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_dtod, extrap_metric_runtime, current_analysis_type, measurement_memcopy_dtod_runtime)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_htoh, extrap_metric_runtime, current_analysis_type, measurement_memcopy_htoh_runtime)

    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_htod, extrap_metric_bytes, current_analysis_type, measurement_memcopy_htod_bytes)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_dtoh, extrap_metric_bytes, current_analysis_type, measurement_memcopy_dtoh_bytes)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_dtod, extrap_metric_bytes, current_analysis_type, measurement_memcopy_dtod_bytes)
    extrap_experiment = add_measurement_list_to_experiment(extrap_experiment, unique_configs, memcopy_callpath_htoh, extrap_metric_bytes, current_analysis_type, measurement_memcopy_htoh_bytes)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_cublas_events_to_experiment(extrap_experiment, callpath_list_cublas_events, measurement_visits_cublas_events, measurement_runtime_cublas_events, pbar, unique_configs, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("CUBLAS events"))

    # create and add measurements to experiment for mpi event analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "CUBLAS events")
    
    # create and add callpaths to experiment
    callpaths = []
    for i in range(len(callpath_list_cublas_events)):
        callpaths.append(Callpath(callpath_list_cublas_events[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_visits, measurement_visits_cublas_events, callpath_list_cublas_events)
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_runtime, measurement_runtime_cublas_events, callpath_list_cublas_events)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_app_phases_to_experiment(experiment, pbar, unique_configs, 
                measurement_communication_runtime, measurement_computation_runtime, measurement_memory_runtime, 
                measurement_communication_runtime_epochs, measurement_computation_runtime_epochs, measurement_memory_runtime_epochs,
                arguments, system_info):
    """
    add_app_phases_to_experiment function ...
    """

    experiment.add_analysistype(AnalysisType("App phases"))

    current_analysis_type = get_current_analysis_type(experiment, "App phases")

    # create and add callpaths to experiment
    callpaths = []
    comp_callpath = Callpath("computation training steps")
    callpaths.append(comp_callpath)
    com_callpath = Callpath("communication training steps")
    callpaths.append(com_callpath)
    mem_callpath = Callpath("memory training steps")
    callpaths.append(mem_callpath)
    comp_callpath_epochs = Callpath("computation epochs")
    callpaths.append(comp_callpath_epochs)
    com_callpath_epochs = Callpath("communication epochs")
    callpaths.append(com_callpath_epochs)
    mem_callpath_epochs = Callpath("memory epochs")
    callpaths.append(mem_callpath_epochs)
    experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_runtime)
    experiment.add_metric(current_analysis_type, temp_metrics)

    experiment = add_measurement_list_to_experiment(experiment, unique_configs, com_callpath, extrap_metric_runtime, current_analysis_type, measurement_communication_runtime)
    experiment = add_measurement_list_to_experiment(experiment, unique_configs, comp_callpath, extrap_metric_runtime, current_analysis_type, measurement_computation_runtime)
    experiment = add_measurement_list_to_experiment(experiment, unique_configs, mem_callpath, extrap_metric_runtime, current_analysis_type, measurement_memory_runtime)
    experiment = add_measurement_list_to_experiment(experiment, unique_configs, com_callpath_epochs, extrap_metric_runtime, current_analysis_type, measurement_communication_runtime_epochs)
    experiment = add_measurement_list_to_experiment(experiment, unique_configs, comp_callpath_epochs, extrap_metric_runtime, current_analysis_type, measurement_computation_runtime_epochs)
    experiment = add_measurement_list_to_experiment(experiment, unique_configs, mem_callpath_epochs, extrap_metric_runtime, current_analysis_type, measurement_memory_runtime_epochs)

    # remove measurements and callpaths that do not have enough points for modeling
    experiment = remove_measurements_callpaths(experiment, current_analysis_type, arguments)

    # create a calltree for this analysis type
    call_tree = create_call_tree(experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
    experiment.call_trees[current_analysis_type] = call_tree
    return experiment


def add_nvtx_events_to_experiment(extrap_experiment, pbar, callpath_list_nvtx, unique_configs, measurement_visits_nvtx, measurement_runtime_nvtx, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("NVTX user instrumentation"))

    # create and add measurements to experiment for nvtx user analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "NVTX user instrumentation")

    # create and add callpaths to experiment
    callpaths = []
    for i in range(len(callpath_list_nvtx)):
        callpaths.append(Callpath(callpath_list_nvtx[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_visits, measurement_visits_nvtx, callpath_list_nvtx)
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_runtime, measurement_runtime_nvtx, callpath_list_nvtx)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_cuda_events_to_experiment(extrap_experiment, pbar, callpath_list_cuda_kernel, measurement_visits_cuda_kernel, unique_configs, measurement_runtime_cuda_kernel, arguments, system_info):
    extrap_experiment.add_analysistype(AnalysisType("CUDA kernel"))

    # create and add measurements to experiment for cuda kernel analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "CUDA kernel")

    # create and add callpaths to experiment
    callpaths = []
    for i in range(len(callpath_list_cuda_kernel)):
        callpaths.append(Callpath(callpath_list_cuda_kernel[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_visits, measurement_visits_cuda_kernel, callpath_list_cuda_kernel)
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_runtime, measurement_runtime_cuda_kernel, callpath_list_cuda_kernel)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_mpi_events_to_experiment(extrap_experiment, callpath_list_mpi_events, measurement_visits_mpi_events, unique_configs, measurement_runtime_mpi_events, pbar, arguments, system_info):
    """
    add_mpi_events_to_experiment function ...
    """
    
    extrap_experiment.add_analysistype(AnalysisType("MPI events"))

    # create and add measurements to experiment for mpi event analysis
    current_analysis_type = get_current_analysis_type(extrap_experiment, "MPI events")
    
    # create and add callpaths to experiment
    callpaths = []
    for i in range(len(callpath_list_mpi_events)):
        callpaths.append(Callpath(callpath_list_mpi_events[i]))
    extrap_experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_visits = Metric("visits")
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_visits)
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    extrap_experiment.add_metric(current_analysis_type, temp_metrics)

    # add the measurements to the experiment
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_visits, measurement_visits_mpi_events, callpath_list_mpi_events)
    extrap_experiment = add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, extrap_metric_runtime, measurement_runtime_mpi_events, callpath_list_mpi_events)

    # remove measurements and callpaths that do not have enough points for modeling
    extrap_experiment = remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments)

    if len(extrap_experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            extrap_experiment = add_speedup_measuremens(extrap_experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            extrap_experiment = add_efficiency_measurements(extrap_experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            extrap_experiment = add_cost_measurements(extrap_experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(extrap_experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        extrap_experiment.call_trees[current_analysis_type] = call_tree
        return extrap_experiment


def add_testing_steps_to_experiment(experiment, pbar, unique_configs, measurement_testing_step_runtime, arguments, system_info):
    """
    add_testing_steps_to_experiment function...
    """

    # create and add measurements to experiment for the testing step analysis
    experiment.add_analysistype(AnalysisType("Testing steps"))

    current_analysis_type = get_current_analysis_type(experiment, "Testing steps")

    # create and add callpaths to experiment
    callpaths = []
    testing_step_callpath = Callpath("testing step")
    callpaths.append(testing_step_callpath)
    experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    experiment.add_metric(current_analysis_type, temp_metrics)

    experiment = add_measurement_list_to_experiment(experiment, unique_configs, testing_step_callpath, extrap_metric_runtime, current_analysis_type, measurement_testing_step_runtime)

    # remove measurements and callpaths that do not have enough points for modeling
    experiment = remove_measurements_callpaths(experiment, current_analysis_type, arguments)

    if len(experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            experiment = add_speedup_measuremens(experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            experiment = add_efficiency_measurements(experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            experiment = add_cost_measurements(experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        experiment.call_trees[current_analysis_type] = call_tree
        return experiment


def add_training_steps_to_experiment(experiment, pbar, unique_configs, measurement_training_step_runtime, arguments, system_info):
    """
    add_training_steps_to_experiment function...
    """

    # create and add measurements to experiment for the training step analysis
    experiment.add_analysistype(AnalysisType("Training steps"))

    current_analysis_type = get_current_analysis_type(experiment, "Training steps")

    # create and add callpaths to experiment
    callpaths = []
    training_step_callpath = Callpath("training step")
    callpaths.append(training_step_callpath)
    experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    experiment.add_metric(current_analysis_type, temp_metrics)

    experiment = add_measurement_list_to_experiment(experiment, unique_configs, training_step_callpath, extrap_metric_runtime, current_analysis_type, measurement_training_step_runtime)

    # remove measurements and callpaths that do not have enough points for modeling
    experiment = remove_measurements_callpaths(experiment, current_analysis_type, arguments)

    if len(experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            experiment = add_speedup_measuremens(experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            experiment = add_efficiency_measurements(experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            experiment = add_cost_measurements(experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        experiment.call_trees[current_analysis_type] = call_tree
        return experiment


def add_epochs_to_experiment(experiment, pbar, unique_configs, measurement_epoch_runtime, arguments, system_info):
    """
    add_epochs_to_experiment function ...
    """

    # create and add measurements to experiment for the epoch analysis
    experiment.add_analysistype(AnalysisType("Epochs"))

    current_analysis_type = get_current_analysis_type(experiment, "Epochs")

    # create and add callpaths to experiment
    callpaths = []
    epoch_callpath = Callpath("epoch")
    callpaths.append(epoch_callpath)
    experiment.add_callpath(current_analysis_type, callpaths)

    # create and add metrics to experiment
    temp_metrics = []
    extrap_metric_runtime = Metric("runtime")
    temp_metrics.append(extrap_metric_runtime)
    # add additional metrics if required
    if arguments.speedup == True:
        extrap_metric_speedup = Metric("speedup")
        temp_metrics.append(extrap_metric_speedup)
    if arguments.efficiency == True:
        if arguments.speedup == True:
            extrap_metric_efficiency = Metric("parallel efficiency")
            temp_metrics.append(extrap_metric_efficiency)
        else:
            arguments.efficiency = False
            logging.error("Could not model parallel efficiency. Modeling the speedup is required for modeling the parallel efficiency.")
    if arguments.cost == True:
        if system_info.cpu_cores == None and arguments.cpucores == None:
            logging.error("The cost could not be modeled because the number of CPU cores per rank could not be read from the measurements, neither was it provided as a parameter with --cpu-cores.")
            arguments.cost = False
        else:
            cpu_cores_per_process = system_info.cpu_cores
            if arguments.cpucores:
                cpu_cores_per_process = arguments.cpucores
            extrap_metric_cost = Metric("cost")
            extrap_metric_cost_total = Metric("total cost")
            temp_metrics.append(extrap_metric_cost)
            temp_metrics.append(extrap_metric_cost_total)

    experiment.add_metric(current_analysis_type, temp_metrics)

    experiment = add_measurement_list_to_experiment(experiment, unique_configs, epoch_callpath, extrap_metric_runtime, current_analysis_type, measurement_epoch_runtime)

    # remove measurements and callpaths that do not have enough points for modeling
    experiment = remove_measurements_callpaths(experiment, current_analysis_type, arguments)

    if len(experiment.callpaths[current_analysis_type]) == 0:
        logging.error("There are no callpaths left for modeling.")
        return None
    else:
        # add measurements for additional metrics if required
        if arguments.speedup == True:
            experiment = add_speedup_measuremens(experiment, current_analysis_type, arguments, extrap_metric_speedup)
        if arguments.efficiency == True:
            experiment = add_efficiency_measurements(experiment, arguments, current_analysis_type, extrap_metric_efficiency)
        if arguments.cost == True:
            experiment = add_cost_measurements(experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process)

        # create a calltree for this analysis type
        call_tree = create_call_tree(experiment.callpaths[current_analysis_type], pbar, progress_scale=0.1)
        experiment.call_trees[current_analysis_type] = call_tree
        return experiment


def get_current_analysis_type(experiment, type_name):
    analysis_type_id = None
    for i in range(len(experiment.analysistypes)):
        if experiment.analysistypes[i].name == type_name:
            analysis_type_id = i
    current_analysis_type = experiment.analysistypes[analysis_type_id]
    return current_analysis_type


def add_cost_measurements(experiment, arguments, extrap_metric_cost, extrap_metric_cost_total, current_analysis_type, cpu_cores_per_process):
    
    # get the id of the runtime metric in the experiment
    metric_list = experiment.metrics[current_analysis_type]
    metric_id = None
    for i in range(len(metric_list)):
        if str(metric_list[i]) == "runtime":
            metric_id = i
            break

    callpath_list = experiment.callpaths[current_analysis_type]
    for i in range(len(callpath_list)):
        for j in range(len(experiment.coordinates)):
            if arguments.median == True:
                runtime = experiment.get_measurement(j, i, metric_id, current_analysis_type).median
            else:
                runtime = experiment.get_measurement(j, i, metric_id, current_analysis_type).mean

            if runtime != None:
                cost = runtime * cpu_cores_per_process
                extrap_measurement = Measurement(experiment.coordinates[j], callpath_list[i], extrap_metric_cost, current_analysis_type, cost)
                experiment.add_measurement(extrap_measurement)

    callpath_list = experiment.callpaths[current_analysis_type]
    for i in range(len(callpath_list)):
        for j in range(len(experiment.coordinates)):
            
            if arguments.median == True:
                runtime = experiment.get_measurement(j, i, metric_id, current_analysis_type).median
            else:
                runtime = experiment.get_measurement(j, i, metric_id, current_analysis_type).mean

            if runtime != None:

                # get G the number of mpi ranks
                G = None
                g_p_name = None

                if arguments.rparam.isnumeric() == True:
                    G = int(arguments.rparam)
                else:
                    g_p_name = arguments.rparam
                    g_p_id = None
                    for i in range(len(experiment.parameters)):
                        if str(experiment.parameters[i]) == g_p_name:
                            g_p_id = i
                            break
                    G = experiment.coordinates[j].__getitem__(g_p_id)

                cost = runtime * G * cpu_cores_per_process
                extrap_measurement = Measurement(experiment.coordinates[j], callpath_list[i], extrap_metric_cost_total, current_analysis_type, cost)
                experiment.add_measurement(extrap_measurement)
    return experiment


def add_efficiency_measurements(experiment, arguments, current_analysis_type, extrap_metric_efficiency):
    # first find the coordinate with the smallest parameter values as this will be the baseline for the efficiency calculation
    temp = []
    for coordinate_id in range(len(experiment.coordinates)):
        coordinate = experiment.coordinates[coordinate_id]
        dimensions = coordinate.dimensions
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

    valid_cord_id = temp[0][0]

    # get the id of the speedup metric in the experiment
    metric_list = experiment.metrics[current_analysis_type]
    metric_id = None
    for i in range(len(metric_list)):
        if str(metric_list[i]) == "speedup":
            metric_id = i
            break

    # calculate the speedup for the points based on the runtime per epoch per point
    callpath_list = experiment.callpaths[current_analysis_type]
    for i in range(len(callpath_list)):
        for j in range(len(experiment.coordinates)):
            if j == valid_cord_id:
                efficiency = 100
            else:

                g_p_name = None
                if arguments.rparam.isnumeric() == True:
                    x11 = int(arguments.rparam)
                else:
                    g_p_name = arguments.rparam
                    g_p_id = None
                    for k in range(len(experiment.parameters)):
                        if str(experiment.parameters[k]) == g_p_name:
                            g_p_id = k
                            break
                    x11 = experiment.coordinates[valid_cord_id].__getitem__(g_p_id)

                try:
                    if arguments.median == True:
                        true_speedup = experiment.get_measurement(j, i, metric_id, current_analysis_type).median
                    else:
                        true_speedup = experiment.get_measurement(j, i, metric_id, current_analysis_type).mean
                except IndexError:
                    true_speedup = -1
                    pass

                g_p_name = None
                if arguments.rparam.isnumeric() == True:
                    x1k = int(arguments.rparam)
                else:
                    g_p_name = arguments.rparam
                    g_p_id = None
                    for k in range(len(experiment.parameters)):
                        if str(experiment.parameters[k]) == g_p_name:
                            g_p_id = k
                            break
                    x1k = experiment.coordinates[j].__getitem__(g_p_id)

                theoretical_speedup = (x1k - x11) / (x11 / 100)

                if theoretical_speedup == 0:
                    efficiency = 100 - abs(true_speedup)
                else:
                    efficiency = (true_speedup / theoretical_speedup) * 100

            extrap_measurement = Measurement(experiment.coordinates[j], callpath_list[i], extrap_metric_efficiency, current_analysis_type, efficiency)
            experiment.add_measurement(extrap_measurement)
    return experiment


def add_speedup_measuremens(experiment, current_analysis_type, arguments, extrap_metric_speedup):
    # first find the coordinate with the smallest parameter values as this will be the baseline for the speedup calculation
    temp = []
    for coordinate_id in range(len(experiment.coordinates)):
        coordinate = experiment.coordinates[coordinate_id]
        dimensions = coordinate.dimensions
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

    # find the runtime metric id
    metric_list = experiment.metrics[current_analysis_type]
    runtime_metric_id = None
    for i in range(len(metric_list)):
        if str(metric_list[i]) == "runtime":
            runtime_metric_id = i
            break

    valid_cord_id = temp[0][0]

    callpath_list = experiment.callpaths[current_analysis_type]

    # calculate the speedup for the points based on the runtime for each callpath per point
    for i in range(len(callpath_list)):
        for j in range(len(experiment.coordinates)):
            if j == valid_cord_id:
                speedup = 0
            else:
                if arguments.median == True:
                    prev_value = experiment.get_measurement(valid_cord_id, i, runtime_metric_id, current_analysis_type).median
                else:
                    prev_value = experiment.get_measurement(valid_cord_id, i, runtime_metric_id, current_analysis_type).mean
                try:
                    if arguments.median == True:
                        curr_value = experiment.get_measurement(j, i, runtime_metric_id, current_analysis_type).median
                    else:
                        curr_value = experiment.get_measurement(j, i, runtime_metric_id, current_analysis_type).mean
                except IndexError:
                    curr_value = -1
                    pass

                onep = prev_value / 100
                difference = prev_value - curr_value
                if difference == 0 or onep == 0:
                    speedup = 0
                else:
                    speedup = difference / onep

            extrap_measurement = Measurement(experiment.coordinates[j], callpath_list[i], extrap_metric_speedup, current_analysis_type, speedup)
            experiment.add_measurement(extrap_measurement)
    return experiment


def read_files(paths, pbar, files):
    """
    read_files function to read all Nsight Systems .sqlite files in the given directory
    """

    #system info
    system_info = None

    # variable to save all seen experiment configurations
    configs = []

    # variable for different analysis types
    analysis_types = []

    # data container for different metrics

    # NVTX user instrumentation
    nvtx_visits = {}
    nvtx_runtime = {}
    callpaths_nvtx = []

    # Training step analysis
    nvtx_training_step_runtimes = {}

    # Epoch analysis
    nvtx_epoch_runtimes = {}

    # CUDA kernels
    cuda_kernel_runtime = {}
    cuda_kernel_visits = {}
    callpaths_kernel = []

    # MPI events
    mpi_event_visits = {}
    mpi_event_runtime = {}
    callpaths_mpi = []

    # CUBLAS events
    cublas_event_visits = {}
    cublas_event_runtime = {}
    callpaths_cublas = []

    # CUDNN events
    cudnn_event_visits = {}
    cudnn_event_runtime = {}
    callpaths_cudnn = []

    # CUDA API events
    cuda_api_event_visits = {}
    cuda_api_event_runtime = {}
    callpaths_cuda_api = []
    callpaths_cuda_api_kernel_launches = []
    cuda_api_kernel_launch_visits = {}
    cuda_api_kernel_launch_runtime = {}
    cuda_api_kernel_launch_visits_sum = {}
    cuda_api_kernel_launch_runtime_sum = {}

    # OS events
    os_event_visits = {}
    os_event_runtime = {}
    callpaths_os = []

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

    # Testing step analysis
    nvtx_testing_step_runtimes = {}


    # read system info from first sqlite file
    #db = sqlite3.connect(value)
    #cursor = db.cursor()
    #query = "SELECT value AS cpu_cores FROM TARGET_INFO_SYSTEM_ENV WHERE name LIKE \"CpuCores\";"
    #result = get_data_from_db(cursor, query)
    #print(result)
    #system_info

    # iterate over all Nsight Systems .sqlite files and read the data in them
    for count, value in enumerate(paths):

        # update the progress bar
        pbar.update(1)

        # get the performance experiment configuration of this file
        config = read_experiment_configurations(count, files[count])
        configs.append(config)

        # open database in path and create connection
        db = sqlite3.connect(value)
        cursor = db.cursor()

        # load the nvtx events created by the user with custom text
        nvtx_user_events_runtime, nvtx_user_events_visits = read_nvtx_user_instrumentation(cursor)

        if nvtx_user_events_runtime != None and nvtx_user_events_visits != None:
            nvtx_visits[config.id] = nvtx_user_events_visits
            nvtx_runtime[config.id] = nvtx_user_events_runtime

            # concat a list of unique kernels
            concat_kernel_list_nvtx = []
            for key, value in enumerate(nvtx_user_events_runtime):
                concat_kernel_list_nvtx.append(value)

            # save the unique kernel list for this experiment config
            callpaths_nvtx.append(concat_kernel_list_nvtx)

            if not bool(any(item in analysis_types for item in analysis_types)):
                analysis_types.append("NVTX user instrumentation")

        # load the nvtx events for the training step analysis
        nvtx_training_steps = read_nvtx_training_steps(cursor)

        # if training steps could be identified do some analysis
        if nvtx_training_steps != None:
            
            # add the training step analysis to the selection menue
            if "Training steps" not in analysis_types:
                analysis_types.append("Training steps")

            # get all training step runtimes available
            runtimes = []
            for i in range(len(nvtx_training_steps)):
                nvtx_training_step = nvtx_training_steps[i]
                runtime = nvtx_training_step.run_time_seconds
                runtimes.append(runtime)

            # remove outliers, this removes the steps with high initialization in the beginning of an epoch
            runtimes = remove_outliers(runtimes)

            # compute the mean of the runtimes for modeling
            mean = np.mean(runtimes)

            # add the mean runtime for current experiment configuration
            nvtx_training_step_runtimes[config.id] = mean

        # load the nvtx events for the testing step analysis
        nvtx_testing_steps = read_nvtx_testing_steps(cursor)

        # if training steps could be identified do some analysis
        if nvtx_testing_steps != None:
            
            # add the training step analysis to the selection menue
            if "Testing steps" not in analysis_types:
                analysis_types.append("Testing steps")

            # get all training step runtimes available
            runtimes = []
            for i in range(len(nvtx_testing_steps)):
                nvtx_testing_step = nvtx_testing_steps[i]
                runtime = nvtx_testing_step.run_time_seconds
                runtimes.append(runtime)

            # remove outliers, this removes the steps with high initialization in the beginning of an epoch
            runtimes = remove_outliers(runtimes)

            # compute the mean of the runtimes for modeling
            mean = np.mean(runtimes)

            # add the mean runtime for current experiment configuration
            nvtx_testing_step_runtimes[config.id] = mean

        # load the nvtx epoch events for epoch analysis
        nvtx_epochs = read_nvtx_epochs(cursor)

        # if epochs exist
        if nvtx_epochs != None:

            # add the training step analysis to the selection menue
            if "Epochs" not in analysis_types:
                analysis_types.append("Epochs")

            # get all epoch runtimes available
            runtimes = []
            for i in range(len(nvtx_epochs)):
                nvtx_epoch = nvtx_epochs[i]
                runtime = nvtx_epoch.run_time_seconds
                runtimes.append(runtime)

            # remove outliers, should remove first epoch with high initialization and optimization overhead
            runtimes = remove_outliers(runtimes)

            # compute the mean of the runtimes for modeling
            mean = np.mean(runtimes)

            # add the mean runtime for current experiment configuration
            nvtx_epoch_runtimes[config.id] = mean

        # load the cuda kernel data
        kernel_callpaths, kernel_visits, kernel_runtimes = read_cuda_kernel(cursor)

        if kernel_callpaths != None and kernel_runtimes != None and kernel_visits != None:

            # add the training step analysis to the selection menue
            if "Cuda kernels" not in analysis_types:
                analysis_types.append("Cuda kernels")

            cuda_kernel_visits[config.id] = kernel_visits
            cuda_kernel_runtime[config.id] = kernel_runtimes

            # save the unique kernel list for this experiment config
            callpaths_kernel.append(kernel_callpaths)

        # load the mpi events data 
        mpi_callpaths, mpi_visits, mpi_runtime = read_mpi_events(cursor)

        if mpi_callpaths != None and mpi_visits != None and mpi_runtime != None:

            # add the MPI analysis to the selection menue
            if "MPI events" not in analysis_types:
                analysis_types.append("MPI events")

            mpi_event_visits[config.id] = mpi_visits
            mpi_event_runtime[config.id] = mpi_runtime

            # save the unique kernel list for this experiment config
            callpaths_mpi.append(mpi_callpaths)

        # load the cublas events data
        cublas_callpaths, cublas_visits, cublas_runtime = read_cublas_events(cursor)

        if cublas_callpaths != None and cublas_visits != None and cublas_runtime != None:
            
            # add the cublas analysis to the selection menue
            if "CUBLAS events" not in analysis_types:
                analysis_types.append("CUBLAS events")

            cublas_event_visits[config.id] = cublas_visits
            cublas_event_runtime[config.id] = cublas_runtime

            # save the unique kernel list for this experiment config
            callpaths_cublas.append(cublas_callpaths)
        
        # load the cudnn events data
        cudnn_callpaths, cudnn_visits, cudnn_runtime = read_cudnn_events(cursor)

        if cudnn_callpaths != None and cudnn_visits != None and cudnn_runtime != None:
            
            # add the CUDNN analysis to the selection menue
            if "CUDNN events" not in analysis_types:
                analysis_types.append("CUDNN events")

            cudnn_event_visits[config.id] = cudnn_visits
            cudnn_event_runtime[config.id] = cudnn_runtime

            # save the unique kernel list for this experiment config
            callpaths_cudnn.append(cudnn_callpaths)

        # load the cuda api events data
        cuda_api_callpaths, cuda_api_visits, cuda_api_runtime, unique_cuda_kernel_launches, cuda_api_kernel_launches_runtime, cuda_api_kernel_launches_visits, cuda_api_kernel_launches_visits_sum, cuda_api_kernel_launches_runtime_sum = read_cuda_api_events(cursor)

        if cuda_api_callpaths != None and cuda_api_visits != None and cuda_api_runtime != None:
            
            # add the CUDA API analysis to the selection menue
            if "CUDA API events" not in analysis_types:
                analysis_types.append("CUDA API events")

            cuda_api_event_visits[config.id] = cuda_api_visits
            cuda_api_event_runtime[config.id] = cuda_api_runtime

            # save the unique kernel list for this experiment config
            callpaths_cuda_api.append(cuda_api_callpaths)

            callpaths_cuda_api_kernel_launches.append(unique_cuda_kernel_launches)
            cuda_api_kernel_launch_visits[config.id] = cuda_api_kernel_launches_visits
            cuda_api_kernel_launch_runtime[config.id] = cuda_api_kernel_launches_runtime
            cuda_api_kernel_launch_visits_sum[config.id] = cuda_api_kernel_launches_visits_sum
            cuda_api_kernel_launch_runtime_sum[config.id] = cuda_api_kernel_launches_runtime_sum

        # load the os events data
        os_callpaths, os_visits, os_runtime = read_os_events(cursor)

        if os_callpaths != None and os_visits != None and os_runtime != None:
            
            # add the OS analysis to the selection menue
            if "OS events" not in analysis_types:
                analysis_types.append("OS events")

            os_event_visits[config.id] = os_visits
            os_event_runtime[config.id] = os_runtime

            # save the unique kernel list for this experiment config
            callpaths_os.append(os_callpaths)

        # load the memory events data
        memset_visits, memset_runtime, memcopy_visits, memcopy_runtime, memset_bytes, memcopy_bytes, memcopy_htod_visits, memcopy_dtoh_visits, memcopy_dtod_visits, memcopy_htoh_visits, memcopy_htod_runtime, memcopy_dtoh_runtime, memcopy_dtod_runtime, memcopy_htoh_runtime, memcopy_htod_bytes, memcopy_dtoh_bytes, memcopy_dtod_bytes, memcopy_htoh_bytes = read_memory_events(cursor)

        if memset_visits != None and memset_runtime != None and memcopy_visits != None and memcopy_runtime != None:
            
            # add the Memory analysis to the selection menue
            if "Memory events" not in analysis_types:
                analysis_types.append("Memory events")

            memset_event_visits[config.id] = memset_visits
            memset_event_runtime[config.id] = memset_runtime
            memcopy_event_visits[config.id] = memcopy_visits
            memcopy_event_runtime[config.id] = memcopy_runtime
            memset_event_bytes[config.id] = memset_bytes
            memcopy_event_bytes[config.id] = memcopy_bytes

            memcopy_htod_event_visits[config.id] = memcopy_htod_visits
            memcopy_dtoh_event_visits[config.id] = memcopy_dtoh_visits
            memcopy_dtod_event_visits[config.id] = memcopy_dtod_visits
            memcopy_htoh_event_visits[config.id] = memcopy_htoh_visits 
            memcopy_htod_event_runtime[config.id] = memcopy_htod_runtime
            memcopy_dtoh_event_runtime[config.id] = memcopy_dtoh_runtime
            memcopy_dtod_event_runtime[config.id] = memcopy_dtod_runtime
            memcopy_htoh_event_runtime[config.id] = memcopy_htoh_runtime 
            memcopy_htod_event_bytes[config.id] = memcopy_htod_bytes 
            memcopy_dtoh_event_bytes[config.id] = memcopy_dtoh_bytes 
            memcopy_dtod_event_bytes[config.id] = memcopy_dtod_bytes 
            memcopy_htoh_event_bytes[config.id] = memcopy_htoh_bytes

        # pack all the data into a container
        data = (
            configs, 
            callpaths_nvtx, 
            nvtx_visits, 
            nvtx_runtime, 
            nvtx_training_step_runtimes, 
            cuda_kernel_runtime, 
            cuda_kernel_visits, 
            callpaths_kernel,
            mpi_event_visits,
            mpi_event_runtime,
            callpaths_mpi,
            analysis_types,
            cublas_event_visits,
            cublas_event_runtime,
            callpaths_cublas,
            cudnn_event_visits,
            cudnn_event_runtime,
            callpaths_cudnn,
            cuda_api_event_visits,
            cuda_api_event_runtime,
            callpaths_cuda_api,
            os_event_visits,
            os_event_runtime,
            callpaths_os,
            memset_event_visits,
            memset_event_runtime,
            memcopy_event_visits,
            memcopy_event_runtime,
            memset_event_bytes,
            memcopy_event_bytes,
            memcopy_htod_event_visits,
            memcopy_dtoh_event_visits,
            memcopy_dtod_event_visits,
            memcopy_htoh_event_visits,
            memcopy_htod_event_runtime,
            memcopy_dtoh_event_runtime,
            memcopy_dtod_event_runtime,
            memcopy_htoh_event_runtime,
            memcopy_htod_event_bytes,
            memcopy_dtoh_event_bytes,
            memcopy_dtod_event_bytes,
            memcopy_htoh_event_bytes,
            callpaths_cuda_api_kernel_launches,
            cuda_api_kernel_launch_visits,
            cuda_api_kernel_launch_runtime,
            cuda_api_kernel_launch_visits_sum,
            cuda_api_kernel_launch_runtime_sum,
            nvtx_epoch_runtimes,
            nvtx_testing_step_runtimes,
        )

    return data


def read_cublas_events(cursor):
    """
    read_cublas_events function...
    """

    query = "SELECT cublas.start, cublas.end, cublas.eventClass, cublas.globalTid, strings.value, (cublas.end - cublas.start) AS duration, (1.0 * cublas.start / 1000000000) AS start_seconds, (1.0 * cublas.end / 1000000000) AS end_seconds, (1.0 * (cublas.end - cublas.start) / 1000000000) AS duration_seconds FROM CUBLAS_EVENTS AS cublas INNER JOIN StringIds AS strings ON cublas.nameId = strings.id;"
    result = get_data_from_db(cursor, query)

    unique_cublas_events = None
    cublas_visits = None
    cublas_runtime = None

    if result is not None:

        if len(result) != 0:

            cublas_events = convert_cublas_events_to_objects(result)

            # get a list of unique MPI events by their name
            query = "SELECT Distinct strings.value FROM CUBLAS_EVENTS AS cublas INNER JOIN StringIds AS strings ON cublas.nameId = strings.id;"
            cublas_event_names = get_data_from_db(cursor, query)

            unique_cublas_events = []
            # get the unique cublas events into a list to get rid of the tuple
            for i in range(len(cublas_event_names)):
                unique_cublas_events.append(cublas_event_names[i][0])

            # sum up the duration of each unique cublas event in seconds
            cublas_runtime = {}
            for i in range(len(cublas_event_names)):
                cublas_runtime[cublas_event_names[i][0]] = 0.0
            for i in range(len(cublas_events)):
                cublas_runtime[cublas_events[i].name] += cublas_events[i].duration_seconds

            # sum up the number of visits of each unique cupti kernel
            cublas_visits = {}
            for i in range(len(cublas_event_names)):
                cublas_visits[cublas_event_names[i][0]] = 0
            for i in range(len(cublas_events)):
                cublas_visits[cublas_events[i].name] += 1

    return unique_cublas_events, cublas_visits, cublas_runtime


def read_cudnn_events(cursor):
    """
    read_cudnn_events function...
    """
    
    query = "SELECT events.start, events.end, events.eventClass, events.globalTid, strings.value, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUDNN_EVENTS AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id;"
    result = get_data_from_db(cursor, query)

    unique_cudnn_events = None
    cudnn_visits = None
    cudnn_runtime = None

    if result is not None:

        if len(result) != 0:

            cudnn_events = convert_cudnn_events_to_objects(result)

            # get a list of unique CUDNN events by their name
            query = "SELECT Distinct strings.value FROM CUDNN_EVENTS AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id;"
            cudnn_event_names = get_data_from_db(cursor, query)

            unique_cudnn_events = []
            # get the unique cudnn events into a list to get rid of the tuple
            for i in range(len(cudnn_event_names)):
                unique_cudnn_events.append(cudnn_event_names[i][0])

            # sum up the duration of each unique mpi event in seconds
            cudnn_runtime = {}
            for i in range(len(cudnn_event_names)):
                cudnn_runtime[cudnn_event_names[i][0]] = 0.0
            for i in range(len(cudnn_events)):
                cudnn_runtime[cudnn_events[i].name] += cudnn_events[i].duration_seconds

            # sum up the number of visits of each unique cupti kernel
            cudnn_visits = {}
            for i in range(len(cudnn_event_names)):
                cudnn_visits[cudnn_event_names[i][0]] = 0
            for i in range(len(cudnn_events)):
                cudnn_visits[cudnn_events[i].name] += 1

    return unique_cudnn_events, cudnn_visits, cudnn_runtime


def read_cuda_api_events(cursor):
    """
    read_cuda_api_events function...
    """

    unique_cuda_api_events_runtime = None
    cuda_api_runtime_visits = None
    cuda_api_runtime_runtime = None
    unique_cuda_kernel_launches = None
    cuda_api_kernel_launches_runtime = None
    cuda_api_kernel_launches_visits = None
    cuda_api_kernel_launches_visits_sum = None
    cuda_api_kernel_launches_runtime_sum = None

    # read kernel launches
    query = """SELECT events.start, events.end, events.eventClass, events.globalTid, events.correlationId, events.returnValue, events.callchainId, (1.0 * events.start / 1000000000) AS start_seconds, (1.0 * events.end / 1000000000) AS end_seconds, (events.end - events.start) AS duration, (1.0 * (events.end - events.start) / 1000000000) AS duration_seconds,
strings2.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_RUNTIME AS events INNER JOIN 
StringIds As strings ON events.nameId = strings.Id INNER JOIN CUPTI_ACTIVITY_KIND_KERNEL 
AS kernel ON events.correlationId = kernel.correlationId INNER JOIN StringIds AS 
strings2 ON kernel.shortName = strings2.Id;"""
    result = get_data_from_db(cursor, query)

    if result is not None:

        if len(result) != 0:
        
            cuda_kernel_launches = convert_cupti_kernel_launch_to_objects(result)

            # get a list of unique CUDA API kernel launch events by their name
            query = """SELECT DISTINCT strings2.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_RUNTIME AS events INNER JOIN 
    StringIds As strings ON events.nameId = strings.Id INNER JOIN CUPTI_ACTIVITY_KIND_KERNEL 
    AS kernel ON events.correlationId = kernel.correlationId INNER JOIN StringIds AS 
    strings2 ON kernel.shortName = strings2.Id;"""
            cuda_kernel_launches_names = get_data_from_db(cursor, query)

            unique_cuda_kernel_launches = []
            # get the unique cuda api kernel launch events into a list to get rid of the tuple
            for i in range(len(cuda_kernel_launches_names)):
                unique_cuda_kernel_launches.append(cuda_kernel_launches_names[i][0])

            # sum up the duration of each unique cuda api kernel launch event in seconds
            cuda_api_kernel_launches_runtime = {}
            cuda_api_kernel_launches_runtime_sum = 0.0
            for i in range(len(cuda_kernel_launches_names)):
                cuda_api_kernel_launches_runtime[cuda_kernel_launches_names[i][0]] = 0.0
            for i in range(len(cuda_kernel_launches)):
                cuda_api_kernel_launches_runtime[cuda_kernel_launches[i].kernel_name] += cuda_kernel_launches[i].duration_seconds
                cuda_api_kernel_launches_runtime_sum += cuda_kernel_launches[i].duration_seconds

            # sum up the number of visits of each unique cuda api kernel launch event
            cuda_api_kernel_launches_visits = {}
            cuda_api_kernel_launches_visits_sum = 0
            for i in range(len(cuda_kernel_launches_names)):
                cuda_api_kernel_launches_visits[cuda_kernel_launches_names[i][0]] = 0
            for i in range(len(cuda_kernel_launches)):
                cuda_api_kernel_launches_visits[cuda_kernel_launches[i].kernel_name] += 1
                cuda_api_kernel_launches_visits_sum += 1

    
    query = "SELECT events.start, events.end, events.eventClass, events.globalTid, events.correlationId, strings.value, events.returnValue, events.callchainId, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_RUNTIME AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE strings.value NOT LIKE \"%LaunchKernel%\";"
    result = get_data_from_db(cursor, query)

    if result is not None:

        if len(result) != 0:

            cuda_api_events_runtime = convert_cupti_runtimes_to_objects(result)

            # get a list of unique CUDA API runtime events by their name
            query = "SELECT DISTINCT strings.value FROM CUPTI_ACTIVITY_KIND_RUNTIME AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id  WHERE strings.value NOT LIKE \"%LaunchKernel%\";"
            cuda_api_event_names_runtime = get_data_from_db(cursor, query)

            unique_cuda_api_events_runtime = []
            # get the unique cuda api runtime events into a list to get rid of the tuple
            for i in range(len(cuda_api_event_names_runtime)):
                unique_cuda_api_events_runtime.append(cuda_api_event_names_runtime[i][0])

            # sum up the duration of each unique cuda api runtime event in seconds
            cuda_api_runtime_runtime = {}
            for i in range(len(cuda_api_event_names_runtime)):
                cuda_api_runtime_runtime[cuda_api_event_names_runtime[i][0]] = 0.0
            for i in range(len(cuda_api_events_runtime)):
                cuda_api_runtime_runtime[cuda_api_events_runtime[i].name] += cuda_api_events_runtime[i].duration_seconds

            # sum up the number of visits of each unique cuda api runtime event
            cuda_api_runtime_visits = {}
            for i in range(len(cuda_api_event_names_runtime)):
                cuda_api_runtime_visits[cuda_api_event_names_runtime[i][0]] = 0
            for i in range(len(cuda_api_events_runtime)):
                cuda_api_runtime_visits[cuda_api_events_runtime[i].name] += 1

    #TODO: synchronization of CUDA API stuff
    # need to be selected from cuda activity kind sync, runtime, memset, memcopy tables together with joins
    # similar thing could be also done for memcopy and memset calls from cuda api driver calls...
    """
    query = "SELECT *, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION AS events;"
    result = get_data_from_db(cursor, query)

    if len(result) != 0:

        cuda_api_events_sync = convert_cupti_synchronize_to_objects(result)

        # get a list of unique CUDA API sync events by their name
        cuda_api_event_names_sync = []
        for i in range(len(cuda_api_events_sync)):
            if i == 0:
                cuda_api_event_names_sync.append(cuda_api_events_sync[i].sync_type)
            else:
                exists = False
                for j in range(len(cuda_api_event_names_sync)):
                    if cuda_api_events_sync[i].sync_type == cuda_api_event_names_sync[j]:
                        exists = True
                        break
                if exists == False:
                    cuda_api_event_names_sync.append(cuda_api_events_sync[i].sync_type)

        # sum up the duration of each unique cuda api sync event in seconds
        cuda_api_sync_runtime = {}
        for i in range(len(cuda_api_event_names_sync)):
            cuda_api_sync_runtime[cuda_api_event_names_sync[i]] = 0.0
        for i in range(len(cuda_api_events_sync)):
            cuda_api_sync_runtime[cuda_api_events_sync[i].sync_type] += cuda_api_events_sync[i].duration_seconds

        # sum up the number of visits of each unique cuda api sync event
        cuda_api_sync_visits = {}
        for i in range(len(cuda_api_event_names_sync)):
            cuda_api_sync_visits[cuda_api_event_names_sync[i]] = 0
        for i in range(len(cuda_api_events_sync)):
            cuda_api_sync_visits[cuda_api_events_sync[i].sync_type] += 1
    """

    return unique_cuda_api_events_runtime, cuda_api_runtime_visits, cuda_api_runtime_runtime, unique_cuda_kernel_launches, cuda_api_kernel_launches_runtime, cuda_api_kernel_launches_visits, cuda_api_kernel_launches_visits_sum, cuda_api_kernel_launches_runtime_sum


def read_os_events(cursor):
    """
    read_os_events function...
    """
    
    query = "SELECT events.start, events.end, events.eventClass, events.globalTid, events.correlationId, strings.value, events.returnValue, events.nestingLevel, events.callchainId, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM OSRT_API AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id;"
    result = get_data_from_db(cursor, query)

    unique_os_events = None
    os_visits = None
    os_runtime = None

    if result is not None:

        if len(result) != 0:

            os_events = convert_os_events_to_objects(result)

            # get a list of unique os events by their name
            query = "SELECT DISTINCT strings.value FROM OSRT_API AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id;"
            os_event_names = get_data_from_db(cursor, query)

            unique_os_events = []
            # get the unique os event into a list to get rid of the tuple
            for i in range(len(os_event_names)):
                unique_os_events.append(os_event_names[i][0])

            # sum up the duration of each unique os event in seconds
            os_runtime = {}
            for i in range(len(os_event_names)):
                os_runtime[os_event_names[i][0]] = 0.0
            for i in range(len(os_events)):
                os_runtime[os_events[i].name] += os_events[i].duration_seconds

            # sum up the number of visits of each unique os event
            os_visits = {}
            for i in range(len(os_event_names)):
                os_visits[os_event_names[i][0]] = 0
            for i in range(len(os_events)):
                os_visits[os_events[i].name] += 1

    return unique_os_events, os_visits, os_runtime


def read_memory_events(cursor):
    """
    read_memory_events function...
    """
    
    memset_visits = None
    memset_runtime = None
    memcopy_visits = None
    memcopy_runtime = None
    memset_bytes = None
    memcopy_bytes = None
    memcopy_htod_visits = None
    memcopy_dtoh_visits = None
    memcopy_dtod_visits = None
    memcopy_htoh_visits = None
    memcopy_htod_runtime = None
    memcopy_dtoh_runtime = None
    memcopy_dtod_runtime = None
    memcopy_htoh_runtime = None
    memcopy_htod_bytes = None
    memcopy_dtoh_bytes = None
    memcopy_dtod_bytes = None
    memcopy_htoh_bytes = None

    # memset operations
    query = "SELECT start, end, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, deviceId, contextId, streamId, correlationId, globalPid, value, bytes, graphNodeId, memKind FROM CUPTI_ACTIVITY_KIND_MEMSET;"
    result = get_data_from_db(cursor, query)

    if result is not None:

        if len(result) != 0:

            memset_events = convert_memset_events_to_objects(result)

            # sum up the duration of each unique memset event in seconds
            memset_runtime = 0.0
            for i in range(len(memset_events)):
                memset_runtime += memset_events[i].duration_seconds

            # sum up the number of visits of each unique memset event
            memset_visits = 0.0
            for i in range(len(memset_events)):
                memset_visits += 1

            # sum up the number of bytes set
            memset_bytes = 0.0
            for i in range(len(memset_events)):
                memset_bytes += memset_events[i].bytes

    # memcopy operations
    query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY;"
    result = get_data_from_db(cursor, query)

    if result is not None:

        if len(result) != 0:

            memcopy_events = convert_memcopy_events_to_objects(result)

            # sum up the duration of each unique memcopy event in seconds
            memcopy_runtime = 0.0
            for i in range(len(memcopy_events)):
                memcopy_runtime += memcopy_events[i].duration_seconds

            # sum up the number of visits of each unique memcopy event
            memcopy_visits = 0.0
            for i in range(len(memcopy_events)):
                memcopy_visits += 1

            # sum up the number of bytes transferred
            memcopy_bytes = 0.0
            for i in range(len(memcopy_events)):
                memcopy_bytes += memcopy_events[i].bytes

            # sum up the number of metrics for different types of memcopy operations
            memcopy_htod_visits = 0
            memcopy_dtoh_visits = 0
            memcopy_dtod_visits = 0
            memcopy_htoh_visits = 0
            memcopy_htod_runtime = 0
            memcopy_dtoh_runtime = 0
            memcopy_dtod_runtime = 0
            memcopy_htoh_runtime = 0
            memcopy_htod_bytes = 0
            memcopy_dtoh_bytes = 0
            memcopy_dtod_bytes = 0
            memcopy_htoh_bytes = 0
            for i in range(len(memcopy_events)):

                #htod
                if memcopy_events[i].copy_kind == "A host to device memory copy.":
                    memcopy_htod_visits += 1
                    memcopy_htod_runtime += memcopy_events[i].duration_seconds
                    memcopy_htod_bytes += memcopy_events[i].bytes
                #dtoh
                if memcopy_events[i].copy_kind == "A device to host memory copy.":
                    memcopy_dtoh_visits += 1
                    memcopy_dtoh_runtime += memcopy_events[i].duration_seconds
                    memcopy_dtoh_bytes += memcopy_events[i].bytes
                #dtod
                if memcopy_events[i].copy_kind == "A device to device memory copy on the same device.":
                    memcopy_dtod_visits += 1
                    memcopy_dtod_runtime += memcopy_events[i].duration_seconds
                    memcopy_dtod_bytes += memcopy_events[i].bytes
                #htoh
                if memcopy_events[i].copy_kind == "A host to host memory copy.":
                    memcopy_htoh_visits += 1
                    memcopy_htoh_runtime += memcopy_events[i].duration_seconds
                    memcopy_htoh_bytes += memcopy_events[i].bytes

    return memset_visits, memset_runtime, memcopy_visits, memcopy_runtime, memset_bytes, memcopy_bytes, memcopy_htod_visits, memcopy_dtoh_visits, memcopy_dtod_visits, memcopy_htoh_visits, memcopy_htod_runtime, memcopy_dtoh_runtime, memcopy_dtod_runtime, memcopy_htoh_runtime, memcopy_htod_bytes, memcopy_dtoh_bytes, memcopy_dtod_bytes, memcopy_htoh_bytes


def read_mpi_events(cursor):
    """
    read_mpi_events funciton ...
    """

    query = "SELECT events.start, events.end, strings.value, (events.end - events.start) AS duration, (1.0 * events.start / 1000000000) AS start_seconds, (1.0 * events.end / 1000000000) AS end_seconds, (1.0 * (events.end - events.start) / 1000000000) AS duration_seconds FROM NVTX_EVENTS AS events INNER JOIN StringIds AS strings ON events.textId = strings.id WHERE strings.value LIKE \"%MPI%\";"
    result = get_data_from_db(cursor, query)

    unique_mpi_events = None
    mpi_visits = None
    mpi_runtime = None

    if result is not None:

        if len(result) != 0:

            mpi_events = convert_mpi_events_to_objects(result)

            # get a list of unique MPI events by their name
            query = "SELECT Distinct strings.value FROM NVTX_EVENTS AS events INNER JOIN StringIds AS strings ON events.textId = strings.id WHERE strings.value LIKE \"%MPI%\";"
            mpi_event_names = get_data_from_db(cursor, query)

            unique_mpi_events = []
            # get the unique mpi_events into a list to get rid of the tuple
            for i in range(len(mpi_event_names)):
                unique_mpi_events.append(mpi_event_names[i][0])

            # sum up the duration of each unique mpi event in seconds
            mpi_runtime = {}
            for i in range(len(mpi_event_names)):
                mpi_runtime[mpi_event_names[i][0]] = 0.0
            for i in range(len(mpi_events)):
                mpi_runtime[mpi_events[i].name] += mpi_events[i].duration_seconds

            # sum up the number of visits of each unique cupti kernel
            mpi_visits = {}
            for i in range(len(mpi_event_names)):
                mpi_visits[mpi_event_names[i][0]] = 0
            for i in range(len(mpi_events)):
                mpi_visits[mpi_events[i].name] += 1

    return unique_mpi_events, mpi_visits, mpi_runtime


def read_cuda_kernel(cursor):
    """
    read_cuda_kernel funciton ...
    """

    query = "SELECT kernel.start, kernel.end, (end - start) AS duration, (1.0 * kernel.start / 1000000000) AS start_seconds, (1.0 * kernel.end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, strings.value AS demangledName, strings2.value AS shortName, ('(' || gridX || ',' || gridY || ',' || gridZ || ')') AS grid, ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block, staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted, registersPerThread, localMemoryTotal, localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel INNER JOIN StringIds AS strings ON kernel.demangledName = strings.id INNER JOIN StringIds AS strings2 ON kernel.shortName = strings2.id;"
    query_result = get_data_from_db(cursor, query)

    unique_cupti_events = None
    visits_dict = None
    runtime_sum_dict = None

    if query_result is not None:

        if len(query_result) != 0:
            
            cupti_kernels = convert_cupti_kernels_to_objects(query_result)

            # get the unique cupti kernel events in this time interval
            query = "SELECT Distinct strings.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel LEFT JOIN StringIds AS strings ON kernel.demangledName = strings.id LEFT JOIN StringIds AS strings2 ON kernel.shortName = strings2.id;"
            # get only short name for kernel launch use the other for fullname with strings.value
            #query = "SELECT Distinct strings2.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel LEFT JOIN StringIds AS strings ON kernel.demangledName = strings.id LEFT JOIN StringIds AS strings2 ON kernel.shortName = strings2.id;"
            cupti_events = get_data_from_db(cursor, query)

            unique_cupti_events = []
            # get the unique cupti events into a list to get rid of the tuple
            for i in range(len(cupti_events)):
                unique_cupti_events.append(cupti_events[i][0])

            # sum up the duration of each unique cupti kernel in seconds
            #TODO: should use not demangled Name here if I want to use short name...
            runtime_sum_dict = {}
            for i in range(len(cupti_events)):
                runtime_sum_dict[cupti_events[i][0]] = 0.0
            for i in range(len(cupti_kernels)):
                #TODO: should use not demangled Name here if I want to use short name...
                runtime_sum_dict[cupti_kernels[i].demangledName] += cupti_kernels[i].duration_seconds

            # sum up the number of visits of each unique cupti kernel
            visits_dict = {}
            for i in range(len(cupti_events)):
                visits_dict[cupti_events[i][0]] = 0
            for i in range(len(cupti_kernels)):
                #TODO: should use not demangled Name here if I want to use short name...
                visits_dict[cupti_kernels[i].demangledName] += 1

    return unique_cupti_events, visits_dict, runtime_sum_dict


def read_nvtx_epochs(cursor):
    """
    read_nvtx_epochs function 
    """

    # read the nvtx category data first
    query = "SELECT category, text FROM NVTX_EVENTS WHERE eventType = 33 AND text IS NOT NULL;"
    query_result = get_data_from_db(cursor, query)
    
    #epoch_category_exist = None
    nvtx_epochs = None

    if query_result is not None:

        if len(query_result) != 0:
            #epoch_category_exist == False

            # put them into a dict
            categories = {}
            for x in query_result:
                categories[x[0]] = str(x[1])

            # find the training step category
            key = [k for k, v in categories.items() if v == "epoch"]

            if len(key) == 0:
                epoch_category_exist = False
            
            else:
                key = key[0]
                # get all the nvtx epoch events
                query = "SELECT start, end, text, color, textId, domainId, eventType, rangeId, category, globalTid FROM NVTX_EVENTS WHERE eventType = 59 AND category = "+str(key)+" AND text IS NOT NULL;"
                query_result = get_data_from_db(cursor, query)

                if query_result is not None:

                    if len(query_result) != 0:

                        nvtx_epochs = convert_nvtx_events_to_objects(query_result)

    return nvtx_epochs


def read_app_phases(cursor):
    """
    read_app_phases function ...
    """

    # get the training step timestamps first
    nvtx_training_steps = read_nvtx_training_steps(cursor)

    # get the epoch timestamps
    nvtx_epochs = read_nvtx_epochs(cursor)

    # filter the training steps with high variatio
    filtered_steps = None
    if nvtx_training_steps != None:
        runtimes = []
        for i in range(len(nvtx_training_steps)):
            runtime = nvtx_training_steps[i].run_time_seconds
            runtimes.append(runtime)
        # remove outliers, should remove first epoch with 
        # high initialization and optimization overhead and others
        data_ids = get_outliers(runtimes)
        filtered_steps = []
        for i in range(len(data_ids)):
            filtered_steps.append(nvtx_training_steps[data_ids[i]])

    # filter the training steps with high variatio
    filtered_epochs = None
    if nvtx_epochs != None:
        runtimes = []
        for i in range(len(nvtx_epochs)):
            runtime = nvtx_epochs[i].run_time_seconds
            runtimes.append(runtime)
        # remove outliers, should remove first epoch with 
        # high initialization and optimization overhead and others
        data_ids = get_outliers(runtimes)
        filtered_epochs = []
        for i in range(len(data_ids)):
            filtered_epochs.append(nvtx_epochs[data_ids[i]])
    
    communication_time_steps = None
    memory_time_steps = None
    computation_time_steps = None

    # get the start and end times 
    if filtered_steps != None:
        starttimes = []
        endtimes = []
        for i in range(len(filtered_steps)):
            step = filtered_steps[i]
            starttimes.append(step.start_time_long)
            endtimes.append(step.end_time_long)

        computation_time = []
        communication_time = []
        memory_time = []
        memory_bytes = []

        # read the runtimes from all different events and aggregate them
        for i in range(len(starttimes)):
            start = starttimes[i]
            end = endtimes[i]

            time_mem = 0
            time_comp = 0
            time_com = 0
            bytes_mem = 0

            ######### START KERNELS #########

            query = "SELECT kernel.start, kernel.end, (end - start) AS duration, (1.0 * kernel.start / 1000000000) AS start_seconds, (1.0 * kernel.end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, strings.value AS demangledName, strings2.value AS shortName, ('(' || gridX || ',' || gridY || ',' || gridZ || ')') AS grid, ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block, staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted, registersPerThread, localMemoryTotal, localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel INNER JOIN StringIds AS strings ON kernel.demangledName = strings.id INNER JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            query_result = get_data_from_db(cursor, query)

            if query_result is not None:

                if len(query_result) != 0:
                    
                    cupti_kernels = convert_cupti_kernels_to_objects(query_result)

                    # get the unique cupti kernel events in this time interval
                    query = "SELECT Distinct strings.value, strings2.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel LEFT JOIN StringIds AS strings ON kernel.demangledName = strings.id LEFT JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE kernel.start >="+str(start)+" AND kernel.end <="+str(end)+";"
                    
                    # sum up the duration of each unique cupti kernel in seconds
                    runtime_sum_kernels = 0
                    for i in range(len(cupti_kernels)):
                        if cupti_kernels[i].shortName.find("nccl") != -1 or cupti_kernels[i].demangledName.find("NCCL") != -1:
                            time_com += cupti_kernels[i].duration_seconds
                        else:
                            runtime_sum_kernels += cupti_kernels[i].duration_seconds

                    #print("Sum runtime kernels:",runtime_sum_kernels)

                    time_comp += runtime_sum_kernels

            ######### END KERNELS #########

            ######### START MEMSET #########

            # memset operations
            query = "SELECT start, end, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, deviceId, contextId, streamId, correlationId, globalPid, value, bytes, graphNodeId, memKind FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    memset_events = convert_memset_events_to_objects(result)
                    #print(memset_events)
                    #print(len(memset_events))

                    # sum up the duration of each memset events in seconds
                    runtime_sum_memsets = 0
                    bytes_sum_memsets = 0
                    for i in range(len(memset_events)):
                        runtime_sum_memsets += memset_events[i].duration_seconds
                        bytes_sum_memsets += memset_events[i].bytes

                    #print("Sum runtime memsets:",runtime_sum_memsets)
                    #print("Sum bytes memsets:",bytes_sum_memsets)

                    time_mem += runtime_sum_memsets
                    bytes_mem += bytes_sum_memsets

            ######### END MEMSET #########

            ######### START MEMCOPY #########

            # memcopy operations
            query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    memcopy_events = convert_memcopy_events_to_objects(result)

                    #print(memcopy_events)
                    #print(len(memcopy_events))

                    # sum up the duration of each memcopy events in seconds
                    runtime_sum_memcopies = 0
                    bytes_sum_memcopies = 0
                    for i in range(len(memcopy_events)):
                        runtime_sum_memcopies += memcopy_events[i].duration_seconds
                        bytes_sum_memcopies += memcopy_events[i].bytes

                    #print("Sum runtime memcopies:",runtime_sum_memcopies)
                    #print("Sum bytes memcopies:",bytes_sum_memcopies)

                    time_mem += runtime_sum_memcopies
                    bytes_mem += bytes_sum_memcopies

            ######### END MEMCOPY #########

            ######### START CUBLAS #########

            query = "SELECT cublas.start, cublas.end, cublas.eventClass, cublas.globalTid, strings.value, (cublas.end - cublas.start) AS duration, (1.0 * cublas.start / 1000000000) AS start_seconds, (1.0 * cublas.end / 1000000000) AS end_seconds, (1.0 * (cublas.end - cublas.start) / 1000000000) AS duration_seconds FROM CUBLAS_EVENTS AS cublas INNER JOIN StringIds AS strings ON cublas.nameId = strings.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    cublas_events = convert_cublas_events_to_objects(result)

                    #print(cublas_events)
                    #print(len(cublas_events))

                    # sum up the duration of each cublas event in seconds
                    runtime_sum_cublas = 0
                    for i in range(len(cublas_events)):
                        runtime_sum_cublas += cublas_events[i].duration_seconds
                        #print(cublas_events[i].name)

                    #print("Sum runtime cublas events:",runtime_sum_cublas)

                    time_comp += runtime_sum_cublas

            ######### END CUBLAS #########

            ######### START CUDNN #########

            query = "SELECT events.start, events.end, events.eventClass, events.globalTid, strings.value, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUDNN_EVENTS AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    cudnn_events = convert_cudnn_events_to_objects(result)

                    #print(cudnn_events)
                    #print(len(cudnn_events))

                    # sum up the duration of each cublas event in seconds
                    runtime_sum_cudnn = 0
                    for i in range(len(cudnn_events)):
                        runtime_sum_cudnn += cudnn_events[i].duration_seconds

                    #print("Sum runtime cudnn events:",runtime_sum_cudnn)

                    time_comp += runtime_sum_cudnn

            ######### END CUDNN #########

            ######### START CUPTI SYNCHRONIZE #########

            query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start >="+str(start)+" AND end <="+str(end)+";"
            list = get_data_from_db(cursor, query)

            if list != None:

                cupti_synch_events = convert_cupti_synchronize_to_objects(list)

                #print(cupti_synch_events)
                #print(len(cupti_synch_events))

                # sum up the duration of each cupti synch event in seconds
                runtime_sum_cupti_synch = 0
                for i in range(len(cupti_synch_events)):
                    runtime_sum_cupti_synch += cupti_synch_events[i].duration_seconds

                #print("Sum runtime cupti synchronization events:",runtime_sum_cupti_synch)

                time_comp += runtime_sum_cupti_synch

            ######### END CUPTI SYNCHRONIZE #########

            ######### START CUPTI RUNTIME #########

            query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >="+str(start)+" AND end <="+str(end)+";"
            list = get_data_from_db(cursor, query)

            if list != None:

                cupti_runtime_events = convert_cupti_runtimes_to_objects(list)

                #print(cupti_runtime_events)
                #print(len(cupti_runtime_events))

                # sum up the duration of each cupti synch event in seconds
                runtime_sum_cupti_runtime = 0
                for i in range(len(cupti_runtime_events)):
                    runtime_sum_cupti_runtime += cupti_runtime_events[i].duration_seconds

                #print("Sum runtime cupti runtime events:",runtime_sum_cupti_runtime)

                time_comp += runtime_sum_cupti_runtime

            ######### END CUPTI RUNTIME #########

            ######### START OS RUNTIME #########

            query = "SELECT events.start, events.end, events.eventClass, events.globalTid, events.correlationId, strings.value, events.returnValue, events.nestingLevel, events.callchainId, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM OSRT_API AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    os_events = convert_os_events_to_objects(result)

                    # sum up the duration of each os event in seconds
                    runtime_sum_os_runtime = 0
                    for i in range(len(os_events)):
                        runtime_sum_os_runtime += os_events[i].duration_seconds

                    #print("Sum runtime os runtime events:",runtime_sum_os_runtime)

                    time_comp += runtime_sum_os_runtime

            ######### END OS RUNTIME #########

            ######### START MPI RUNTIME #########

            # this finds all mpi events within one training step

            query = "SELECT events.start, events.end, strings.value, (events.end - events.start) AS duration, (1.0 * events.start / 1000000000) AS start_seconds, (1.0 * events.end / 1000000000) AS end_seconds, (1.0 * (events.end - events.start) / 1000000000) AS duration_seconds FROM NVTX_EVENTS AS events INNER JOIN StringIds AS strings ON events.textId = strings.id WHERE strings.value LIKE \"%MPI%\" AND start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    mpi_events = convert_mpi_events_to_objects(result)

                    runtime_sum_mpi = 0
                    for j in range(len(mpi_events)):
                        runtime_sum_mpi += mpi_events[j].duration_seconds

                    time_com += runtime_sum_mpi

            ######### END MPI #########

            memory_time.append(time_mem)
            computation_time.append(time_comp)
            communication_time.append(time_com)
            memory_bytes.append(bytes_mem)

        comtime = remove_outliers(communication_time)
        memtime = remove_outliers(memory_time)
        comptime = remove_outliers(computation_time)
        membytes = remove_outliers(memory_bytes)

        communication_time_steps = np.mean(comtime)
        memory_time_steps = np.mean(memtime)
        computation_time_steps = np.mean(comptime)
        memory_bytes_steps = np.mean(membytes)

    communication_time_epochs = None
    memory_time_epochs = None
    computation_time_epochs = None

    # get the start and end times 
    if filtered_epochs != None:
        starttimes = []
        endtimes = []
        for i in range(len(filtered_epochs)):
            epoch = filtered_epochs[i]
            starttimes.append(epoch.start_time_long)
            endtimes.append(epoch.end_time_long)

        computation_time = []
        communication_time = []
        memory_time = []
        memory_bytes = []

        # read the runtimes from all different events and aggregate them
        for i in range(len(starttimes)):
            start = starttimes[i]
            end = endtimes[i]

            time_mem = 0
            time_comp = 0
            time_com = 0
            bytes_mem = 0

            ######### START KERNELS #########

            query = "SELECT kernel.start, kernel.end, (end - start) AS duration, (1.0 * kernel.start / 1000000000) AS start_seconds, (1.0 * kernel.end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, strings.value AS demangledName, strings2.value AS shortName, ('(' || gridX || ',' || gridY || ',' || gridZ || ')') AS grid, ('(' || blockX || ',' || blockY || ',' || blockZ || ')') AS block, staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted, registersPerThread, localMemoryTotal, localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel INNER JOIN StringIds AS strings ON kernel.demangledName = strings.id INNER JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            query_result = get_data_from_db(cursor, query)

            if query_result is not None:

                if len(query_result) != 0:
                    
                    cupti_kernels = convert_cupti_kernels_to_objects(query_result)

                    # get the unique cupti kernel events in this time interval
                    query = "SELECT Distinct strings.value, strings2.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel LEFT JOIN StringIds AS strings ON kernel.demangledName = strings.id LEFT JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE kernel.start >="+str(start)+" AND kernel.end <="+str(end)+";"
                    
                    # sum up the duration of each unique cupti kernel in seconds
                    runtime_sum_kernels = 0
                    for i in range(len(cupti_kernels)):
                        if cupti_kernels[i].shortName.find("nccl") != -1 or cupti_kernels[i].demangledName.find("NCCL") != -1:
                            time_com += cupti_kernels[i].duration_seconds
                        else:
                            runtime_sum_kernels += cupti_kernels[i].duration_seconds

                    #print("Sum runtime kernels:",runtime_sum_kernels)

                    time_comp += runtime_sum_kernels

            ######### END KERNELS #########

            ######### START MEMSET #########

            # memset operations
            query = "SELECT start, end, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds, deviceId, contextId, streamId, correlationId, globalPid, value, bytes, graphNodeId, memKind FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    memset_events = convert_memset_events_to_objects(result)
                    #print(memset_events)
                    #print(len(memset_events))

                    # sum up the duration of each memset events in seconds
                    runtime_sum_memsets = 0
                    bytes_sum_memsets = 0
                    for i in range(len(memset_events)):
                        runtime_sum_memsets += memset_events[i].duration_seconds
                        bytes_sum_memsets += memset_events[i].bytes

                    #print("Sum runtime memsets:",runtime_sum_memsets)
                    #print("Sum bytes memsets:",bytes_sum_memsets)

                    time_mem += runtime_sum_memsets
                    bytes_mem += bytes_sum_memsets

            ######### END MEMSET #########

            ######### START MEMCOPY #########

            # memcopy operations
            query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    memcopy_events = convert_memcopy_events_to_objects(result)

                    #print(memcopy_events)
                    #print(len(memcopy_events))

                    # sum up the duration of each memcopy events in seconds
                    runtime_sum_memcopies = 0
                    bytes_sum_memcopies = 0
                    for i in range(len(memcopy_events)):
                        runtime_sum_memcopies += memcopy_events[i].duration_seconds
                        bytes_sum_memcopies += memcopy_events[i].bytes

                    #print("Sum runtime memcopies:",runtime_sum_memcopies)
                    #print("Sum bytes memcopies:",bytes_sum_memcopies)

                    time_mem += runtime_sum_memcopies
                    bytes_mem += bytes_sum_memcopies

            ######### END MEMCOPY #########

            ######### START CUBLAS #########

            query = "SELECT cublas.start, cublas.end, cublas.eventClass, cublas.globalTid, strings.value, (cublas.end - cublas.start) AS duration, (1.0 * cublas.start / 1000000000) AS start_seconds, (1.0 * cublas.end / 1000000000) AS end_seconds, (1.0 * (cublas.end - cublas.start) / 1000000000) AS duration_seconds FROM CUBLAS_EVENTS AS cublas INNER JOIN StringIds AS strings ON cublas.nameId = strings.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    cublas_events = convert_cublas_events_to_objects(result)

                    #print(cublas_events)
                    #print(len(cublas_events))

                    # sum up the duration of each cublas event in seconds
                    runtime_sum_cublas = 0
                    for i in range(len(cublas_events)):
                        runtime_sum_cublas += cublas_events[i].duration_seconds
                        #print(cublas_events[i].name)

                    #print("Sum runtime cublas events:",runtime_sum_cublas)

                    time_comp += runtime_sum_cublas

            ######### END CUBLAS #########

            ######### START CUDNN #########

            query = "SELECT events.start, events.end, events.eventClass, events.globalTid, strings.value, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUDNN_EVENTS AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    cudnn_events = convert_cudnn_events_to_objects(result)

                    #print(cudnn_events)
                    #print(len(cudnn_events))

                    # sum up the duration of each cublas event in seconds
                    runtime_sum_cudnn = 0
                    for i in range(len(cudnn_events)):
                        runtime_sum_cudnn += cudnn_events[i].duration_seconds

                    #print("Sum runtime cudnn events:",runtime_sum_cudnn)

                    time_comp += runtime_sum_cudnn

            ######### END CUDNN #########

            ######### START CUPTI SYNCHRONIZE #########

            query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start >="+str(start)+" AND end <="+str(end)+";"
            list = get_data_from_db(cursor, query)

            if list != None:

                cupti_synch_events = convert_cupti_synchronize_to_objects(list)

                #print(cupti_synch_events)
                #print(len(cupti_synch_events))

                # sum up the duration of each cupti synch event in seconds
                runtime_sum_cupti_synch = 0
                for i in range(len(cupti_synch_events)):
                    runtime_sum_cupti_synch += cupti_synch_events[i].duration_seconds

                #print("Sum runtime cupti synchronization events:",runtime_sum_cupti_synch)

                time_comp += runtime_sum_cupti_synch

            ######### END CUPTI SYNCHRONIZE #########

            ######### START CUPTI RUNTIME #########

            query = "SELECT *, (end - start) AS duration, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >="+str(start)+" AND end <="+str(end)+";"
            list = get_data_from_db(cursor, query)

            if list != None:

                cupti_runtime_events = convert_cupti_runtimes_to_objects(list)

                #print(cupti_runtime_events)
                #print(len(cupti_runtime_events))

                # sum up the duration of each cupti synch event in seconds
                runtime_sum_cupti_runtime = 0
                for i in range(len(cupti_runtime_events)):
                    runtime_sum_cupti_runtime += cupti_runtime_events[i].duration_seconds

                #print("Sum runtime cupti runtime events:",runtime_sum_cupti_runtime)

                time_comp += runtime_sum_cupti_runtime

            ######### END CUPTI RUNTIME #########

            ######### START OS RUNTIME #########

            query = "SELECT events.start, events.end, events.eventClass, events.globalTid, events.correlationId, strings.value, events.returnValue, events.nestingLevel, events.callchainId, (1.0 * start / 1000000000) AS start_seconds, (1.0 * end / 1000000000) AS end_seconds, (end - start) AS duration, (1.0 * (end - start) / 1000000000) AS duration_seconds FROM OSRT_API AS events INNER JOIN StringIds AS strings ON events.nameId = strings.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    os_events = convert_os_events_to_objects(result)

                    # sum up the duration of each os event in seconds
                    runtime_sum_os_runtime = 0
                    for i in range(len(os_events)):
                        runtime_sum_os_runtime += os_events[i].duration_seconds

                    #print("Sum runtime os runtime events:",runtime_sum_os_runtime)

                    time_comp += runtime_sum_os_runtime

            ######### END OS RUNTIME #########

            ######### START MPI RUNTIME #########

            # this finds all mpi events within one training step

            query = "SELECT events.start, events.end, strings.value, (events.end - events.start) AS duration, (1.0 * events.start / 1000000000) AS start_seconds, (1.0 * events.end / 1000000000) AS end_seconds, (1.0 * (events.end - events.start) / 1000000000) AS duration_seconds FROM NVTX_EVENTS AS events INNER JOIN StringIds AS strings ON events.textId = strings.id WHERE strings.value LIKE \"%MPI%\" AND start>="+str(start)+" AND end<="+str(end)+";"
            result = get_data_from_db(cursor, query)

            if result is not None:

                if len(result) != 0:

                    mpi_events = convert_mpi_events_to_objects(result)

                    runtime_sum_mpi = 0
                    for j in range(len(mpi_events)):
                        runtime_sum_mpi += mpi_events[j].duration_seconds

                    time_com += runtime_sum_mpi

            ######### END MPI #########

            memory_time.append(time_mem)
            computation_time.append(time_comp)
            communication_time.append(time_com)
            memory_bytes.append(bytes_mem)

        comtime = remove_outliers(communication_time)
        memtime = remove_outliers(memory_time)
        comptime = remove_outliers(computation_time)
        membytes = remove_outliers(memory_bytes)

        communication_time_epochs = np.mean(comtime)
        memory_time_epochs = np.mean(memtime)
        computation_time_epochs = np.mean(comptime)
        memory_bytes_epochs = np.mean(membytes)

    return communication_time_steps, memory_time_steps, computation_time_steps, communication_time_epochs, memory_time_epochs, computation_time_epochs
    
    
def read_nvtx_testing_steps(cursor):
    """
    read_nvtx_testing_steps function 
    """

    # read the nvtx category data first
    query = "SELECT category, text FROM NVTX_EVENTS WHERE eventType = 33 AND text IS NOT NULL;"
    query_result = get_data_from_db(cursor, query)
    
    nvtx_testing_steps = None

    if query_result is not None:

        if len(query_result) != 0:
        
            # put them into a dict
            categories = {}
            for x in query_result:
                categories[x[0]] = str(x[1])

            # find the testing step category
            key = [k for k, v in categories.items() if v == "testing_step"]

            if len(key) == 0:
                testing_category_exist = False
            
            else:
                key = key[0]
                # get all the nvtx testing step events
                query = "SELECT start, end, text, color, textId, domainId, eventType, rangeId, category, globalTid FROM NVTX_EVENTS WHERE eventType = 59 AND category = "+str(key)+" AND text IS NOT NULL;"
                query_result = get_data_from_db(cursor, query)

                if query_result is not None:

                    if len(query_result) != 0:

                        nvtx_testing_steps = convert_nvtx_events_to_objects(query_result)

    return nvtx_testing_steps


def read_nvtx_training_steps(cursor):
    """
    read_nvtx_training_steps function 
    """

    # read the nvtx category data first
    query = "SELECT category, text FROM NVTX_EVENTS WHERE eventType = 33 AND text IS NOT NULL;"
    query_result = get_data_from_db(cursor, query)
    
    #training_category_exist = None
    nvtx_training_steps = None

    if query_result is not None:

        if len(query_result) != 0:
            #training_category_exist == False

            # put them into a dict
            categories = {}
            for x in query_result:
                categories[x[0]] = str(x[1])

            # find the training step category
            key = [k for k, v in categories.items() if v == "training_step"]

            if len(key) == 0:
                training_category_exist = False
            
            else:
                key = key[0]
                # get all the nvtx training step events
                query = "SELECT start, end, text, color, textId, domainId, eventType, rangeId, category, globalTid FROM NVTX_EVENTS WHERE eventType = 59 AND category = "+str(key)+" AND text IS NOT NULL;"
                query_result = get_data_from_db(cursor, query)

                if query_result is not None:

                    if len(query_result) != 0:

                        nvtx_training_steps = convert_nvtx_events_to_objects(query_result)

    return nvtx_training_steps


def read_nvtx_user_instrumentation(cursor):

    nvtx_user_events_runtime = None
    nvtx_user_events_visits = None

    query = "SELECT start, end, text, color, textId, domainId, eventType, rangeId, category, globalTid FROM NVTX_EVENTS WHERE eventType = 59 AND text IS NOT NULL;"
    result = get_data_from_db(cursor, query)

    if result is not None:

        if len(result) != 0:

            nvtx_user_events = convert_nvtx_events_to_objects(result)

            nvtx_user_events_runtime = {}
            nvtx_user_events_visits = {}

            for count, event in enumerate(nvtx_user_events):
                if count == 0:
                    nvtx_user_events_runtime[event.callpath_name] = event.run_time_seconds
                    nvtx_user_events_visits[event.callpath_name] = 1
                else:
                    callpath_exists = None
                    for key in nvtx_user_events_runtime:
                        if key == event.callpath_name:
                            callpath_exists = key
                            break
                    if callpath_exists != None:
                        sum = nvtx_user_events_runtime[key] + event.run_time_seconds
                        nvtx_user_events_runtime[key] = sum
                        sum = nvtx_user_events_visits[key] + 1
                        nvtx_user_events_visits[key] = sum
                    else:
                        nvtx_user_events_runtime[event.callpath_name] = event.run_time_seconds
                        nvtx_user_events_visits[event.callpath_name] = 1
    
    return nvtx_user_events_runtime, nvtx_user_events_visits


def find_unique_experiment_configurations(configs):
    nr_experiment_configs = len(configs)
    unique_configs = []
    for i in range(len(configs)):
        if i==0:
            unique_configs.append(configs[i])
        else:
            exists = False
            for j in range(len(unique_configs)):
                if unique_configs[j].app_name == configs[i].app_name:
                    if unique_configs[j].parameter_names == configs[i].parameter_names:
                        if unique_configs[j].parameter_values == configs[i].parameter_values:
                            exists = True
                            break
            if exists == False:
                unique_configs.append(configs[i])
    return nr_experiment_configs, unique_configs


class Storage():
    def __init__(self, config_object):
        self.config_object = config_object
        self.repetitions = []
        self.mpi_ranks = []


def create_storage_objects_for_exp_configs(unique_configs):
    # create a storage object for each unique experiment config
    storages = []
    for i in range(len(unique_configs)):
        storages.append(Storage(unique_configs[i]))
    return storages


def find_reps_for_unique_exps(storages, configs):
    for i in range(len(storages)):
        config = storages[i].config_object
        for j in range(len(configs)):
            check = configs[j]
            if config.app_name == check.app_name:
                if config.parameter_names == check.parameter_names:
                    if config.parameter_values == check.parameter_values:
                        if check.repetition_nr not in storages[i].repetitions:
                            storages[i].repetitions.append(check.repetition_nr)
                        if check.mpi_rank not in storages[i].mpi_ranks:
                            storages[i].mpi_ranks.append(check.mpi_rank)


def concat_unique_kernel_list(callpaths_nvtx):
    concat_kernel_list_nvtx = []
    for i in range(len(callpaths_nvtx)):
        for j in range(len(callpaths_nvtx[i])):
            if i == 0:
                concat_kernel_list_nvtx.append(callpaths_nvtx[i][j])
            else:
                exists = False
                for k in range(len(concat_kernel_list_nvtx)):
                    if callpaths_nvtx[i][j] == concat_kernel_list_nvtx[k]:
                        exists = True
                        break
                if exists == False:
                     concat_kernel_list_nvtx.append(callpaths_nvtx[i][j])
    return concat_kernel_list_nvtx         


def get_experiment_config_ids(storages, configs):
    experiment_config_ids = []
    temp_ids = []
    for i in range(len(storages)):
        temp = []
        reps = storages[i].repetitions
        config = storages[i].config_object
        temp_reps = []
        for j in range(len(reps)):
            rep = reps[j]
            #print("rep:",rep)
            ranks = storages[i].mpi_ranks
            #print(ranks)
            temp_ranks = []
            for k in range(len(ranks)):
                rank = ranks[k]
                #print("rank:",rank)
                #print(configs)
                for l in range(len(configs)):
                    check = configs[l]
                    if config.app_name == check.app_name:
                        if config.parameter_names == check.parameter_names:
                            if config.parameter_values == check.parameter_values:
                                if rep == check.repetition_nr:
                                    if rank == check.mpi_rank:
                                        temp.append(check.id)
                                        temp_ranks.append(check.id)
            temp_reps.append(temp_ranks)
        experiment_config_ids.append(temp)
        temp_ids.append(temp_reps)  
    return temp_ids 


def create_metric_dict(experiment_config_ids, concat_kernel_list_nvtx):
    # create new free metric dict for any metric
    measurements = []
    for _ in range(len(experiment_config_ids)):
        temp = {}
        for j in range(len(concat_kernel_list_nvtx)):
            temp[concat_kernel_list_nvtx[j]] = 0
        measurements.append(temp)
    return measurements


def populate_measurement_callpath(concat_kernel_list, experiment_config_ids, visits, runtime, measurement_visits, measurement_runtime):

    # calculate median for all visits

    # go through the list of unique kernels
    for k in range(len(concat_kernel_list)):
        # go through all unique experiment configurations
        for i in range(len(experiment_config_ids)):
            rep_values = []
            for l in range(len(experiment_config_ids[i])):
                # calculate mean value for this kernel over all ranks
                rank_values = []
                # go through the mpi ranks
                for j in range(len(experiment_config_ids[i][l])):
                    config_id = experiment_config_ids[i][l][j]
                    try:
                        value = visits[config_id][concat_kernel_list[k]]
                        rank_values.append(value)
                    except KeyError:
                        pass
                #print("rank_values:",rank_values)
                if len(rank_values) != 0:
                    all_ranks_mean_value = np.mean(rank_values)
                    rep_values.append(all_ranks_mean_value)
            #print("y:",rep_values)
            measurement_visits[i][concat_kernel_list[k]] = rep_values

    # remove callpaths with empty value arrays
    for i in range(len(measurement_visits)):
        for j in range(len(concat_kernel_list)):
            if len(measurement_visits[i][concat_kernel_list[j]]) == 0:
                measurement_visits[i].pop(concat_kernel_list[j])

    # create a new list of callpaths that exist in all experiment conifgurations
    callpath_list = []
    for j in range(len(concat_kernel_list)):
        exists_in_all = True
        for i in range(len(measurement_visits)):
            try:
                measurement_visits[i][concat_kernel_list[j]]
            except KeyError:
                exists_in_all = False
                break
        if exists_in_all == True:
            callpath_list.append(concat_kernel_list[j])

    # calculate median for all runtime median
    # go through the list of unique kernels
    for k in range(len(concat_kernel_list)):
        # go through all unique experiment configurations
        for i in range(len(experiment_config_ids)):
            rep_values = []
            for l in range(len(experiment_config_ids[i])):
                # calculate mean value for this kernel over all ranks
                rank_values = []
                # go through the mpi ranks
                for j in range(len(experiment_config_ids[i][l])):
                    config_id = experiment_config_ids[i][l][j]
                    try:
                        value = runtime[config_id][concat_kernel_list[k]]
                        rank_values.append(value)
                    except KeyError:
                        pass
                if len(rank_values) != 0:
                    all_ranks_mean_value = np.mean(rank_values)
                    rep_values.append(all_ranks_mean_value)
            #if concat_kernel_list[k] == "main->train->training_step":
                #print("y:",rep_values)
            #print("y:",rep_values)
            measurement_runtime[i][concat_kernel_list[k]] = rep_values

    # remove callpaths with empty value arrays
    delete_this = []
    for i in range(len(measurement_runtime)):
        for j in range(len(concat_kernel_list)):
            if len(measurement_runtime[i][concat_kernel_list[j]]) == 0:
                measurement_runtime[i].pop(concat_kernel_list[j])
                delete_this.append(concat_kernel_list[j])

    return callpath_list, measurement_visits, measurement_runtime


def create_measurement_list(experiment_config_ids, metric_values):
    measurement_list = []
    for i in range(len(experiment_config_ids)):
        temp = {}
        measurement_list.append(temp)
    for i in range(len(experiment_config_ids)):
        rep_values = []
        for l in range(len(experiment_config_ids[i])):
            rank_values = []
            for j in range(len(experiment_config_ids[i][l])):
                config_id = experiment_config_ids[i][l][j]
                try:
                    value = metric_values[config_id]
                    if value != {}:
                        rank_values.append(value)
                except KeyError:
                    pass
            if len(rank_values) != 0:
                all_ranks_mean_value = np.mean(rank_values)
                rep_values.append(all_ranks_mean_value)
        measurement_list[i] = rep_values
    return measurement_list


def add_measurement_list_to_experiment(extrap_experiment, unique_configs, callpath, metric, current_analysis_type, measurement_list):
    for l in range(len(measurement_list)):
        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        if len(measurement_list[l]) > 0:
            extrap_experiment.add_coordinate(extrap_coordinate)
            if extrap_experiment.scaling == "strong":
                # get the number of mpi ranks
                # this assumes that the mpi rank is the parameter with index 0
                mpi_ranks = float(unique_configs[l].parameter_values[0])
                # multiply the runtime values with the number of mpi ranks
                strong_scaling_values = measurement_list[l]
                for k in range(len(strong_scaling_values)):
                    strong_scaling_values[k] = strong_scaling_values[k] * mpi_ranks
            if extrap_experiment.scaling == "strong":
                extrap_measurement = Measurement(extrap_coordinate, callpath, metric, current_analysis_type, strong_scaling_values)
                extrap_experiment.add_measurement(extrap_measurement)
            else:
                extrap_measurement = Measurement(extrap_coordinate, callpath, metric, current_analysis_type, measurement_list[l])
                extrap_experiment.add_measurement(extrap_measurement)
    return extrap_experiment


def add_measurement_dict_to_experiment(extrap_experiment, unique_configs, current_analysis_type, metric, measurement_dict, callpath_list):
    for l in range(len(measurement_dict)):

        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        extrap_experiment.add_coordinate(extrap_coordinate)

        for i in range(len(callpath_list)):

            # get the callpath from the experiment
            callpaths = extrap_experiment.callpaths[current_analysis_type]
            id = -1
            for j in range(len(callpaths)):
                callpath = callpaths[j]
                if callpath.name == callpath_list[i]:
                    id = j
            callpath = callpaths[id]

            if extrap_experiment.scaling == "strong":
                # get the number of mpi ranks
                # this assumes that the mpi rank is the parameter with index 0
                mpi_ranks = float(unique_configs[l].parameter_values[0])
            
                # multiply the runtime values with the number of mpi ranks
                strong_scaling_values = measurement_dict[l][callpath_list[i]]
                for k in range(len(strong_scaling_values)):
                    strong_scaling_values[k] = strong_scaling_values[k] * mpi_ranks

            if extrap_experiment.scaling == "strong":
                extrap_measurement = Measurement(extrap_coordinate, callpath, metric, current_analysis_type, strong_scaling_values)
                extrap_experiment.add_measurement(extrap_measurement)
            else:
                extrap_measurement = Measurement(extrap_coordinate, callpath, metric, current_analysis_type, measurement_dict[l][callpath_list[i]])
                extrap_experiment.add_measurement(extrap_measurement)
    return extrap_experiment


def remove_measurements_callpaths(extrap_experiment, current_analysis_type, arguments):
    # check if there are enough data points for extra-p to model for all callpath and experiment configurations
    # find the visit metric
    metrics = extrap_experiment.metrics[current_analysis_type]
    metric_id = 0
    for i in range(len(metrics)):
        if metrics[i].name == "visits":
            metric_id = i
            break

    callpaths = extrap_experiment.callpaths[current_analysis_type]
    temp = []
    for i in range(len(callpaths)):
        counter = 0
        for j in range(len(extrap_experiment.coordinates)):
            # only need to check for one metric, if the kernel has no visit it also can't have runtime
            if extrap_experiment.get_measurement(j, i, metric_id, current_analysis_type) != None:
                counter += 1
        min_points = 5
        if arguments.minimum_required_points != None:
            min_points = int(arguments.minimum_required_points)
        if counter < min_points:
            logging.warning("Not enough data points for the callpath:"+str(callpaths[i])+". Found "+str(counter)+" point(s) need at least "+str(5)+" points. Deleting this callpath from the experiment.")
            temp.append(callpaths[i].name)

    if len(temp) > 0:
        logging.warning("Skipping "+str(len(temp))+" callpaths.")

    callpath_id_del = []
    for i in range(len(temp)):
        for j in range(len(callpaths)):
            if callpaths[j].name == temp[i]:
                callpath_id_del.append(j)

    # delete the measurements for the callpaths that have been deleted
    for i in range(len(callpath_id_del)):
        for k in range(len(extrap_experiment.coordinates)):
            m = extrap_experiment.get_measurement(k, callpath_id_del[i], metric_id, current_analysis_type)
            if m == None:
                print(metrics)
                for o in range(len(metrics)):
                    metric = metrics[o]
                    callpath = callpaths[callpath_id_del[i]]
                    try:
                        extrap_experiment.delete_measurement(callpath, metric, current_analysis_type)
                        break
                    except KeyError:
                        pass

    # removing callpaths that do not have enough data points available from experiment
    for i in range(len(temp)):
        for j in range(len(callpaths)):
            if callpaths[j].name == temp[i]:
                extrap_experiment.callpaths[current_analysis_type].remove(callpaths[j])
                break
    
    return extrap_experiment
