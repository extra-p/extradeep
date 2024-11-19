from extradeep.entities.parameter import Parameter
from extradeep.entities.metric import Metric
from extradeep.entities.measurement import Measurement
from extradeep.entities.coordinate import Coordinate
from extradeep.entities.callpath import Callpath
from extradeep.entities.experiment import Experiment

from extradeep.util.progress_bar import DUMMY_PROGRESS
from extradeep.fileio.sqlite_helper import *
from extradeep.util.util_functions import get_sqlite_files_in_path

import logging

def read_gpu_util(folder, arguments, pbar=DUMMY_PROGRESS):
    """
    read_gpu_util function to read the average gpu utilization during the training steps and create a model for it as a function of the number of gpus used for training.

    :param folder: the filder where the experiment data for analysis is located
    :param arguments: the arguments passed in the command line for analysis
    :param pbar: dummy progress bar for showing the progress of file reading in the command line
    :return experiment: an extra-p experiment containing the loaded measurement data
    """

    paths, files = get_sqlite_files_in_path(folder)

    pbar.total += len(paths) + 2
    pbar.step("Reading Nsight Systems .sqlite files")

    # variable to save all seen experiment configurations
    configs = []

    # data container for different metrics
    gpu_utilization = {}
    gpu_utilization_between = {}

    for count, value in enumerate(paths):

        # update the progress bar
        pbar.update(1)

        # get the performance experiment configuration of this file
        config = read_experiment_configurations(count, files[count])

        # check if there is a problem with reading the config of the performance experiments
        if len(config.parameter_names) == len(config.parameter_values):
            if len(config.parameter_values) > 3 or len(config.parameter_values) < 1:
                return False, None
            else:
                nr_parameters = len(config.parameter_values)
        else:
            return False, None

        # open database in path and create connection
        db_instance = sqlite3.connect(value)
        cursor = db_instance.cursor()

        # load all nvtx markers created by the user with custom text
        query = "SELECT start, text, color, textId, domainId, eventType, rangeId, category, globalTid FROM NVTX_EVENTS WHERE eventType = 34;"
        query_result = get_data_from_db(cursor, query)

        if query_result != None:

            nvtx_marks = convert_nvtx_marks_to_objects(query_result)

            # identify important time stamps
            success, training = identify_training_process(nvtx_marks)
            if success == True:

                training_steps = identify_training_steps(nvtx_marks)
                epochs = identify_training_epochs(nvtx_marks)

                # set epoch to get the data from
                if arguments.epoch_for_modeling:
                    target_epoch = int(arguments.epoch_for_modeling)
                else:
                    target_epoch = 2
                if len(epochs) == 1:
                    target_epoch = 1

                # get the number of training steps for the target epoch
                # and get the runtime for the epoch from training and testing steps
                number = 0
                training_runtimes = []
                for i in range(len(training_steps)):
                    if training_steps[i].epoch_number == target_epoch:
                        number += 1
                        training_runtimes.append(training_steps[i].duration_seconds)

                # get the number of training steps for the target epoch
                number = 0
                for i in range(len(training_steps)):
                    if training_steps[i].epoch_number == target_epoch:
                        number += 1

                starttimes = []
                endtimes = []

                #logging.warning("There are "+str(number)+" training steps in the selected epoch "+str(target_epoch)+" that will be used for modeling.")

                # find the ith training step from training epoch 2
                # and get their start and end timestamps
                for i in range(number):
                    target_step_number = i + 1

                    start_time = None
                    end_time = None
                    start_time_long = None
                    end_time_long = None
                    for j in range(len(training_steps)):
                        if training_steps[j].step_number == target_step_number and training_steps[j].epoch_number == 2:
                            start_time = training_steps[j].start_time_seconds
                            start_time_long = training_steps[j].start_time_long
                            end_time = training_steps[j].end_time_seconds
                            end_time_long = training_steps[j].end_time_long
                            break
                    starttimes.append(start_time_long)
                    endtimes.append(end_time_long)

                kernel_runtime_sums = []
                memset_runtime_sums = []
                memcopy_runtime_sums = []

                time_between_steps = []
                runtime_between_steps = []

                for i in range(len(starttimes)):
                    start = starttimes[i]
                    end = endtimes[i]

                    # get memset operations runtime
                    query = "SELECT (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMSET AS memset WHERE start>="+str(start)+" AND end<="+str(end)+";"
                    memset_events = get_data_from_db(cursor, query)

                    memset_duration_sum = 0
                    for k in range(len(memset_events)):
                        memset_duration_seconds = memset_events[k][0]
                        memset_duration_sum += memset_duration_seconds

                    memset_runtime_sums.append(memset_duration_sum)

                    # get memcopy operations runtime
                    query = "SELECT (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY AS memcopy WHERE start>="+str(start)+" AND end<="+str(end)+";"
                    memcopy_events = get_data_from_db(cursor, query)

                    memcopy_duration_sum = 0
                    for k in range(len(memcopy_events)):
                        memcopy_duration_seconds = memcopy_events[k][0]
                        memcopy_duration_sum += memcopy_duration_seconds

                    memcopy_runtime_sums.append(memcopy_duration_sum)

                    # get kernel runtime
                    query = "SELECT (1.0 * (end - start) / 1000000000) AS duration_seconds, strings2.value AS shortName FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel INNER JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE start>="+str(start)+" AND end<="+str(end)+";"
                    kernel_events = get_data_from_db(cursor, query)

                    kernel_runtime_sum = 0
                    for k in range(len(kernel_events)):
                        kernel_runtime_sum += kernel_events[k][0]

                    kernel_runtime_sums.append(kernel_runtime_sum)

                    # get the data for the gpu utilization between training steps
                    try:
                        query = "SELECT (1.0 * (end - start) / 1000000000) AS duration_seconds, strings2.value AS shortName FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel INNER JOIN StringIds AS strings2 ON kernel.shortName = strings2.id WHERE start>="+str(endtimes[i])+" AND end<="+str(starttimes[i+1])+";"
                        kernel_events = get_data_from_db(cursor, query)
                        kernel_runtime_sum = 0
                        for k in range(len(kernel_events)):
                            kernel_runtime_sum += kernel_events[k][0]
                        query = "SELECT (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMSET AS memset WHERE start>="+str(endtimes[i])+" AND end<="+str(starttimes[i+1])+";"
                        memset_events = get_data_from_db(cursor, query)
                        memset_duration_sum = 0
                        for k in range(len(memset_events)):
                            memset_duration_seconds = memset_events[k][0]
                            memset_duration_sum += memset_duration_seconds
                        query = "SELECT (1.0 * (end - start) / 1000000000) AS duration_seconds FROM CUPTI_ACTIVITY_KIND_MEMCPY AS memcopy WHERE start>="+str(endtimes[i])+" AND end<="+str(starttimes[i+1])+";"
                        memcopy_events = get_data_from_db(cursor, query)
                        memcopy_duration_sum = 0
                        for k in range(len(memcopy_events)):
                            memcopy_duration_seconds = memcopy_events[k][0]
                            memcopy_duration_sum += memcopy_duration_seconds
                        runtime = memcopy_duration_sum + memset_duration_sum + kernel_runtime_sum
                        runtime_between_steps.append(runtime)
                        time_between_steps.append((starttimes[i+1]-endtimes[i])/1000000000)
                    except IndexError:
                        pass

                average_kernel_runtime_sum = np.median(kernel_runtime_sums)

                #print("average kernel runtime sum:", average_kernel_runtime_sum)

                average_memset_runtime_sum = np.median(memset_runtime_sums)

                #print("average memset runtime sum:", average_memset_runtime_sum)

                average_memcopy_runtime_sum = np.median(memcopy_runtime_sums)

                #print("average memcopy runtime sum:", average_memcopy_runtime_sum)

                average_step_runtime = np.median(training_runtimes)

                #print("average step runtime:", average_step_runtime)

                # calculate the gpu utilization in percent as model for steps
                one_percent = average_step_runtime / 100
                average_gpu_utilization_time = average_memcopy_runtime_sum + average_memset_runtime_sum + average_kernel_runtime_sum
                average_gpu_utilization_percent_per_step = average_gpu_utilization_time / one_percent
                #print("average gpu utilization %:", average_gpu_utilization_percent_per_step)

                gpu_utilization[config.id] = average_gpu_utilization_percent_per_step


                # calculate averages for stuff between training steps
                average_time_between_steps = np.median(time_between_steps)
                average_runtime_between_steps = np.median(runtime_between_steps)

                # calc the percentage for inbetween step utilization
                one_percent = average_time_between_steps / 100
                average_gpu_utilization_percent_between_steps = average_runtime_between_steps / one_percent

                gpu_utilization_between[config.id] = average_gpu_utilization_percent_between_steps

                configs.append(config)

            else:
                logging.warning("Data from file "+str(value)+" is missing NVTX training end timestamp indicating missing data in the input file. Input file will be ignored.")
        else:
            logging.warning("NVTX table does not exist in file "+str(value)+". Input file will be ignored.")

    pbar.step("Compute measurement data")
    pbar.update(1)

    # find unique experiment configurations
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

    class Storage():
        def __init__(self, config_object):
            self.config_object = config_object
            self.repetitions = []
            self.mpi_ranks = []

    # create a storage object for each unique experiment config
    storages = []
    for i in range(len(unique_configs)):
        storages.append(Storage(unique_configs[i]))

    # find the repetitions for each unique experiment config
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

    # for each unique experiment config get a list containing its repetitions and differen mpi ranks to average the metrics over
    experiment_config_ids = []
    for i in range(len(storages)):
        temp = []
        reps = storages[i].repetitions
        config = storages[i].config_object
        for j in range(len(reps)):
            rep = reps[j]
            ranks = storages[i].mpi_ranks
            for k in range(len(ranks)):
                rank = ranks[k]

                for l in range(len(configs)):
                    check = configs[l]
                    if config.app_name == check.app_name:
                        if config.parameter_names == check.parameter_names:
                            if config.parameter_values == check.parameter_values:
                                if rep == check.repetition_nr:
                                    if rank == check.mpi_rank:
                                        temp.append(check.id)
        experiment_config_ids.append(temp)

    # create new free metric dict for gpu utilization steps
    measurement_gpu_utilization = {}
    for i in range(len(unique_configs)):
        measurement_gpu_utilization[unique_configs[i].id] = 0

    # create new free metric dict for gpu utilization steps
    measurement_gpu_utilization_between = {}
    for i in range(len(unique_configs)):
        measurement_gpu_utilization_between[unique_configs[i].id] = 0

    # calculate median for all gpu utilizations
    # go through all unique experiment configurations
    for i in range(len(unique_configs)):
        values = []
        # go through this experiments repetitions and different mpi ranks
        for j in range(len(experiment_config_ids[i])):
            id = experiment_config_ids[i][j]
            # if a kernel does not exist in a training step, simply do not add any number for the metric
            try:
                value = gpu_utilization[id]
                values.append(value)
            except KeyError:
                pass
        # calculate the median of the metric
        if len(values) == 0:
            median = None
        else:
            median = np.median(values)
        measurement_gpu_utilization[unique_configs[i].id] = median

    # calculate median for all gpu utilizations
    # go through all unique experiment configurations between steps
    for i in range(len(unique_configs)):
        values = []
        # go through this experiments repetitions and different mpi ranks
        for j in range(len(experiment_config_ids[i])):
            id = experiment_config_ids[i][j]
            # if a kernel does not exist in a training step, simply do not add any number for the metric
            try:
                value = gpu_utilization_between[id]
                values.append(value)
            except KeyError:
                pass
        # calculate the median of the metric
        if len(values) == 0:
            median = None
        else:
            median = np.median(values)
        measurement_gpu_utilization_between[unique_configs[i].id] = median


    pbar.step("Create extrap experiment")
    pbar.update(1)

    # convert the dicts and other info to Extra-P objects to prepare for modeling them...
    # create one experiment for modeling and one for evaluation and one for checking the callpaths for both of them, to make sure they are compatible for analysis

    # create new empty extrap experiment
    extrap_experiment = Experiment()

    # create and add parameters to experiment
    for x in unique_configs[0].parameter_names:
        extrap_parameter = Parameter(x)
        extrap_experiment.add_parameter(extrap_parameter)

    # create and add metrics to experiment
    extrap_metric = Metric("gpu utilization [%]")
    extrap_experiment.add_metric(extrap_metric)

    extrap_callpath = Callpath("during training steps")
    extrap_experiment.add_callpath(extrap_callpath)

    extrap_callpath2 = Callpath("between training steps")
    extrap_experiment.add_callpath(extrap_callpath2)

    # create and add measurements to experiment
    for l in range(len(measurement_gpu_utilization)):

        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        extrap_experiment.add_coordinate(extrap_coordinate)
        cord_id = unique_configs[l].id

        # add measurements for gpu utilization steps
        if measurement_gpu_utilization[cord_id] != None:
            extrap_measurement = Measurement(extrap_coordinate, extrap_callpath, extrap_metric, measurement_gpu_utilization[cord_id])
            extrap_experiment.add_measurement(extrap_measurement)

    # create and add measurements to experiment
    for l in range(len(measurement_gpu_utilization_between)):

        # create and add coordinates to experiment
        extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
        extrap_experiment.add_coordinate(extrap_coordinate)
        cord_id = unique_configs[l].id

        # add measurements for gpu utilization steps
        if measurement_gpu_utilization_between[cord_id] != None:
            extrap_measurement = Measurement(extrap_coordinate, extrap_callpath2, extrap_metric, measurement_gpu_utilization_between[cord_id])
            extrap_experiment.add_measurement(extrap_measurement)

    # delete measurements that have no values and their callpaths...
    # this should be not the case in this analysis anyway...
    temp = []
    for i in range(len(extrap_experiment.callpaths)):
        counter = 0
        for j in range(len(extrap_experiment.coordinates)):
            # only need to check for one metric, if the kernel has no visit it also can't have runtime
            if extrap_experiment.get_measurement(j, i, 0) != None:
                counter += 1
        #if counter < int(arguments.minimum_required_points):
        if counter < len(unique_configs):
            logging.warning("Not enough data points for the callpath:"+str(extrap_experiment.callpaths[i])+".\nFound "+str(counter)+" point need at least "+str(arguments.minimum_required_points)+" points. Deleting this callpath from the experiment.")
            temp.append(extrap_experiment.callpaths[i].name)

    if len(temp) > 0:
        logging.warning("Skipping "+str(len(temp))+" callpaths.")

    callpath_id_del = []
    for i in range(len(temp)):
        for j in range(len(extrap_experiment.callpaths)):
            if extrap_experiment.callpaths[j].name == temp[i]:
                callpath_id_del.append(j)

    # delete the measurements for the callpaths that have been deleted
    for i in range(len(callpath_id_del)):
        for k in range(len(extrap_experiment.coordinates)):
            m = extrap_experiment.get_measurement(k, callpath_id_del[i], 0)
            if m == None:
                callpath = extrap_experiment.callpaths[callpath_id_del[i]]
                metric0 = extrap_experiment.metrics[0]
                extrap_experiment.delete_measurement(callpath, metric0)
                break

    # removing callpaths that do not have enough data points available from experiment
    for i in range(len(temp)):
        for j in range(len(extrap_experiment.callpaths)):
            if extrap_experiment.callpaths[j].name == temp[i]:
                extrap_experiment.callpaths.remove(extrap_experiment.callpaths[j])
                break

    return True, extrap_experiment
