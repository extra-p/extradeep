import csv
import numpy as np
from extradeep.util.progress_bar import DUMMY_PROGRESS
from os import listdir
from os.path import isfile, join

#DEBUG: ncu
#TODO: figure out how to do aggregation for total network and not by kernels
# use the flag: arguments.analysis == "kernel" or "total"
def read_ncu_file(folder, arguments, pbar=DUMMY_PROGRESS):

    # get the files in the folder
    allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    # only consider .sqlite data type files
    files = []
    for i in range(len(allfiles)):
        if allfiles[i].find(".csv") != -1:
            files.append(allfiles[i])
    files.sort()

    # construct the filepaths
    paths = []
    for i in range(len(files)):
        paths.append(folder+"/"+files[i])

    # variable to save all seen experiment configurations
    configs = []

    #DEBUG: create dicts for saving the metric data per experiment config
    kernels_per_config = {}
    unique_kernels_per_config = []
    counter_list = None

    pbar.total += len(paths) + 2

    pbar.step("Reading Nsight Compute .csv files")

    # loop through the different files
    for i in range(len(paths)):

        # update status bar
        pbar.update(1)

        # get the performance experiment configuration of this csv file
        config = read_experiment_configurations(i, files[i])
        configs.append(config)

        #read the csv file and return the kernels found
        kernels, counter_list = read_ncu_raw_csv(paths[i])

        # get the unique callpaths for this list of kernels
        unique_callpaths = get_unique_callpaths(arguments, kernels)

        # aggregate the kernel metrics for the unique callpaths selected
        aggregated_kernels = aggregate_metrics_per_callpath(arguments, unique_callpaths, kernels)

        # save the aggregated kernels in a dict that can be accesses by the configuration id of this experiment
        kernels_per_config[config.id] = aggregated_kernels

        # save the unique kernel list for this experiment config
        unique_kernels_per_config.append(unique_callpaths)

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

    #DEBUG: This method gets a list of all existing kernels considering all experiment configs
    # instead i could create a list of kernels that exist in all of them already here...
    # concat a list of unique kernels that resembles all considered mpi ranks and repetitions
    concat_kernel_list = []

    for i in range(len(unique_kernels_per_config)):
        for j in range(len(unique_kernels_per_config[i])):
            if i == 0:
                concat_kernel_list.append(unique_kernels_per_config[i][j])
            else:
                exists = False
                for k in range(len(concat_kernel_list)):
                    if unique_kernels_per_config[i][j] == concat_kernel_list[k]:
                        exists = True
                        break
                if exists == False:
                     concat_kernel_list.append(unique_kernels_per_config[i][j])

    # compute statistical metrics only over different mpi ranks and repetitions for each unique experiment configuration

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

    # summarize the data per unique kernel (standard extra-p approach)
    if arguments.analysis == "kernel":

        # create new free metric dict for kernels
        measurements = []
        for i in range(len(experiment_config_ids)):
            kernels = {}
            for j in range(len(concat_kernel_list)):
                kernels[concat_kernel_list[j]] = None
            measurements.append(kernels)

        # calculate median for all hardware counters
        # go through the list of unique kernels
        for k in range(len(concat_kernel_list)):
            # go through all unique experiment configurations
            for i in range(len(experiment_config_ids)):

                # create an empty array for each counters variables values
                counter_storage = []
                for _ in range(len(counter_list)):
                    counter_storage.append([])

                # go through this experiments repetitions and different mpi ranks
                for j in range(len(experiment_config_ids[i])):
                    id = experiment_config_ids[i][j]
                    # if a kernel does not exist in a training step, simply do not add any number for the metric
                    try:

                        kernel = kernels_per_config[id][concat_kernel_list[k]]

                        for l in range(len(counter_list)):
                            counter_name = counter_list[l]
                            counter_storage[l].append(kernel[str(counter_name)])

                    except KeyError:
                        pass

                # calculate the median counter value over all its values
                counter_medians = []
                for j in range(len(counter_storage)):
                    if len(counter_storage[j]) == 0:
                        counter_medians.append(None)
                    else:
                        counter_medians.append(np.median(counter_storage[j]))

                new_kernel = {}

                for j in range(len(counter_medians)):
                    new_kernel[str(counter_list[j])] = counter_medians[j]

                measurements[i][concat_kernel_list[k]] = new_kernel

        # create the extrap experiment and fill it with the data
        pbar.step("Create extrap experiment")
        pbar.update(1)

        # create new empty extrap experiment
        extrap_experiment = Experiment()

        # create and add parameters to experiment
        for x in unique_configs[0].parameter_names:
            extrap_parameter = Parameter(x)
            extrap_experiment.add_parameter(extrap_parameter)

        # create and add metrics to experiment
        extrap_metrics = []
        for i in range(len(counter_list)):
            counter_name = counter_list[i]
            extrap_metric = Metric(str(counter_name))
            extrap_metrics.append(extrap_metric)
            extrap_experiment.add_metric(extrap_metric)

        # create and add callpaths to experiment
        for i in range(len(concat_kernel_list)):
            extrap_callpath = Callpath(concat_kernel_list[i])
            extrap_experiment.add_callpath(extrap_callpath)

        # create and add measurements to experiment
        for l in range(len(measurements)):

            # create and add coordinates to experiment
            extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
            extrap_experiment.add_coordinate(extrap_coordinate)

            for i in range(len(concat_kernel_list)):

                # get the callpath from the experiment
                callpaths = extrap_experiment.callpaths
                id = -1
                for j in range(len(callpaths)):
                    callpath = callpaths[j]
                    if callpath.name == concat_kernel_list[i]:
                        id = j
                callpath = callpaths[id]

                # add measurements for all metrics
                #debug
                #print("kernel:",concat_kernel_list[i])
                #print("cord:",unique_configs[l].parameter_values)
                #print("measurements:",measurements[l][concat_kernel_list[i]])
                #print("\n")
                #if measurements[l][concat_kernel_list[i]] != None:

                # iterate through different hardware counters/metrics
                for k in range(len(counter_list)):

                    if measurements[l][concat_kernel_list[i]][str(counter_name)] != None:

                        counter_name = counter_list[k]
                        extrap_metric = extrap_metrics[k]
                        extrap_measurement = Measurement(extrap_coordinate, callpath, extrap_metric, measurements[l][concat_kernel_list[i]][str(counter_name)])
                        extrap_experiment.add_measurement(extrap_measurement)

        temp = []
        for i in range(len(extrap_experiment.callpaths)):
            for k in range(len(counter_list)):
                # go through the different hardware counters/metrics
                counter = 0
                for j in range(len(extrap_experiment.coordinates)):
                    # only need to check for one metric, if the kernel has no visit it also can't have runtime
                    try:
                        x = extrap_experiment.get_measurement(j, i, k).mean
                        counter += 1
                    except AttributeError:
                        pass

                    #if extrap_experiment.get_measurement(j, i, k).value != None:
                #        counter += 1

                #print("yyy:",counter)

                if counter < int(arguments.minimum_required_points):
                    logging.warning("Not enough data points for the callpath: "+str(extrap_experiment.callpaths[i])+" and metric: "+str(counter_list[k])+". Found "+str(counter)+" point(s) need at least "+str(arguments.minimum_required_points)+" points. Deleting this callpath from the experiment.")
                    temp.append([extrap_experiment.callpaths[i].name,str(counter_list[k])])

        if len(temp) > 0:
            logging.warning("Skipping "+str(len(temp))+" callpaths metric combinations.")

        logging.warning("Total number of callpath metric combinations is: "+str(len(extrap_experiment.callpaths)*len(counter_list)))

        """
        callpath_id_del = []
        for i in range(len(temp)):
            for j in range(len(extrap_experiment.callpaths)):
                if extrap_experiment.callpaths[j].name == temp[i]:
                    callpath_id_del.append(j)
        """

        # delete the measurements for the callpaths that have been deleted
        #DEBUG
        """
        for i in range(len(temp)):
            callpath_name = temp[i][0]
            callpath_id = -1
            for j in range(len(extrap_experiment.callpaths)):
                if extrap_experiment.callpaths[j].name == callpath_name:
                    callpath_id = j
                    break
            metric_name = temp[i][1]
            metric_id = -1
            for j in range(len(extrap_experiment.metrics)):
                if extrap_experiment.metrics[j].name == metric_name:
                    metric_id = j
                    break
            for k in range(len(extrap_experiment.coordinates)):
                m = extrap_experiment.get_measurement(k, callpath_id, metric_id)
                if m == None:
                    callpath = extrap_experiment.callpaths[callpath_id]
                    metric = extrap_metrics[metric_id]
                    extrap_experiment.delete_measurement(callpath, metric)
        """

        """
        for i in range(len(callpath_id_del)):
            for k in range(len(extrap_experiment.coordinates)):
                for j in range(len(extrap_metrics)):
                    m = extrap_experiment.get_measurement(k, callpath_id_del[i], j)
                    if m == None:
                        callpath = extrap_experiment.callpaths[callpath_id_del[i]]
                        metric = extrap_metrics[j]
                        extrap_experiment.delete_measurement(callpath, metric)
        """

        # removing callpaths that do not have enough data points available from experiment
        ## DEBUG:
        """
        for i in range(len(temp)):
            for j in range(len(extrap_experiment.callpaths)):
                if extrap_experiment.callpaths[j].name == temp[i]:
                    extrap_experiment.callpaths.remove(extrap_experiment.callpaths[j])
                    break
        """

        return extrap_experiment, temp


    # summarize the data per unique experiment config
    elif arguments.analysis == "total":

        # create new free metric dict for the counters of the different coordinates
        measurements = []
        unique_kernel_list_counter = 0

        # go through all unique experiment configurations
        for i in range(len(experiment_config_ids)):

            repetition_storage = []

            # go through this experiments repetitions and different mpi ranks
            for j in range(len(experiment_config_ids[i])):

                # create an empty array for each counters variables values
                counter_storage = []
                for _ in range(len(counter_list)):
                    counter_storage.append([])

                id = experiment_config_ids[i][j]
                callpaths = unique_kernels_per_config[unique_kernel_list_counter]

                # go through the kernels/callpaths of this unqiue experiment config
                for k in range(len(callpaths)):
                    callpath = callpaths[k]

                    kernel = kernels_per_config[id][callpath]

                    """
                    For finding error in csv files with different units for the counters...
                    #debug
                    print(kernel)
                    print("BOOBS:",experiment_config_ids[i],j)
                    #print(unique_configs)
                    print(unique_configs[i].parameter_values)
                    """

                    # go through the different counters
                    for l in range(len(counter_list)):
                        counter_name = counter_list[l]
                        counter_value = kernel[str(counter_name)]
                        counter_storage[l].append(counter_value)

                #summarize the values of all kernels for each counter
                for k in range(len(counter_storage)):
                    counter_storage[k] = np.sum(counter_storage[k])

                repetition_storage.append(counter_storage)

                unique_kernel_list_counter += 1

            counter_values = {}
            for j in range(len(counter_list)):
                counter_values[counter_list[j]] = 0

            y = []
            for j in range(len(counter_list)):
                x = []
                for k in range(len(repetition_storage)):
                    value = repetition_storage[k][j]
                    x.append(value)
                x = np.median(x)
                y.append(x)

            for j in range(len(counter_list)):
                counter_values[counter_list[j]] = y[j]

            measurements.append(counter_values)

        # create the extrap experiment and fill it with the data
        pbar.step("Create extrap experiment")
        pbar.update(1)

        # create new empty extrap experiment
        extrap_experiment = Experiment()

        # create and add parameters to experiment
        for x in unique_configs[0].parameter_names:
            extrap_parameter = Parameter(x)
            extrap_experiment.add_parameter(extrap_parameter)

        # create and add metrics to experiment
        extrap_metrics = []
        for i in range(len(counter_list)):
            counter_name = counter_list[i]
            extrap_metric = Metric(str(counter_name))
            extrap_metrics.append(extrap_metric)
            extrap_experiment.add_metric(extrap_metric)

        # create and add callpaths to experiment
        extrap_callpath = Callpath("total")
        extrap_experiment.add_callpath(extrap_callpath)

        # create and add measurements to experiment
        for l in range(len(measurements)):

            # create and add coordinates to experiment
            extrap_coordinate = Coordinate(unique_configs[l].parameter_values)
            extrap_experiment.add_coordinate(extrap_coordinate)

            # iterate through different hardware counters/metrics
            for k in range(len(counter_list)):

                value = measurements[l][counter_list[k]]
                extrap_metric = extrap_metrics[k]
                extrap_measurement = Measurement(extrap_coordinate, extrap_callpath, extrap_metric, value)
                extrap_experiment.add_measurement(extrap_measurement)

        return extrap_experiment, None


def aggregate_metrics_per_callpath(arguments, unique_callpaths, kernels):
    """
    aggregate_metrics_per_callpath function that summarizes the metrics of the input kernels by comparing the callpaths of all kernels in the list, if callpaths are the same the metrics are aggregated (sum and median computed).

    :param arguments: the list of arguments from the command line tool to see which type of callpaths should be used
    :param unique_callpaths: a list of unique callpaths from the list of all kernels
    :param kernels: a list of kernel dictionaries that should be aggregated
    :return aggregated_kernels: a list of kernels where each callpath exists only once
    """

    # check which method to use for kernel aggregation
    if arguments.aggregation == "median":
        aggregation_method = "median"
    elif arguments.aggregation == "sum":
        aggregation_method = "sum"

    # create a temporary storage that stores all kernels from one callpath in a box and these boxes in a list
    temp = []
    for j in range(len(unique_callpaths)):
        callpath = unique_callpaths[j]
        storage = []
        for k in range(len(kernels)):
            kernel = kernels[k]
            name = ""
            if arguments.callpaths == "function":
                name = kernel["function_name"]
            elif arguments.callpaths == "mangled":
                name = kernel["mangled_name"]
            elif arguments.callpaths == "demangled":
                name = kernel["demangled_name"]
            if name == callpath:
                storage.append(kernel)
        temp.append(storage)

    # iterate over all temporary storage boxed which equals the unique callpaths
    kernels = {}
    for j in range(len(temp)):
        callpath = unique_callpaths[j]
        data = temp[j]

        # memory counters
        dram_read_bytes = []
        dram_read_throughput = []
        dram_write_bytes = []
        dram_write_throughput = []
        dram_read_transactions = []
        dram_write_transactions = []

        dram_read_bytes_id = 0
        dram_read_throughput_id = 0
        dram_write_bytes_id = 0
        dram_write_throughput_id = 0
        dram_read_transactions_id = 0
        dram_write_transactions_id = 0

        """
        inst_fp_16 = []
        inst_fp_32 = []
        inst_fp_64 = []
        inst_integer = []
        """

        # floating point operations
        flop_count_dp_add = []
        flop_count_dp_mul = []
        flop_count_dp_fma = []
        flop_count_dp = []

        flop_count_sp_add = []
        flop_count_sp_mul = []
        flop_count_sp_fma = []
        flop_count_sp = []

        flop_count_hp_add = []
        flop_count_hp_mul = []
        flop_count_hp_fma = []
        flop_count_hp = []

        flop_count_dp_add_id = 0
        flop_count_dp_mul_id = 0
        flop_count_dp_fma_id = 0
        flop_count_dp_id = 0

        flop_count_sp_add_id = 0
        flop_count_sp_mul_id = 0
        flop_count_sp_fma_id = 0
        flop_count_sp_id = 0

        flop_count_hp_add_id = 0
        flop_count_hp_mul_id = 0
        flop_count_hp_fma_id = 0
        flop_count_hp_id = 0

        #flop_sp_efficiency = []
        #flop_dp_efficiency = []
        #flop_hp_efficiency = []

        for k in range(len(data)):
            kernel = data[k]

            # memory counters
            try:
                dram_read_bytes.append(kernel["dram_read_bytes"])
            except KeyError:
                dram_read_bytes_id = -1

            try:
                dram_read_throughput.append(kernel["dram_read_throughput"])
            except KeyError:
                dram_read_throughput_id = -1

            try:
                dram_write_bytes.append(kernel["dram_write_bytes"])
            except KeyError:
                dram_write_bytes_id = -1

            try:
                dram_write_throughput.append(kernel["dram_write_throughput"])
            except KeyError:
                dram_write_throughput_id = -1

            try:
                dram_read_transactions.append(kernel["dram_read_transactions"])
            except KeyError:
                dram_read_transactions_id = -1

            try:
                dram_write_transactions.append(kernel["dram_write_transactions"])
            except KeyError:
                dram_write_transactions_id = -1

            """
            inst_fp_16.append(kernel.inst_fp_16)
            inst_fp_32.append(kernel.inst_fp_32)
            inst_fp_64.append(kernel.inst_fp_64)
            inst_integer.append(kernel.inst_integer)
            """

            # floating point operations
            try:
                flop_count_dp_add.append(kernel["flop_count_dp_add"])
            except KeyError:
                flop_count_dp_add_id = -1

            try:
                flop_count_dp_mul.append(kernel["flop_count_dp_mul"])
            except KeyError:
                flop_count_dp_mul_id = -1

            try:
                flop_count_dp_fma.append(kernel["flop_count_dp_fma"])
            except KeyError:
                flop_count_dp_fma_id = -1

            try:
                flop_count_dp.append(kernel["flop_count_dp"])
            except KeyError:
                flop_count_dp_id = -1

            try:
                flop_count_sp_add.append(kernel["flop_count_sp_add"])
            except KeyError:
                flop_count_sp_add_id = -1

            try:
                flop_count_sp_mul.append(kernel["flop_count_sp_mul"])
            except KeyError:
                flop_count_sp_mul_id = -1

            try:
                flop_count_sp_fma.append(kernel["flop_count_sp_fma"])
            except KeyError:
                flop_count_sp_fma_id = -1

            try:
                flop_count_sp.append(kernel["flop_count_sp"])
            except KeyError:
                flop_count_sp_id = -1

            try:
                flop_count_hp_add.append(kernel["flop_count_hp_add"])
            except KeyError:
                flop_count_hp_add_id = -1

            try:
                flop_count_hp_mul.append(kernel["flop_count_hp_mul"])
            except KeyError:
                flop_count_hp_mul_id = -1

            try:
                flop_count_hp_fma.append(kernel["flop_count_hp_fma"])
            except KeyError:
                flop_count_hp_fma_id = -1

            try:
                flop_count_hp.append(kernel["flop_count_hp"])
            except KeyError:
                flop_count_hp_id = -1

            #flop_dp_efficiency.append(kernel["flop_dp_efficiency"])
            #flop_hp_efficiency.append(kernel["flop_hp_efficiency"])
            #flop_sp_efficiency.append(kernel["flop_sp_efficiency"])

        if aggregation_method == "median":

            # memory counters
            if len(dram_read_bytes) > 0:
                dram_read_bytes = np.median(dram_read_bytes)

            if len(dram_read_throughput) > 0:
                dram_read_throughput = np.median(dram_read_throughput)

            if len(dram_write_bytes) > 0:
                dram_write_bytes = np.median(dram_write_bytes)

            if len(dram_write_throughput) > 0:
                dram_write_throughput = np.median(dram_write_throughput)

            if len(dram_read_transactions) > 0:
                dram_read_transactions = np.median(dram_read_transactions)

            if len(dram_write_transactions) > 0:
                dram_write_transactions = np.median(dram_write_transactions)

            """
            inst_fp_16 = np.median(inst_fp_16)
            inst_fp_32 = np.median(inst_fp_32)
            inst_fp_64 = np.median(inst_fp_64)
            inst_integer = np.median(inst_integer)
            """

            # floating point operations medians
            if len(flop_count_dp_add) > 0:
                flop_count_dp_add = np.median(flop_count_dp_add)

            if len(flop_count_dp_mul) > 0:
                flop_count_dp_mul = np.median(flop_count_dp_mul)

            if len(flop_count_dp_fma) > 0:
                flop_count_dp_fma = np.median(flop_count_dp_fma)

            if len(flop_count_dp) > 0:
                flop_count_dp = np.median(flop_count_dp)

            if len(flop_count_sp_add) > 0:
                flop_count_sp_add = np.median(flop_count_sp_add)

            if len(flop_count_sp_mul) > 0:
                flop_count_sp_mul = np.median(flop_count_sp_mul)

            if len(flop_count_sp_fma) > 0:
                flop_count_sp_fma = np.median(flop_count_sp_fma)

            if len(flop_count_sp) > 0:
                flop_count_sp = np.median(flop_count_sp)

            if len(flop_count_hp_add) > 0:
                flop_count_hp_add = np.median(flop_count_hp_add)

            if len(flop_count_hp_mul) > 0:
                flop_count_hp_mul = np.median(flop_count_hp_mul)

            if len(flop_count_hp_fma) > 0:
                flop_count_hp_fma = np.median(flop_count_hp_fma)

            if len(flop_count_hp) > 0:
                flop_count_hp = np.median(flop_count_hp)

            #flop_dp_efficiency = np.median(flop_dp_efficiency)
            #flop_sp_efficiency = np.median(flop_sp_efficiency)
            #flop_hp_efficiency = np.median(flop_hp_efficiency)

        elif aggregation_method == "sum":

            # memory counters
            if len(dram_read_bytes) > 0:
                dram_read_bytes = np.sum(dram_read_bytes)

            if len(dram_read_throughput) > 0:
                dram_read_throughput = np.sum(dram_read_throughput)

            if len(dram_write_bytes) > 0:
                dram_write_bytes = np.sum(dram_write_bytes)

            if len(dram_write_throughput) > 0:
                dram_write_throughput = np.sum(dram_write_throughput)

            if len(dram_read_transactions) > 0:
                dram_read_transactions = np.sum(dram_read_transactions)

            if len(dram_write_transactions) > 0:
                dram_write_transactions = np.sum(dram_write_transactions)

            # floating point operations medians
            if len(flop_count_dp_add) > 0:
                flop_count_dp_add = np.sum(flop_count_dp_add)

            if len(flop_count_dp_mul) > 0:
                flop_count_dp_mul = np.sum(flop_count_dp_mul)

            if len(flop_count_dp_fma) > 0:
                flop_count_dp_fma = np.sum(flop_count_dp_fma)

            if len(flop_count_dp) > 0:
                flop_count_dp = np.sum(flop_count_dp)

            if len(flop_count_sp_add) > 0:
                flop_count_sp_add = np.sum(flop_count_sp_add)

            if len(flop_count_sp_mul) > 0:
                flop_count_sp_mul = np.sum(flop_count_sp_mul)

            if len(flop_count_sp_fma) > 0:
                flop_count_sp_fma = np.sum(flop_count_sp_fma)

            if len(flop_count_sp) > 0:
                flop_count_sp = np.sum(flop_count_sp)

            if len(flop_count_hp_add) > 0:
                flop_count_hp_add = np.sum(flop_count_hp_add)

            if len(flop_count_hp_mul) > 0:
                flop_count_hp_mul = np.sum(flop_count_hp_mul)

            if len(flop_count_hp_fma) > 0:
                flop_count_hp_fma = np.sum(flop_count_hp_fma)

            if len(flop_count_hp) > 0:
                flop_count_hp = np.sum(flop_count_hp)

        kernel = {}

        # store this general stuff
        kernel["ncu_id"] = data[0]["ncu_id"]
        kernel["function_name"] = data[0]["function_name"]
        kernel["mangled_name"] = data[0]["mangled_name"]
        kernel["demangled_name"] = data[0]["demangled_name"]
        kernel["thread_id"] = data[0]["thread_id"]
        kernel["process"] = data[0]["process"]
        kernel["device_name"] = data[0]["device_name"]
        kernel["grid_offset"] = data[0]["grid_offset"]
        kernel["grid_size"] = data[0]["grid_size"]
        kernel["block_size"] = data[0]["block_size"]
        kernel["grid_dimensions"] = data[0]["grid_dimensions"]
        kernel["gpu_time_duration_sum"] = data[0]["gpu_time_duration_sum"]

        if flop_count_dp_add_id != -1:
            kernel["flop_count_dp_add"] = flop_count_dp_add

        if flop_count_dp_mul_id != -1:
            kernel["flop_count_dp_mul"] = flop_count_dp_mul

        if flop_count_dp_fma_id != -1:
            kernel["flop_count_dp_fma"] = flop_count_dp_fma

        if flop_count_dp_id != -1:
            kernel["flop_count_dp"] = flop_count_dp

        if flop_count_sp_add_id != -1:
            kernel["flop_count_sp_add"] = flop_count_sp_add

        if flop_count_sp_mul_id != -1:
            kernel["flop_count_sp_mul"] = flop_count_sp_mul

        if flop_count_sp_fma_id != -1:
            kernel["flop_count_sp_fma"] = flop_count_sp_fma

        if flop_count_sp_id != -1:
            kernel["flop_count_sp"] = flop_count_sp

        if flop_count_hp_add_id != -1:
            kernel["flop_count_hp_add"] = flop_count_hp_add

        if flop_count_hp_mul_id != -1:
            kernel["flop_count_hp_mul"] = flop_count_hp_mul

        if flop_count_hp_fma_id != -1:
            kernel["flop_count_hp_fma"] = flop_count_hp_fma

        if flop_count_hp_id != -1:
            kernel["flop_count_hp"] = flop_count_hp

        """
        kernel["flop_count_dp_add_sum"] = flop_count_dp_add_sum
        kernel["flop_count_dp_mul_sum"] = flop_count_dp_mul_sum
        kernel["flop_count_dp_fma_sum"] = flop_count_dp_fma_sum
        kernel["flop_count_dp_sum"] = flop_count_dp_sum
        kernel["flop_count_sp_add_sum"] = flop_count_sp_add_sum
        kernel["flop_count_sp_mul_sum"] = flop_count_sp_mul_sum
        kernel["flop_count_sp_fma_sum"] = flop_count_sp_fma_sum
        kernel["flop_count_sp_sum"] = flop_count_sp_sum
        kernel["flop_count_hp_add_sum"] = flop_count_hp_add_sum
        kernel["flop_count_hp_mul_sum"] = flop_count_hp_mul_sum
        kernel["flop_count_hp_fma_sum"] = flop_count_hp_fma_sum
        kernel["flop_count_hp_sum"] = flop_count_hp_sum
        """

        if dram_read_bytes_id != -1:
            kernel["dram_read_bytes"] = dram_read_bytes

        if dram_read_throughput_id != -1:
            kernel["dram_read_throughput"] = dram_read_throughput

        if dram_write_bytes_id != -1:
            kernel["dram_write_bytes"] = dram_write_bytes

        if dram_write_throughput_id != -1:
            kernel["dram_write_throughput"] = dram_write_throughput

        if dram_read_transactions_id != -1:
            kernel["dram_read_transactions"] = dram_read_transactions

        if dram_write_transactions_id != -1:
            kernel["dram_write_transactions"] = dram_write_transactions

        kernels[callpath] = kernel

    return kernels

def get_unique_callpaths(arguments, kernels):
    """
    get_unique_callpaths function creates a list of unique callpaths found in a given list of metrics

    :param arguments: the command line arguments from the modeling tool to see which type of callpath to use to create the unique list
    :param kernels: a list of kernels as dicts
    :returns callpaths: a list of unique callpaths as strings
    """

    callpaths = []
    if arguments.callpaths:
        if arguments.callpaths == "demangled":
            for k in range(len(kernels)):
                if k == 0:
                    callpaths.append(kernels[k]["demangled_name"])
                else:
                    exists = False
                    for j in range(len(callpaths)):
                        if callpaths[j] == kernels[k]["demangled_name"]:
                            exists = True
                            break
                    if exists == False:
                        callpaths.append(kernels[k]["demangled_name"])

        elif arguments.callpaths == "mangled":
            for k in range(len(kernels)):
                if k == 0:
                    callpaths.append(kernels[k]["mangled_name"])
                else:
                    exists = False
                    for j in range(len(callpaths)):
                        if callpaths[j] == kernels[k]["mangled_name"]:
                            exists = True
                            break
                    if exists == False:
                        callpaths.append(kernels[k]["mangled_name"])

        elif arguments.callpaths == "function":
            for k in range(len(kernels)):
                if k == 0:
                    callpaths.append(kernels[k]["function_name"])
                else:
                    exists = False
                    for j in range(len(callpaths)):
                        if callpaths[j] == kernels[k]["function_name"]:
                            exists = True
                            break
                    if exists == False:
                        callpaths.append(kernels[k]["function_name"])
    return callpaths

def read_ncu_raw_csv(path):
    """
    read_ncu_raw_csv function reads the raw output of Nsight Compute in .csv format. Returns a list of kernels where each item is a dictonary containing a profiled kernel with its metrics.

    :param path: the path to the file to read as a string
    :return kernels: the list of profiled kernels
    :return counter_list: a list of counter names that exist in the file
    """

    # read the csv file
    fields = []
    rows = []
    with open(path, newline="\n") as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)

    flop_count_dp_add_id = -1
    flop_count_dp_mul_id = -1
    flop_count_dp_fma_id = -1
    flop_count_sp_add_id = -1
    flop_count_sp_mul_id = -1
    flop_count_sp_fma_id = -1
    flop_count_hp_add_id = -1
    flop_count_hp_mul_id = -1
    flop_count_hp_fma_id = -1

    dram_bytes_read_sum_id = -1
    dram_bytes_write_sum_id = -1
    dram_sectors_read_sum_id = -1
    dram_sectors_write_sum_id = -1
    dram_bytes_read_sum_per_second_id = -1
    dram_bytes_write_sum_per_second_id = -1

    counter_list = []


    #DEBUG: carefull the units in csv data from nsight compute can have different units
    # for resnet-110 I get Kbytes instead of Mbytes only for dram__bytes_write.sum ...
    # need to fix this somehow later...
    special_condition = False
    special_condition2 = False

    # get the ids of the metrics I am interested in
    for k in range(len(fields)):

        #memory counters
        if fields[k] == "dram__bytes_read.sum [Mbytes]":
            dram_bytes_read_sum_id = k

        if fields[k] == "dram__bytes_read.sum.per_second [Gbyte/second]":
            dram_bytes_read_sum_per_second_id = k

        # debug exception...
        if fields[k] == "dram__bytes_write.sum [Mbytes]" or fields[k] == "dram__bytes_write.sum [Kbytes]":
            dram_bytes_write_sum_id = k
            if fields[k] == "dram__bytes_write.sum [Kbytes]":
                special_condition = True

        if fields[k] == "dram__bytes_write.sum.per_second [Gbyte/second]" or fields[k] == "dram__bytes_write.sum.per_second [Mbyte/second]":
            dram_bytes_write_sum_per_second_id = k
            if fields[k] == "dram__bytes_write.sum.per_second [Mbyte/second]":
                special_condition2 = True

        if fields[k] == "dram__sectors_read.sum [sector]":
            dram_sectors_read_sum_id = k

        if fields[k] == "dram__sectors_write.sum [sector]":
            dram_sectors_write_sum_id = k

        """
        if fields[k] == "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum [inst]":
            inst_fp_16_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum [inst]":
            inst_fp_32_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum [inst]":
            inst_fp_64_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_integer_pred_on.sum [inst]":
            inst_integer_id = k
        """

        # floating point operations
        if fields[k] == "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum [inst]":
            flop_count_dp_add_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum [inst]":
            flop_count_dp_mul_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum [inst]":
            flop_count_dp_fma_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum [inst]":
            flop_count_sp_add_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum [inst]":
            flop_count_sp_mul_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum [inst]":
            flop_count_sp_fma_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum [inst]":
            flop_count_hp_add_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum [inst]":
            flop_count_hp_mul_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum [inst]":
            flop_count_hp_fma_id = k

        """
        if fields[k] == "smsp__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.avg.pct_of_peak_sustained_elapsed [%]":
            flop_dp_efficiency_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.avg.pct_of_peak_sustained_elapsed [%]":
            flop_sp_efficiency_id = k

        if fields[k] == "smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed [%]":
            flop_hp_efficiency_id = k
        """

    kernels = []

    # create objects for each callpath
    for k in range(len(rows)):

        # create new dict kernel to store all the values
        kernel = {}

        # general stuff
        ncu_id = int(rows[k][0])
        function_name = rows[k][3]
        mangled_name = rows[k][4]
        demangled_name = rows[k][5]
        process = rows[k][6]
        thread_id = int(rows[k][7])
        device_name = rows[k][8]
        grid_offset = rows[k][11]
        grid_size = rows[k][12]
        block_size = rows[k][13]
        grid_dimensions = rows[k][14]
        gpu_time_duration_sum = rows[k][15]

        # store this general stuff
        kernel["ncu_id"] = ncu_id
        kernel["function_name"] = function_name
        kernel["mangled_name"] = mangled_name
        kernel["demangled_name"] = demangled_name
        kernel["thread_id"] = thread_id
        kernel["process"] = process
        kernel["device_name"] = device_name
        kernel["grid_offset"] = grid_offset
        kernel["grid_size"] = grid_size
        kernel["block_size"] = block_size
        kernel["grid_dimensions"] = grid_dimensions
        kernel["gpu_time_duration_sum"] = gpu_time_duration_sum

        # memory metrics
        if dram_bytes_read_sum_id != -1:
            dram_read_bytes = rows[k][dram_bytes_read_sum_id]
            dram_read_bytes = dram_read_bytes.replace(".","")
            dram_read_bytes = float(dram_read_bytes.replace(",","."))
            kernel["dram_read_bytes"] = dram_read_bytes
            if k == 0:
                counter_list.append("dram_read_bytes")

        if dram_bytes_read_sum_per_second_id != -1:
            dram_read_throughput = rows[k][dram_bytes_read_sum_per_second_id]
            dram_read_throughput = dram_read_throughput.replace(".","")
            dram_read_throughput = float(dram_read_throughput.replace(",","."))
            kernel["dram_read_throughput"] = dram_read_throughput
            if k == 0:
                counter_list.append("dram_read_throughput")

        if dram_bytes_write_sum_id != -1:
            dram_write_bytes = rows[k][dram_bytes_write_sum_id]
            dram_write_bytes = dram_write_bytes.replace(".","")
            dram_write_bytes = float(dram_write_bytes.replace(",","."))
            if special_condition == True:
                dram_write_bytes = dram_write_bytes/1000
            kernel["dram_write_bytes"] = dram_write_bytes
            if k == 0:
                counter_list.append("dram_write_bytes")

        if dram_bytes_write_sum_per_second_id != -1:
            dram_write_throughput = rows[k][dram_bytes_write_sum_per_second_id]
            dram_write_throughput = dram_write_throughput.replace(".","")
            dram_write_throughput = float(dram_write_throughput.replace(",","."))
            if special_condition2 == True:
                dram_write_throughput = dram_write_throughput / 1000
            kernel["dram_write_throughput"] = dram_write_throughput
            if k == 0:
                counter_list.append("dram_write_throughput")

        if dram_sectors_read_sum_id != -1:
            dram_read_transactions = rows[k][dram_sectors_read_sum_id]
            dram_read_transactions = dram_read_transactions.replace(".","")
            dram_read_transactions = float(dram_read_transactions.replace(",","."))
            kernel["dram_read_transactions"] = dram_read_transactions
            if k == 0:
                counter_list.append("dram_read_transactions")

        if dram_sectors_write_sum_id != -1:
            dram_write_transactions = rows[k][dram_sectors_write_sum_id]
            dram_write_transactions = dram_write_transactions.replace(".","")
            dram_write_transactions = float(dram_write_transactions.replace(",","."))
            kernel["dram_write_transactions"] = dram_write_transactions
            if k == 0:
                counter_list.append("dram_write_transactions")

        """
        # fp instruction metrics
        inst_fp_16 = rows[k][inst_fp_16_id]
        inst_fp_16 = inst_fp_16.replace(".","")
        inst_fp_16 = float(inst_fp_16.replace(",","."))

        inst_fp_32 = rows[k][inst_fp_32_id]
        inst_fp_32 = inst_fp_32.replace(".","")
        inst_fp_32 = float(inst_fp_32.replace(",","."))

        inst_fp_64 = rows[k][inst_fp_64_id]
        inst_fp_64 = inst_fp_64.replace(".","")
        inst_fp_64 = float(inst_fp_64.replace(",","."))

        inst_integer = rows[k][inst_integer_id]
        inst_integer = inst_integer.replace(".","")
        inst_integer = float(inst_integer.replace(",","."))
        """

        # fp operations metrics
        if flop_count_dp_add_id != -1:
            flop_count_dp_add = rows[k][flop_count_dp_add_id]
            flop_count_dp_add = flop_count_dp_add.replace(".","")
            flop_count_dp_add = float(flop_count_dp_add.replace(",","."))
            kernel["flop_count_dp_add"] = flop_count_dp_add
            if k == 0:
                counter_list.append("flop_count_dp_add")

        if flop_count_dp_mul_id != -1:
            flop_count_dp_mul = rows[k][flop_count_dp_mul_id]
            flop_count_dp_mul = flop_count_dp_mul.replace(".","")
            flop_count_dp_mul = float(flop_count_dp_mul.replace(",","."))
            kernel["flop_count_dp_mul"] = flop_count_dp_mul
            if k == 0:
                counter_list.append("flop_count_dp_mul")

        if flop_count_dp_fma_id != -1:
            flop_count_dp_fma = rows[k][flop_count_dp_fma_id]
            flop_count_dp_fma = flop_count_dp_fma.replace(".","")
            flop_count_dp_fma = float(flop_count_dp_fma.replace(",","."))
            kernel["flop_count_dp_fma"] = flop_count_dp_fma
            if k == 0:
                counter_list.append("flop_count_dp_fma")

        if flop_count_dp_add_id != -1 and flop_count_dp_mul_id != -1 and flop_count_dp_fma_id != -1:
            flop_count_dp = flop_count_dp_add + flop_count_dp_mul + flop_count_dp_fma * 2
            kernel["flop_count_dp"] = flop_count_dp
            if k == 0:
                counter_list.append("flop_count_dp")

        """
        flop_dp_efficiency = rows[k][flop_dp_efficiency_id]
        flop_dp_efficiency = flop_dp_efficiency.replace(".","")
        flop_dp_efficiency = float(flop_dp_efficiency.replace(",","."))
        """

        if flop_count_sp_add_id != -1:
            flop_count_sp_add = rows[k][flop_count_sp_add_id]
            flop_count_sp_add = flop_count_sp_add.replace(".","")
            flop_count_sp_add = float(flop_count_sp_add.replace(",","."))
            kernel["flop_count_sp_add"] = flop_count_sp_add
            if k == 0:
                counter_list.append("flop_count_sp_add")

        if flop_count_sp_mul_id != -1:
            flop_count_sp_mul = rows[k][flop_count_sp_mul_id]
            flop_count_sp_mul = flop_count_sp_mul.replace(".","")
            flop_count_sp_mul = float(flop_count_sp_mul.replace(",","."))
            kernel["flop_count_sp_mul"] = flop_count_sp_mul
            if k == 0:
                counter_list.append("flop_count_sp_mul")

        if flop_count_sp_fma_id != -1:
            flop_count_sp_fma = rows[k][flop_count_sp_fma_id]
            flop_count_sp_fma = flop_count_sp_fma.replace(".","")
            flop_count_sp_fma = float(flop_count_sp_fma.replace(",","."))
            kernel["flop_count_sp_fma"] = flop_count_sp_fma
            if k == 0:
                counter_list.append("flop_count_sp_fma")

        if flop_count_sp_add_id != -1 and flop_count_sp_mul_id != -1 and flop_count_sp_fma_id != -1:
            flop_count_sp = flop_count_sp_add + flop_count_sp_mul + flop_count_sp_fma * 2
            kernel["flop_count_sp"] = flop_count_sp
            if k == 0:
                counter_list.append("flop_count_sp")

        """
        flop_sp_efficiency = rows[k][flop_sp_efficiency_id]
        flop_sp_efficiency = flop_sp_efficiency.replace(".","")
        flop_sp_efficiency = float(flop_sp_efficiency.replace(",","."))
        """

        if flop_count_hp_add_id != -1:
            flop_count_hp_add = rows[k][flop_count_hp_add_id]
            flop_count_hp_add = flop_count_hp_add.replace(".","")
            flop_count_hp_add = float(flop_count_hp_add.replace(",","."))
            kernel["flop_count_hp_add"] = flop_count_hp_add
            if k == 0:
                counter_list.append("flop_count_hp_add")

        if flop_count_hp_mul_id != -1:
            flop_count_hp_mul = rows[k][flop_count_hp_mul_id]
            flop_count_hp_mul = flop_count_hp_mul.replace(".","")
            flop_count_hp_mul = float(flop_count_hp_mul.replace(",","."))
            kernel["flop_count_hp_mul"] = flop_count_hp_mul
            if k == 0:
                counter_list.append("flop_count_hp_mul")

        if flop_count_hp_fma_id != -1:
            flop_count_hp_fma = rows[k][flop_count_hp_fma_id]
            flop_count_hp_fma = flop_count_hp_fma.replace(".","")
            flop_count_hp_fma = float(flop_count_hp_fma.replace(",","."))
            kernel["flop_count_hp_fma"] = flop_count_hp_fma
            if k == 0:
                counter_list.append("flop_count_hp_fma")

        if flop_count_hp_add_id != -1 and flop_count_hp_mul_id != -1 and flop_count_hp_fma_id != -1:
            flop_count_hp = flop_count_hp_add + flop_count_hp_mul + flop_count_hp_fma * 2
            kernel["flop_count_hp"] = flop_count_hp
            if k == 0:
                counter_list.append("flop_count_hp")

        """
        flop_hp_efficiency = rows[k][flop_hp_efficiency_id]
        flop_hp_efficiency = flop_hp_efficiency.replace(".","")
        flop_hp_efficiency = float(flop_hp_efficiency.replace(",","."))
        """

        kernels.append(kernel)

    return kernels, counter_list




"""

if arguments.ncu and temp != None:
            coordinates = experiment.coordinates
            callpaths = experiment.callpaths
            metrics = experiment.metrics
            modeler = experiment.modelers[0]
            text = ""

            cc = 0

            for callpath_id in range(len(callpaths)):
                callpath = callpaths[callpath_id]
                callpath_string = callpath.name
                temp_text = "Callpath: " + callpath_string + "\n"
                temp_text3 = ""

                for metric_id in range(len(metrics)):
                    metric = metrics[metric_id]
                    metric_string = metric.name

                    temp_text2 = ""

                    exists = False
                    for k in range(len(temp)):
                        x = temp[k]
                        call = x[0]
                        met = x[1]
                        if met == metric_string and call == callpath_string:
                            exists = True
                            break
                    if exists == False:
                        cc += 1
                        temp_text2 += "\n\tMetric: " + metric_string + "\n"
                        for coordinate_id in range(len(coordinates)):
                            coordinate = coordinates[coordinate_id]
                            dimensions = coordinate.dimensions
                            coordinate_text = "Measurement point: ("
                            for dimension in range(dimensions):
                                value = coordinate[dimension]
                                value_string = "{:.2E}".format(value)
                                coordinate_text += value_string + ","
                            coordinate_text = coordinate_text[:-1]
                            coordinate_text += ")"
                            measurement = experiment.get_measurement(coordinate_id, callpath_id, metric_id)
                            if measurement == None:
                                value_mean = 0
                                value_median = 0
                            else:
                                value_mean = measurement.mean
                                value_median = measurement.median
                            temp_text2 += f"\t\t{coordinate_text} Mean: {value_mean:.2E} Median: {value_median:.2E}\n"
                        try:
                            model = modeler.models[callpath, metric]
                        except KeyError as e:
                            model = None
                        if model != None:
                            hypothesis = model.hypothesis
                            function = hypothesis.function
                            rss = hypothesis.RSS
                            ar2 = hypothesis.AR2
                            re = hypothesis.RE
                            smape = hypothesis.SMAPE
                            function_string = function.to_string(*experiment.parameters)
                        else:
                            rss = None
                            ar2 = None
                            re = None
                            smape = None
                            function_string = "None"
                        temp_text2 += "\t\tModel: " + function_string + "\n"
                        temp_text2 += "\t\tRSS: {:.2E}\n".format(rss)
                        temp_text2 += "\t\tAdjusted R^2: {:.2E}\n".format(ar2)
                        temp_text2 += "\t\tRE: {:.2E}\n".format(re)
                        temp_text2 += "\t\tSMAPE: {:.2E}\n".format(smape)
                        temp_text3 += temp_text2

                if temp_text3 != "":
                    text += temp_text
                    text += temp_text3
                    text += "\n\n"

            print(text)
            logging.warning("Combinations with "+str(arguments.minimum_required_points)+" points for modeling: "+str(cc))

        else:
            # format modeler output into text
            text = format_output(experiment, printtype)

            # print formatted output to command line
            print(text)



"""