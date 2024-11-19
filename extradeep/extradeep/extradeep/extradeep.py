# This file is part of the Extra-Deep software (https://github.com/extra-p/extradeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import logging
import os
import sys
from itertools import chain
from time import sleep

import extradeep
from extradeep.modelers import multi_parameter
from extradeep.modelers import single_parameter
from extradeep.modelers.abstract_modeler import MultiParameterModeler
from extradeep.modelers.model_generator import ModelGenerator
from extradeep.util.options_parser import ModelerOptionsAction, ModelerHelpAction
from extradeep.util.options_parser import SINGLE_PARAMETER_MODELER_KEY, SINGLE_PARAMETER_OPTIONS_KEY
from extradeep.util.progress_bar import ProgressBar
from extradeep.fileio.io_helper import format_output
from extradeep.fileio.io_helper import save_output
from extradeep.util.exceptions import *
from extradeep.fileio import experiment_io
from extradeep.fileio.sqlite_helper import *

from extradeep.util.evaluation import evaluate_phases
from extradeep.fileio.gpu_util_reader import read_gpu_util
from extradeep.util.evaluation import evaluate_gpu_util

from extradeep.fileio.os_reader import read_os
from extradeep.fileio.mpi_reader import read_mpi
from extradeep.fileio.cublas_reader import read_cublas
from extradeep.fileio.cudnn_reader import read_cudnn
from extradeep.fileio.cuda_api_reader import read_cuda_api
from extradeep.fileio.cuda_kernel_reader import read_kernel
from extradeep.fileio.memory_reader import read_memory
from extradeep.fileio.nvtx_reader import read_nvtx
from extradeep.fileio.training_step_reader import read_training_steps
from extradeep.fileio.epoch_reader import read_epochs
from extradeep.fileio.testing_step_reader import read_testing_steps
from extradeep.fileio.app_phase_reader import read_application_phases

from extradeep.fileio.experiment_io import read_experiment

from extradeep.util.evaluate_training_steps import evaluate

from extradeep.fileio.ncu_reader import read_ncu_file

from extradeep.util.evaluation import evaluate_kernels



def main(args=None, prog=None):
    """
    main function that runs the command line tool version of extradeep

    :param args: parameter for arguments entered in the terminal
    :param prog: parameter for programm
    """

    # Define argparse commands for input, output operations

    modelers_list = list(set(k.lower() for k in chain(single_parameter.all_modelers.keys(), multi_parameter.all_modelers.keys())))

    parser = argparse.ArgumentParser(prog=prog, description=extradeep.__description__, add_help=False)

    positional_arguments = parser.add_argument_group("Positional arguments")


    # Define basic program arguements such as log, help, and version outputs

    basic_arguments = parser.add_argument_group("Optional arguments")

    basic_arguments.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")

    basic_arguments.add_argument("-v", "--version", action="version", version=extradeep.__title__ + " " + extradeep.__version__, help="Show program's version number and exit.")

    basic_arguments.add_argument("--log", action="store", dest="log_level", type=str.lower, default='warning', choices=['debug', 'info', 'warning', 'error', 'critical'], help="Set program's log level (default: warning).")


    # Define the path argument that points to the folder the data will be loaded from

    positional_arguments.add_argument("path", metavar="FILE_PATH", type=str, action="store",
                                      help="Specify a file path for Extra-Deep to work with")


    # Define the input options of Extra-Deep

    input_options = parser.add_argument_group("Input options")

    input_options.add_argument("-i", "--input", action="store", default="nsys", type=str.lower, choices=["nsys", "extradeep", "ncu"], dest="input_format", required=True, help="Specify the input format that Extra-Deep should load the data from. Path needs to be a directory.")
    input_options.add_argument("-ie", "--input-eval", action="store", type=str.lower, choices=["nsys", "extradeep", "ncu"], dest="input_format_eval", required=False, help="Specify the input format that Extra-Deep should load the data from for the evaluation data. Path needs to be a directory.")

    # Define the analysis options of Extra-Deep for epochs, cuda-kernels, etc.

    analysis_options = parser.add_argument_group("Analysis options")

    analysis_options.add_argument("-a", "--analysis", action="store", dest="analysis", type=str.lower, choices=["nvtx", "cuda-kernel", "os", "mpi", "cublas", "cudnn", "memory", "cuda-api", "epochs", "training-steps", "testing-steps", "app-phases", "gpu-util"], required=True, help="Specify the type of analyis that will be performed by Extra-Deep.")

    #analysis_options.add_argument("--batchsize", metavar="BATCH_SIZE", action="store", dest="batch_size", type=str, required=True, help="Specify the batch size for the analysis that was used for the experiments.")

    analysis_options.add_argument("-rp", "--rparameter", metavar="RESOURCE_PARAMETER", action="store", dest="rparam", type=str, required=True, help="Specify name of the parameter that represents the resource allocation in the experiments. Can be either a string to set the parameter name, e.g. p. Or a number if their is no such parameter and set a constant value for the number of mpi ranks.")

    #analysis_options.add_argument("--per", metavar="APPROXIMATION_TYPE", action="store", dest="per", type=str, default="step", choices=["step","epoch"], help="Specify to either analyse the kernels per step or per epoch.")

    analysis_options.add_argument("-p", "--parameter", metavar="PARAMETERS", action="store", dest="param", type=str, required=True, help="Specify the parameter with name (one letter) that are used in the experiments as a comma separated list, e.g., p,n,b. Should be in the same order as in the name of the experiment files!")

    #analysis_options.add_argument("--dtrain", metavar="DTRAIN_SIZE", action="store", dest="data_set_size_train", type=int, required=True, help="Specify the size of the training data set for the analysis that was used for the experiments. Extra-Deep assumes that you divide your training data by the number of mpi ranks, so each worker has an equally large portion to work on.")

    #analysis_options.add_argument("--dval", metavar="DVAL_SIZE", action="store", dest="data_set_size_val", type=int, required=True, help="Specify the size of the validation data set for the analysis that was used for the experiments. Extra-Deep assumes that you are using the full validation data set for testing on each worker.")

    #analysis_options.add_argument("--epoch", action="store", metavar="EPOCH_NUMBER", dest="epoch_for_modeling", default=2, type=int, required=True, help="Set the epoch from which the data will be used for modeling. Should not be the initial epoch since it contains initialization and optimization overheads.")

    analysis_options.add_argument("--speedup", action="store_true", dest="speedup", default=False, help="Additionally analyze the speedup metric of the selected analysis choice.")

    analysis_options.add_argument("--cost", action="store_true", dest="cost", default=False, help="Additionally analyze the cost of the experiments.")

    analysis_options.add_argument("--cpu-cores", action="store", metavar="CPU_CORES_PER_RANK", type=int, dest="cpucores", help="Sets the number of CPU cores used per MPI rank for the cost analysis.")

    analysis_options.add_argument("--efficiency", action="store_true", dest="efficiency", default=False, help="Additionally analyze the parallel efficiency of the selected analysis choice.")


    # Define the options for plotting

    #plotting_options = parser.add_argument_group("Plotting options")

    #plotting_options.add_argument("-p", "--plot", action="store_true", dest="plot", default=False, help="Specify if the modeled functions should be plotted by Extra-Deep.")


    # Define the evaluation options of Extra-Deep for epochs, kernels, etc.
    #INFO: This is for the paper only, in the release version this part has to be removed...

    evaluation_options = parser.add_argument_group("Evaluation options")

    #evaluation_options.add_argument("-e", "--evaluation", action="store", dest="evaluation", type=str.lower, choices=["nvtx", "cuda-kernel", "os", "mpi", "cublas", "cudnn", "memory", "cuda-api", "epochs", "training-steps", "validation-steps", "phases", "gpu-util"], required=False, help="Specify the type of evaluation that will be performed by Extra-Deep. Needs to match with the analysis type.")

    evaluation_options.add_argument("-e", "--evalpath", action="store", metavar="EVAL_DATA_PATH", dest="evalpath", type=str, help="Specify the path to the folder containing the evaluation data.")


    # Define options for modeler

    modeling_options = parser.add_argument_group("Modeling options")
    modeling_options.add_argument("--median", action="store_true", dest="median", help="Use median values for computation instead of mean values.")

    modeling_options.add_argument("--modeler", action="store", dest="modeler", default='default', type=str.lower, choices=modelers_list, help="Selects the modeler for generating the performance models.")

    modeling_options.add_argument("--options", dest="modeler_options", default={}, nargs='+', metavar="KEY=VALUE", action=ModelerOptionsAction, help="Options for the selected modeler")

    modeling_options.add_argument("--help-modeler", choices=modelers_list, type=str.lower, help="Show help for modeler options and exit.", action=ModelerHelpAction)

    modeling_options.add_argument("--minpoints", action="store", metavar="MIN_POINTS_MODELING", dest="minimum_required_points", default=5, help="Set the minimum number of points required for creating a model.")

    modeling_options.add_argument("--strong", action="store_true", dest="strong_scaling", default=False, help="Use strong scaling for modeling and analysis.")


    # Define output options

    output_options = parser.add_argument_group("Output options")
    output_options.add_argument("-o", "--out", action="store", metavar="OUTPUT_PATH", dest="out", help="Specify the output path for Extra-Deep results.")

    output_options.add_argument("--print", action="store", dest="print_type", default="all", choices=["all", "callpaths", "metrics", "parameters", "functions"], help="Set which information should be displayed after modeling (default: all).")

    output_options.add_argument("--save-experiment", action="store", metavar="EXPERIMENT_PATH", dest="save_experiment", help="Saves the experiment including all models as Extra-Deep experiment (if no extension is specified, '.extra-p' is appended).")

    arguments = parser.parse_args(args)


    # set log level
    loglevel = logging.getLevelName(arguments.log_level.upper())
    # set output print type
    printtype = arguments.print_type.upper()

    # set log format location etc.
    if loglevel == logging.DEBUG:
        logging.basicConfig(
            format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s",
            level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p")
    else:
        logging.basicConfig(
            format="%(levelname)s: %(message)s", level=loglevel)

    # set use mean or median for computation
    use_median = arguments.median

    # save modeler output to file?
    print_path = None
    if arguments.out is not None:
        print_output = True
        print_path = arguments.out
    else:
        print_output = False

    # check if an input type is defined for the evaluation part
    input_format_eval = None
    if arguments.input_format_eval == None:
        input_format_eval = arguments.input_format
    else:
        input_format_eval = arguments.input_format_eval

    # load the data from the files
    if arguments.path is not None:

        # load data from extradeep file
        if arguments.input_format == "extradeep":

            if os.path.isdir(arguments.path) == False:

                # check if the input path exists
                if os.path.exists(arguments.path) == True:

                    with ProgressBar(desc='Loading file') as pbar:
                        # load data from extradeep file
                        success, experiment = read_experiment(arguments.path, progress_bar=pbar)
                        if success == False:
                            logging.error("Data could not be read successfully.")
                            sys.exit(1)
                        pbar.total = 100
                        sleep(0.001)
                        for _ in range(89):
                            pbar.update()

                else:
                    logging.error("Input file path does not exist.")
                    sys.exit(1)

            else:
                logging.error("Provided input path is a directory but should be a file.")
                sys.exit(1)

        # load data from nsight systems files
        if arguments.input_format == "nsys":

            # check if path for folder with data is okay
            if os.path.isdir(arguments.path):

                # load nvtx data
                if arguments.analysis == "nvtx":
                    success, experiment = read_nvtx(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load cuda kernel data
                elif arguments.analysis == "cuda-kernel":
                    success, experiment = read_kernel(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load nvtx data
                elif arguments.analysis == "os":
                    success, experiment = read_os(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load mpi data
                elif arguments.analysis == "mpi":
                    success, experiment = read_mpi(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load cublas data
                elif arguments.analysis == "cublas":
                    success, experiment = read_cublas(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load cudnn data
                elif arguments.analysis == "cudnn":
                    success, experiment = read_cudnn(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load memory data
                elif arguments.analysis == "memory":
                    success, experiment = read_memory(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load cuda api data
                elif arguments.analysis == "cuda-api":
                    success, experiment = read_cuda_api(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load training step data
                elif arguments.analysis == "training-steps":
                    success, experiment = read_training_steps(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load epoch data
                elif arguments.analysis == "epochs":
                    success, experiment = read_epochs(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load test step data
                elif arguments.analysis == "testing-steps":
                    success, experiment = read_testing_steps(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                # load data for app phase modeling
                elif arguments.analysis == "app-phases":
                    success, experiment = read_application_phases(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

                ##############################################
                #TODO: finish these...


                elif arguments.analysis == "gpu-util":
                    success, experiment = read_gpu_util(arguments.path, arguments)
                    if success == False:
                        logging.error("Data could not be read successfully.")
                        sys.exit(1)

            else:
                logging.error("The given path is not valid. It must point to a directory.")
                sys.exit(1)

        # load data from nsight systems files
        if arguments.input_format == "ncu":

            # check if path for folder with data is okay
            if os.path.isdir(arguments.path):
                
                #TODO: experimental ncu reader
                experiment, temp = read_ncu_file(arguments.path, arguments, pbar=pbar)

            else:
                logging.error("The given path is not valid. It must point to a directory.")
                sys.exit(1)

        # initialize model generator
        model_generator = ModelGenerator(
            experiment, modeler=arguments.modeler, use_median=use_median)

        # apply modeler options
        modeler = model_generator.modeler
        if isinstance(modeler, MultiParameterModeler) and arguments.modeler_options:
            # set single-parameter modeler of multi-parameter modeler
            single_modeler = arguments.modeler_options[SINGLE_PARAMETER_MODELER_KEY]
            if single_modeler is not None:
                modeler.single_parameter_modeler = single_parameter.all_modelers[single_modeler]()
            # apply options of single-parameter modeler
            if modeler.single_parameter_modeler is not None:
                for name, value in arguments.modeler_options[SINGLE_PARAMETER_OPTIONS_KEY].items():
                    if value is not None:
                        setattr(modeler.single_parameter_modeler, name, value)

        # set attributes for modeler options
        for name, value in arguments.modeler_options.items():
            if value is not None:
                setattr(modeler, name, value)

        # create models from data
        with ProgressBar(desc='Generating models') as pbar:
            model_generator.model_all(pbar)

        # Format the modeler output
        modeler_output_text = format_output(experiment, printtype, arguments.analysis)
        
        # print the output of the modeler to the console
        print(modeler_output_text)

        # Evaluation
        if arguments.evalpath is not None:

            if arguments.analysis == "nvtx":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_nvtx(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "cuda-kernel":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_kernel(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "os":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_os(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "mpi":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_mpi(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "cublas":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_cublas(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "cudnn":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_cudnn(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "memory":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_memory(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "cuda-api":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_cuda_api(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_kernels(eval_experiment, experiment, arguments)

            if arguments.analysis == "training-steps":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_training_steps(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate(eval_experiment, experiment, arguments, "")

            if arguments.analysis == "epochs":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_epochs(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    if os.path.isdir(arguments.evalpath) == False:
                        if os.path.exists(arguments.evalpath) == True:
                            success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                            pbar.total = 100
                            sleep(0.001)
                            for _ in range(91):
                                pbar.update()
                        else:
                            logging.error("Input file path does not exist.")
                            sys.exit(1)
                    else:
                        logging.error("Provided input path is a directory but should be a file.")
                        sys.exit(1)
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate(eval_experiment, experiment, arguments, "")

            if arguments.analysis == "testing-steps":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_testing_steps(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate(eval_experiment, experiment, arguments, "")

            if arguments.analysis == "app-phases":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_application_phases(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                #eval_output_text = evaluate_phases(eval_experiment, experiment, arguments, application_name)
                eval_output_text = evaluate(eval_experiment, experiment, arguments, "")


            #######################################################
            #TODO: finish these...

            if arguments.analysis == "gpu-util":
                if input_format_eval == "nsys":
                    if os.path.isdir(arguments.evalpath):
                        success, eval_experiment = read_gpu_util_eval(arguments.evalpath, arguments)
                    else:
                        logging.error("The given path is not valid. It must point to a directory.")
                        sys.exit(1)
                elif input_format_eval == "extradeep":
                    success, eval_experiment = read_experiment(arguments.evalpath, pbar)
                    pbar.total = 100
                    sleep(0.001)
                    for _ in range(91):
                        pbar.update()
                if success == False:
                    logging.error("Data could not be read successfully.")
                    sys.exit(1)
                eval_output_text = evaluate_gpu_util(eval_experiment, experiment, arguments)

        # save formatted output to text file
        if print_output:
            text = text +"\n"+ eval_output_text
            save_output(text, print_path)

        # save the model and eval experiments
        if arguments.save_experiment:
            try:
                with ProgressBar(desc='Saving experiment') as pbar:
                    if not os.path.splitext(arguments.save_experiment)[1]:
                        save_path1 = arguments.save_experiment + '_model.extradeep'
                        experiment_io.write_experiment(experiment, save_path1, pbar)
                        if arguments.evalpath != None:
                            save_path2 = arguments.save_experiment + '_eval.extradeep'
                            experiment_io.write_experiment(eval_experiment, save_path2, pbar)

            except RecoverableError as err:
                logging.error('Saving experiment: ' + str(err))
                sys.exit(1)

    else:
        logging.error("No file path given to load files.")
        sys.exit(1)


if __name__ == '__main__':
    main()
