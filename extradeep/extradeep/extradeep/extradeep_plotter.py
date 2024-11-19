# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.


import argparse
import logging
import os
import sys
import pickle

from pyparsing import alphas
import extradeep
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np


def load_object(path):
    """
    load_object function to load an object using pickle.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def main(args=None, prog=None):
    """
    main function that runs the extradeep plotter

    :param args: parameter for arguments entered in the terminal
    :param prog: parameter for programm
    """

    # Define argparse commands for input, output operations
    parser = argparse.ArgumentParser(prog=prog, description="Extra-Deep plotter.", add_help=False)
    positional_arguments = parser.add_argument_group("Positional arguments")

    # Define basic program arguements such as log, help, and version outputs
    basic_arguments = parser.add_argument_group("Optional arguments")
    basic_arguments.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
    basic_arguments.add_argument("-v", "--version", action="version", version=extradeep.__title__ + " " + extradeep.__version__, help="Show program's version number and exit.")
    basic_arguments.add_argument("--log", action="store", dest="log_level", type=str.lower, default='warning', choices=['debug', 'info', 'warning', 'error', 'critical'], help="Set program's log level (default: warning).")

    # Define the path argument that points to the folder the data will be loaded from
    positional_arguments.add_argument("path", metavar="FILE_PATH", type=str, action="store",
                                      help="Specify the path of the file to read the data from.")
    positional_arguments.add_argument('--out', dest='out', action='store',type=str, default="instrumented_code",
                                      help='Set the output for the plot to be saved.')

    # Define the analysis options of Extra-Deep for epochs, cuda-kernels, etc.

    analysis_options = parser.add_argument_group("Analysis options")

    analysis_options.add_argument("-a", "--analysis", action="store", dest="analysis", type=str.lower, choices=["nvtx", "cuda-kernel", "os", "mpi", "cublas", "cudnn", "memory", "cuda-api", "epochs", "training-steps", "validation-steps", "phases", "gpu-util"], required=True, help="Specify the type of analyis that will be performed by Extra-Deep.")


    # parse args
    arguments = parser.parse_args(args)

    # set log level
    loglevel = logging.getLevelName(arguments.log_level.upper())

    # set log format location etc.
    if loglevel == logging.DEBUG:
        logging.basicConfig(
            format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s",
            level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p")
    else:
        logging.basicConfig(
            format="%(levelname)s: %(message)s", level=loglevel)

    # load the data from the files
    if arguments.path is not None:

        # check if path for folder with data is okay
        if os.path.exists(arguments.path):

            # load the data from the pickle file
            data_object = load_object(arguments.path)

            # unpack it depending on what type of data it is
            if arguments.analysis == "training-steps":

                model_function = data_object[0]
                data_frame = data_object[1]

                print(model_function)
                print(data_frame)

                plot(model_function, data_frame)

            # unpack it depending on what type of data it is
            elif arguments.analysis == "epochs":

                model_function = data_object[0]
                data_frames = data_object[1]

                #print(model_function)
                #print(data_frame)

                for key, value in enumerate(data_frames):
                    plot(model_function, data_frames[value], value)
            
        else:
            logging.error("File does not exist.")
            sys.exit(1)
    else:
        logging.error("There was no proper file path provided.")
        sys.exit(1)


def plot(model_function, data_frame, metric_name):

    # column width for size for double column ieee conference style paper
    #516.0pt
    # height calculated with golden ratio
    #252.0pt

    figsize = defines()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.grid(True, color="lightgray", linestyle=':', linewidth=0.5)
    ax.set_ylabel("Training time\n per epoch [sec]")
    ax.set_xlabel("Number of MPI ranks $x_1$")

    # 100 linearly spaced numbers
    x = np.linspace(1,72,100)
    #print(model_function)
    # the function, which is y = x^2 here
    y = []
    for i in range(len(x)):
        p = x[i]
        result = eval(model_function)
        y.append(result)

    x2 = []
    actuals = []
    predictions = []
    error_percent = []
    minimum = []
    maximum = []

    for i in range(len(data_frame)):
        x2.append(data_frame.loc[i][1])
        actuals.append(data_frame.loc[i][2])
        predictions.append(data_frame.loc[i][3])
        minimum.append(data_frame.loc[i][6])
        maximum.append(data_frame.loc[i][7])
        er = '%.1f'%(data_frame.loc[i][5])
        er = er+"\%"
        error_percent.append(er)

    div_perc = []
    for i in range(len(minimum)):
        div_perc.append(maximum[i] - minimum[i])

    ci = 1.960 * ( np.std(predictions) / np.sqrt(len(predictions)) )

    #ax.set_xscale('log')
    #ax.set_xticks(x2)
    ax.set_xticks([2,4,6,10,14,18,24,32,40,48,56,64])
    #ax.set_xticklabels(['zero','two','four','six'])
    ax.set(xlim=(1, x2[len(x2)-1]+1))

    # vertical line separator between modeling and evaluation measurements
    plt.axvline(x = 13, color = "gray", linestyle = '--', lw = 0.5)
    plt.text(12, 550, "Modeling points $P$", color="gray", ha="center", va="center", size=5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
    plt.text(52, 170, "Evaluation points $P^{+}$", color="gray", ha="center", va="center", size=5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

    # plot the 95% confidence interval
    plt.fill_between(x, (y-ci), (y+ci), color="mistyrose", alpha=0.75, label="95\% CI")

    # plot the function
    plt.plot(x,y, 'r', label="model")

    # plot the actual runtimes for the training steps
    plt.plot(x2, predictions, marker="o", markersize=4, markeredgecolor="black", markeredgewidth=0.1, markerfacecolor="red", linestyle = 'None', label="prediction")

    # plot the actual runtimes for the training steps
    plt.errorbar(x2, actuals, yerr=div_perc, ecolor='black', elinewidth=0.5, capsize=3, marker="^", markersize=4, markeredgecolor="black", markeredgewidth=0.1, markerfacecolor="blue", linestyle = 'None', label="measured")

    # Loop for annotation of all points
    arrowprops = dict(arrowstyle="-", connectionstyle="angle3,angleA=0,angleB=-100", lw=0.5, color="gray")
    for i in range(len(x2)):
        #plt.annotate(error_percent[i], (x2[i] - 0.2 , predictions[i] + 0.1), fontsize=6)
        plt.annotate(error_percent[i], fontsize=6, rotation=45, xy=(x2[i], predictions[i]), xycoords='data', xytext=(+1, +10), textcoords='offset points', arrowprops=arrowprops)

    # Add a legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        handletextpad=0.25,
        columnspacing=0.8,
        frameon=False,
    )

    pdf = PdfPages("plot_"+str(metric_name)+".pdf")
    pdf.savefig(fig, bbox_inches='tight')
    pdf.close()


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    #fig_height_in = fig_width_in * golden_ratio
    fig_height_in = (100 * fraction) * inches_per_pt

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def defines():
    
    plt.style.use('default')
    #sns.set_context("paper", font_scale=1)
    #sns.set_theme(style="ticks")

    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })

    # in pt
    textwidth = 516.0
    columnwidth = 252.0

    # compute the figuresize
    figsize = set_size(columnwidth)
    return figsize


if __name__ == '__main__':
    main()
