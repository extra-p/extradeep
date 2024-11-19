import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from extradeep.util.util_functions import convert_to_py_function
import math
from matplotlib.widgets import TextBox

def plot_epochs_runtime(experiment):
    """
    plot_epochs_runtime function to plot the training time per epoch as a function of the parameters. Also plots the mean and median values of the measurement points.

    :param experiment: the experiment containing the data for plotting
    """

    if len(experiment.parameters) == 1:

        #coordinates = experiment.coordinates
        parameter = experiment.parameters[0]
        callpath = experiment.callpaths[0]
        metric = experiment.metrics[0]
        modeler = experiment.modelers[0]

        try:
            model = modeler.models[callpath, metric]
        except KeyError as e:
            model = None
        if model != None:
            hypothesis = model.hypothesis
            function = hypothesis.function
            rss = hypothesis.RSS
            ar2 = hypothesis.AR2
            function_string = function.to_string(*experiment.parameters)
            function = convert_to_py_function(function_string)

        x_min = 1
        x_max = None
        y_min = 0
        y_max = None

        # get the values of the measurement points for plotting
        x = []
        y = []
        x2 = []
        y2 = []
        for callpath_id in range(len(experiment.callpaths)):
            callpath = experiment.callpaths[callpath_id]
            for metric_id in range(len(experiment.metrics)):
                metric = experiment.metrics[metric_id]
                for coordinate_id in range(len(experiment.coordinates)):
                    coordinate = experiment.coordinates[coordinate_id]
                    dimensions = coordinate.dimensions
                    values = []
                    for dimension in range(dimensions):
                        value = coordinate[dimension]
                        values.append(value)
                    measurement = experiment.get_measurement(coordinate_id, callpath_id, metric_id)
                    if measurement == None:
                        value_mean = 0
                        value_median = 0
                    else:
                        value_mean = measurement.mean
                        value_median = measurement.median
                        x.append(values[0])
                        y.append(value_mean)
                        x2.append(values[0])
                        y2.append(value_median)


        temp = x[1] - x[0]
        x_max = max(x)+temp*2
        temp = max(y)+(10*(max(y)/100))
        y_max = temp

        # basically the resolution of the plot
        steps = 1000
        step_size = (x_max-x_min) / steps

        # get the values for plotting of the model
        p = x_min
        runtimes = []
        ps = []
        for i in range(steps):
            runtime = eval(function)
            runtimes.append(runtime)
            ps.append(p)
            p += step_size

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.plot(ps, runtimes)
        ax.plot(x, y, marker="o", markerfacecolor="black", markeredgecolor="black", linestyle = 'None', markersize=3)
        ax.plot(x2, y2, marker="_", markerfacecolor="red", markeredgecolor="red", linestyle = 'None', markersize=5)
        ax.set_ylabel("Training time t")
        ax.set_xlabel(parameter)
        ax.set_title("Training time per epoch")

        def submit(text):
            x_max = float(text)

            temp = max(y)+(10*(max(y)/100))
            y_max = temp

            # basically the resolution of the plot
            step_size = (x_max-x_min) / steps

            # get the values for plotting of the model
            p = x_min
            runtimes = []
            ps = []
            for i in range(steps):
                runtime = eval(function)
                runtimes.append(runtime)
                ps.append(p)
                p += step_size

            ax.clear()

            ax.plot(ps, runtimes)
            ax.plot(x, y, marker="o", markerfacecolor="black", markeredgecolor="black", linestyle = 'None', markersize=3)
            ax.plot(x2, y2, marker="_", markerfacecolor="red", markeredgecolor="red", linestyle = 'None', markersize=5)
            ax.set_ylabel("Training time t")
            ax.set_xlabel(parameter)
            ax.set_title("Training time per epoch")

            #l.set_ydata(runtimes)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.relim()
            ax.autoscale_view()
            plt.draw()

        axbox = fig.add_axes([0.1, 0.01, 0.15, 0.045])
        text_box = TextBox(axbox, str(parameter)+" max: ", initial=str(x_max))
        text_box.on_submit(submit)

        plt.show()
