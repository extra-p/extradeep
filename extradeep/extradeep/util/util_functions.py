from os import listdir
from os.path import isfile, join
import math
from tabulate import tabulate
import pandas as pd
import numpy as np

def remove_outliers(data):
    """
    remove_outliers function that identifies and removes the outliers in a list of metric values.

    :param data: the list containing the measurement values
    :return data_new: the list containing the data set with the outliers removed
    """

    df = pd.DataFrame (data, columns = ["data"])
    # outlier detection with IQR
    Q1 = np.percentile(df['data'], 25,
                    interpolation = 'midpoint')
    Q3 = np.percentile(df['data'], 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    # create new list without the outliers
    data_new = []
    for i in range(len(data)):
        if data[i] >= (Q1-1.5*IQR) and data[i] <= (Q3+1.5*IQR):
            data_new.append(data[i])
    return data_new


def get_outliers(data):
    """
    get_outliers function that identifies the outliers in a list of metric values.

    :param data: the list containing the measurement values
    :return data_new: the list of ids of the outliers in the list
    """

    df = pd.DataFrame (data, columns = ["data"])
    # outlier detection with IQR
    Q1 = np.percentile(df['data'], 25,
                    interpolation = 'midpoint')
    Q3 = np.percentile(df['data'], 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    # create new list without the outliers
    data_ids = []
    for i in range(len(data)):
        if data[i] >= (Q1-1.5*IQR) and data[i] <= (Q3+1.5*IQR):
            data_ids.append(i)
    return data_ids
    

def get_tabulate_string(text):
    """
    get_tabulate_string function that creates and returns a string of the input text surronded by a table of lines for console output.

    :param text: the input text that should be surrounded
    :return output: the output string
    """

    table = [[text]]
    output = tabulate(table, tablefmt='grid')
    return output

def get_sqlite_files_in_path(folder):
    """
    get_sqlite_files_in_path function that returns all paths of the
    files in the given folder that are sqlite databases.

    :param folder: the path to the folder containing the experiment data
    :return paths: list of paths of all sqlite files found
    :return files: list of files found (filename only)
    """

    # get the files in the folder
    allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    # only consider .sqlite data type files
    files = []
    for value in enumerate(allfiles):
        if value[1].find(".sqlite") != -1:
            files.append(value[1])
    files.sort()

    # construct the filepaths
    paths = []
    for value in enumerate(files):
        paths.append(join(folder, value[1]))

    return paths, files

def getRGBfromI(RGBint):
    """
    getRGBfromI Get the rgb color as red, green, and blue integer
    values from a single value integer color

    :param RGBint: rgb single integer color value
    :return red, gree, blue: integer values for the rgb values
    """
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue

def clearLayout(layout):
    """
    clearLayout function clears all widgets from a layout

    :param layout: the layout from which the widgets should be cleared
    """

    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()

def get_unique_numbers(numbers):
    """
    get_unique_numbers function to return the unique numbers in a list

    :param numbers: a list of numbers to check the unique values in
    :return list_of_unique_numbers: a list containing all unique numbers found in the given list
    """

    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers

def convert_to_py_function(function):
    """
    convert_to_py_function function that converts a extrap function to a python string based style function

    :param function: the extrap style function as a string
    :return temp: a string of the python style function
    """

    temp = ""
    temp = function.replace(" ", "")
    temp = temp.replace("^", "**")
    temp = temp.replace("log2", "math.log2")
    temp = temp.replace("+-", "-")
    return temp
