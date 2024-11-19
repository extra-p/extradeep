# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class ExperimentConfig():
    """
    Class that holds the config information of a performance experiment.
    """

    def __init__(self, id, app_name, repetition_nr, mpi_rank, parameter_names, parameter_values, filename):
        """
        __init__ function initalizes a ExperimentConfig object

        :param app_name: the name of the application as a string
        :param repetition_nr: the number of the repetition of this configuration as integer
        :param mpi_rank: the number of the mpi rank as integer
        :param parameter_names: the parameter names of the configuration as string list
        :param parameter_values: the parameter values of the configuration as float list
        :param filename: the filename of the file the config was read from as a string
        """

        self.id = id
        self.app_name = app_name
        self.repetition_nr = repetition_nr
        self.mpi_rank = mpi_rank
        self.parameter_names = parameter_names
        self.parameter_values = parameter_values
        self.filename = filename

    def __str__(self):
        """
        __str__ function to return the content of a ExperimentConfig object as a string.

        :return _: string value of object content.
        """

        return "DEBUG print ExperimentConfig: app name=%s, repetition nr=%s, mpi rank=%s, parameter names=%s, parameter values=%s."%(self.app_name, self.repetition_nr, self.mpi_rank, self.parameter_names, self.parameter_values)
