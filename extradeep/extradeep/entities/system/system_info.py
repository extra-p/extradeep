# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class SystemInfo():
    """
    Class that holds the information of the system the measurements have been taken on.
    """

    def __init__(self, cpu_cores):
        """
        __init__ function initalizes a ExperimentConfig object

        :param cpu_cores: the number of CPU cores available per process/rank
        """

        self.cpu_cores = cpu_cores

   