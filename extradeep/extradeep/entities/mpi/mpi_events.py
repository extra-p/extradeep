# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class MPIEvent():
    """
    Class for a MPIEvent events.
    """

    def __init__(self, start, end, name, duration, start_seconds, end_seconds, duration_seconds):
        """
        __init__ function to initalize a MPIEvent object
        """

        self.start = start
        self.end = end
        self.name = name
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.duration = duration
        self.duration_seconds = duration_seconds
