# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class CuptiKernel():
    """
    Class for a cupti kernel.
    """

    def __init__(self, start_time_long, end_time_long, duration_long, start_time_seconds, end_time_seconds, duration_seconds, demangledName, shortName, grid, block, staticSharedMemory, dynamicSharedMemory, sharedMemoryExecuted, registersPerThread, localMemoryTotal, localMemoryPerThread):
        """
        __init__ function to initalize a CuptiKernel object
        """

        self.start_time_long = start_time_long
        self.end_time_long = end_time_long
        self.duration_long = duration_long
        self.start_time_seconds = start_time_seconds
        self.end_time_seconds = end_time_seconds
        self.duration_seconds = duration_seconds
        self.demangledName = demangledName
        self.shortName = shortName
        self.grid = grid
        self.block = block
        self.staticSharedMemory = staticSharedMemory
        self.dynamicSharedMemory = dynamicSharedMemory
        self.sharedMemoryExecuted = sharedMemoryExecuted
        self.registersPerThread = registersPerThread
        self.localMemoryTotal = localMemoryTotal
        self.localMemoryPerThread = localMemoryPerThread
