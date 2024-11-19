# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class CuptiMemset():
    """
    Class for a cupti memset event.
    """

    def __init__(self, start, end, duration, start_seconds, end_seconds, duration_seconds, device_id, context_id,
    stream_id, correlation_id, global_pid, value, bytes, graph_node_ide, mem_kind):
        """
        __init__ function to initalize a CuptiMemset object
        """

        self.start = start
        self.end = end
        self.duration = duration
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.duration_seconds = duration_seconds
        self.device_id = device_id
        self.context_id = context_id
        self.stream_id = stream_id
        self.correlation_id = correlation_id
        self.global_pid = global_pid
        self.value = value
        self.bytes = bytes
        self.graph_node_ide = graph_node_ide
        self.mem_kind = mem_kind

        if self.mem_kind == 0:
            self.mem_kind = "The memory kind is unknown."
        elif self.mem_kind == 1:
            self.mem_kind = "The memory is pageable."
        elif self.mem_kind == 2:
            self.mem_kind = "The memory is pinned."
        elif self.mem_kind == 3:
            self.mem_kind = "The memory is on the device."
        elif self.mem_kind == 4:
            self.mem_kind = "The memory is an array."
        elif self.mem_kind == 5:
            self.mem_kind = "The memory is managed"
        elif self.mem_kind == 6:
            self.mem_kind = "The memory is device static"
        elif self.mem_kind == 7:
            self.mem_kind = "The memory is managed static"
        else:
            self.mem_kind == "The memory kind is unknown."
