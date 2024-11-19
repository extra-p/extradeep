# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class CuptiMemcopy():
    """
    Class for a cupti memcopy event.
    """

    def __init__(self, start, end, duration, start_seconds, end_seconds, duration_seconds, device_id, context_id, stream_id,
    correlation_id, global_pid, bytes, copy_kind, depracted_src_id, src_kind, dst_kind, src_device_id, src_context_id, dst_device_id,
    dst_context_id, migration_cause, graph_node_id):
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
        self.bytes = bytes

        self.copy_kind = copy_kind

        if self.copy_kind == 0:
            self.copy_kind = "The memory copy kind is not known."
        elif self.copy_kind == 1:
            self.copy_kind = "A host to device memory copy."
        elif self.copy_kind == 2:
            self.copy_kind = "A device to host memory copy."
        elif self.copy_kind == 3:
            self.copy_kind = "A host to device array memory copy."
        elif self.copy_kind == 4:
            self.copy_kind = "A device array to host memory copy."
        elif self.copy_kind == 5:
            self.copy_kind = "A device array to device array memory copy."
        elif self.copy_kind == 6:
            self.copy_kind = "A device array to device memory copy."
        elif self.copy_kind == 7:
            self.copy_kind = "A device to device array memory copy."
        elif self.copy_kind == 8:
            self.copy_kind = "A device to device memory copy on the same device."
        elif self.copy_kind == 9:
            self.copy_kind = "A host to host memory copy."
        elif self.copy_kind == 10:
            self.copy_kind = "A peer to peer memory copy across different devices."
        else:
            self.copy_kind = "The memory copy kind is not known."

        self.depracted_src_id = depracted_src_id

        self.src_kind = src_kind

        if self.src_kind == 0:
            self.src_kind = "The memory kind is unknown."
        elif self.src_kind == 1:
            self.src_kind = "The memory is pageable."
        elif self.src_kind == 2:
            self.src_kind = "The memory is pinned."
        elif self.src_kind == 3:
            self.src_kind = "The memory is on the device."
        elif self.src_kind == 4:
            self.src_kind = "The memory is an array."
        elif self.src_kind == 5:
            self.src_kind = "The memory is managed"
        elif self.src_kind == 6:
            self.src_kind = "The memory is device static"
        elif self.src_kind == 7:
            self.src_kind = "The memory is managed static"
        else:
            self.src_kind == "The memory kind is unknown."

        self.dst_kind = dst_kind

        if self.dst_kind == 0:
            self.dst_kind = "The memory kind is unknown."
        elif self.dst_kind == 1:
            self.dst_kind = "The memory is pageable."
        elif self.dst_kind == 2:
            self.dst_kind = "The memory is pinned."
        elif self.dst_kind == 3:
            self.dst_kind = "The memory is on the device."
        elif self.dst_kind == 4:
            self.dst_kind = "The memory is an array."
        elif self.dst_kind == 5:
            self.dst_kind = "The memory is managed"
        elif self.dst_kind == 6:
            self.dst_kind = "The memory is device static"
        elif self.dst_kind == 7:
            self.dst_kind = "The memory is managed static"
        else:
            self.dst_kind == "The memory kind is unknown."

        self.src_device_id = src_device_id
        self.src_context_id = src_context_id
        self.dst_device_id = dst_device_id
        self.dst_context_id = dst_context_id
        self.migration_cause = migration_cause
        self.graph_node_id = graph_node_id
