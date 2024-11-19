# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class CuptiSynchronization():
    """
    Class for a cupti synchronization event.
    """

    def __init__(self, start, end, device_id, context_id, stream_id, correlation_id, global_pid, sync_type, event_id, start_seconds, end_seconds, duration, duration_seconds):
        """
        __init__ function to initalize a CuptiSynchronization object
        """

        self.start = start
        self.end = end
        self.device_id = device_id 
        self.context_id = context_id
        self.stream_id = stream_id
        self.correlation_id = correlation_id
        self.global_pid = global_pid
        self.event_id = event_id
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.duration =  duration
        self.duration_seconds = duration_seconds
        self.sync_type = sync_type

        #https://docs.nvidia.com/cupti/modules.html#group__CUPTI__ACTIVITY__API_1g80e1eb47615e31021f574df8ebbe5d9a

        if self.sync_type == 0:
            self.sync_type = "Unknown data."
        elif self.sync_type == 1:
            self.sync_type = "Event synchronize API."
        elif self.sync_type == 2:
            self.sync_type = "Stream wait event API."
        elif self.sync_type == 3:
            self.sync_type = "Stream synchronize API."
        elif self.sync_type == 4:
            self.sync_type = "Context synchronize API."
        elif self.sync_type == 0x7fffffff:
            self.sync_type = ""
        else:
            self.sync_type == "Unknown data."
        