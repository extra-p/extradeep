# This file is part of the Extra-Deep software (https://github.com/extra-deep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class OSEvent():
    """
    Class for a OSEvent events.
    """

    def __init__(self, start, end, event_class, global_timer_id, correlation_id, name, return_value, nesting_level, callchain_id, duration, start_seconds, end_seconds, duration_seconds):
        """
        __init__ function to initalize a OSEvent object
        """

        self.start = start
        self.end = end
        self.event_class = event_class
        self.global_timer_id = global_timer_id
        self.correlation_id = correlation_id
        self.name = name
        self.return_value = return_value
        self.nesting_level = nesting_level
        self.callchain_id = callchain_id
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.duration = duration
        self.duration_seconds = duration_seconds
