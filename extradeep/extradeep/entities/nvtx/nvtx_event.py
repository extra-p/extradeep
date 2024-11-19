# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extradeep.util.util_functions import *

class NVTX_Event():
    """
    Class that defines an NVTX event and holds all necessary data for the analysis.
    """

    def __init__(self, start_time, end_time, text, color, text_id, domain_id, event_type, range_id, category, global_tid, string_value="None"):
        """
        __init__ function to initialize a NVTX event object.

        :param start_time: the start time of the event as a long value
        :param end_time: the end time of the event as long value
        :param text: the string name of the event
        :param color: the rgb color of the event
        :param text_id: the textid of the event when it has no name it can be found in the string table
        :param domain_id: the domain id of the event as an int
        :param event_type: the type of the event as an int
        :param range_id: the range id of the event as an int
        :param category: the category of the event as an int
        :param global_tid: the gloabal timer id of the event
        :param string_value: (optional) the name of the event from the string value table when it does not have a text entry
        """

        self.start_time_long = int(start_time)
        self.start_time_seconds = self.start_time_long / 1000000000 # convert time stamps from ns to sec
        self.end_time_long = int(end_time)
        self.end_time_seconds = self.end_time_long / 1000000000 # convert time stamps from ns to sec
        self.run_time_long = self.end_time_long - self.start_time_long
        self.run_time_seconds = self.end_time_seconds - self.start_time_seconds

        if text == None:
            self.callpath_name = str(string_value)
        else:
            self.callpath_name = str(text)

        if color == None:
            self.color = getRGBfromI(4278190080)
        else:
            self.color = getRGBfromI(color)

        self.text_id = text_id
        self.domain_id = domain_id
        self.event_type = event_type
        self.range_id = range_id
        self.category = category
        self.global_tid = global_tid

    def __str__(self):
        """
        __str__ function to return the content of a NVTX event object as a string.

        :return _: string value of object content.
        """

        return "NVTX event: start time long=%s, start time seconds=%s, end time long=%s, end time seconds=%s, callpath name=%s, color=%s, text id=%s, domain id=%s, event type=%s, range id=%s, category=%s, global timer id=%s."%(self.start_time_long, self.start_time_seconds, self.end_time_long, self.end_time_seconds, self.callpath_name, self.color, self.text_id, self.domain_id, self.event_type, self.range_id, self.category, self.global_tid)
