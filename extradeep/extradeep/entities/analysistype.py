# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import itertools

from extradeep.util.serialization_schema import make_value_schema


class AnalysisType:
    """
    This class represents a AnalysisType.
    """
    """
    Counter for global metric ids
    """
    ID_COUNTER = itertools.count()

    def __init__(self, name):
        """
        Initializes the AnalysisType object.
        """
        self.name = name
        self.id = next(AnalysisType.ID_COUNTER)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, AnalysisType):
            return NotImplemented
        return self is other or self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"AnalysisType({self.name})"


AnalysisTypeSchema = make_value_schema(AnalysisType, 'name')
