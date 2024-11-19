from enum import Enum

class StatisticalValueType(Enum):
    """
    Class to decide which statistical value is used by extra-deep for modeling.
    Can be mean, median, or geometric mean.
    """

    MEAN = 1
    MEDIAN = 2
    GEOMETRIC_MEAN = 3
