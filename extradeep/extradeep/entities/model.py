# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import Optional, List

import numpy
from marshmallow import fields, post_load
from extradeep.entities import analysistype

from extradeep.entities.callpath import CallpathSchema
from extradeep.entities.hypotheses import Hypothesis, HypothesisSchema
from extradeep.entities.measurement import Measurement
from extradeep.entities.metric import MetricSchema
from extradeep.entities.analysistype import AnalysisTypeSchema
from extradeep.util.caching import cached_property
from extradeep.util.serialization_schema import Schema


class Model:

    def __init__(self, hypothesis, callpath=None, metric=None, analysistype=None):
        self.hypothesis: Hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric
        self.analysistype = analysistype
        self.measurements: Optional[List[Measurement]] = None

    @cached_property
    def predictions(self):
        coordinates = numpy.array([m.coordinate for m in self.measurements])
        return self.hypothesis.function.evaluate(coordinates.transpose())

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.callpath == other.callpath and \
                   self.metric == other.metric and \
                   self.analysistype == other.analysistype and \
                   self.hypothesis == other.hypothesis and \
                   self.measurements == other.measurements


class ModelSchema(Schema):
    def create_object(self):
        return Model(None)

    @post_load
    def report_progress(self, data, **kwargs):
        if 'progress_bar' in self.context:
            self.context['progress_bar'].update()
        return data

    hypothesis = fields.Nested(HypothesisSchema)
    callpath = fields.Nested(CallpathSchema)
    metric = fields.Nested(MetricSchema)
    analysistype = fields.Nested(AnalysisTypeSchema)
