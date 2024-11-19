# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import warnings
from itertools import chain
from typing import List, Dict, Tuple

from marshmallow import fields, validate, pre_load
from packaging.version import Version

import extradeep
from extradeep.entities.analysistype import AnalysisType, AnalysisTypeSchema
from extradeep.entities.callpath import Callpath, CallpathSchema
from extradeep.entities.calltree import CallTree, CallTreeSchema
from extradeep.entities.coordinate import Coordinate, CoordinateSchema
from extradeep.entities.measurement import Measurement, MeasurementSchema
from extradeep.entities.metric import Metric, MetricSchema
from extradeep.entities.parameter import Parameter, ParameterSchema
from extradeep.fileio import io_helper
from extradeep.modelers.model_generator import ModelGenerator, ModelGeneratorSchema
from extradeep.util.deprecation import deprecated
from extradeep.util.progress_bar import DUMMY_PROGRESS
from extradeep.util.serialization_schema import Schema, TupleKeyDict
from extradeep.util.unique_list import UniqueList


class Experiment:

    def __init__(self):
        self.callpaths: Dict[AnalysisType, List[Callpath]] = {}
        self.metrics: Dict[AnalysisType, List[Metric]] = {}
        self.parameters: List[Parameter] = UniqueList()
        self.coordinates: List[Coordinate] = UniqueList()
        self.measurements: Dict[Tuple[Callpath, Metric, AnalysisType], List[Measurement]] = {}
        self.call_trees: Dict[AnalysisType, CallTree] = {}
        self.modelers: List[ModelGenerator] = []
        self.scaling = None
        self.batch_size = None
        self.data_set_size = None
        self.analysistypes: List[AnalysisType] = []

    def add_analysistype(self, analysistype):
        self.analysistypes.append(analysistype)

    def add_modeler(self, modeler):
        self.modelers.append(modeler)

    def add_metric(self, analysistype: AnalysisType, metric_list):
        self.metrics[analysistype] = metric_list

    def add_parameter(self, parameter: Parameter):
        self.parameters.append(parameter)

    def add_coordinate(self, coordinate):
        self.coordinates.append(coordinate)

    def add_callpath(self, analysistype: AnalysisType, callpath_list):
        self.callpaths[analysistype] = callpath_list

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_data_set_size(self, data_set_size):
        self.data_set_size = data_set_size

    @deprecated("Use property directly.")
    def get_measurement(self, coordinate_id, callpath_id, metric_id, analysistype):
        analysistype_id = None
        for i in range(len(self.analysistypes)):
            if self.analysistypes[i] == analysistype:
                analysistype_id = i
                break
        callpaths = self.callpaths[analysistype]
        callpath = callpaths[callpath_id]
        metrics = self.metrics[analysistype]
        metric = metrics[metric_id]
        coordinate = self.coordinates[coordinate_id]
        analysistype = self.analysistypes[analysistype_id]

        try:
            measurements = self.measurements[(callpath, metric, analysistype)]
        except KeyError as e:
            return None

        for measurement in measurements:
            if measurement.coordinate == coordinate:
                return measurement
        return None

    def add_measurement(self, measurement: Measurement):
        key = (measurement.callpath,
               measurement.metric,
               measurement.analysistype)
        if key in self.measurements:
            self.measurements[key].append(measurement)
        else:
            self.measurements[key] = [measurement]

    def delete_measurement(self, callpath, metric, analysistype):
        key = (callpath,
               metric,
               analysistype)
        print(self.measurements)
        self.measurements.pop(key)

    def clear_measurements(self):
        self.measurements = {}

    def debug(self):
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return
        #for i in range(len(self.metrics)):
        #    logging.debug("Metric " + str(i + 1) + ": " + self.metrics[i].name)
        for key, value in self.metrics.items():
            for i in range(len(value)):
                logging.debug("Metric " + str(i + 1) + ": " + value[i].name)
        for i in range(len(self.analysistypes)):
            logging.debug("Analysis Type "+str(i+1)+": "+self.analysistypes[i].name)
        for i in range(len(self.parameters)):
            logging.debug("Parameter " + str(i + 1) + ": " +
                          self.parameters[i].name)
        #for i in range(len(self.callpaths)):
        #    logging.debug("Callpath " + str(i + 1) + ": " +
        #                  self.callpaths[i].name)
        for key, value in self.callpaths.items():
            for i in range(len(value)):
                logging.debug("Callpath " + str(i + 1) + ": " +
                          value[i].name)
        for i, coordinate in enumerate(self.coordinates):
            logging.debug(f"Coordinate {i + 1}: {coordinate}")
        for i, measurement in enumerate(chain.from_iterable(self.measurements.values())):
            callpath = measurement.callpath
            metric = measurement.metric
            analysistype = measurement.analysis_type
            coordinate = measurement.coordinate
            value_mean = measurement.mean
            value_median = measurement.median
            logging.debug(
                f"Measurement {i}: {metric}, {callpath}, {analysistype}, {coordinate}: {value_mean} (mean), {value_median} (median)")


class ExperimentSchema(Schema):
    _version_ = fields.Constant(extradeep.__version__, data_key=extradeep.__title__)
    scaling = fields.Str(required=False, allow_none=True, validate=validate.OneOf(['strong', 'weak']))
    parameters = fields.List(fields.Nested(ParameterSchema))
    measurements = TupleKeyDict(keys=(fields.Nested(CallpathSchema), fields.Nested(MetricSchema), fields.Nested(AnalysisTypeSchema)),
                                values=fields.List(fields.Nested(MeasurementSchema, exclude=('callpath', 'metric', 'analysistype'))))
    metrics = fields.Dict(keys=fields.Nested(AnalysisTypeSchema), values=fields.List(fields.Nested(MetricSchema)))
    callpaths = fields.Dict(keys=fields.Nested(AnalysisTypeSchema), values=fields.List(fields.Nested(CallpathSchema)))
    modelers = fields.List(fields.Nested(ModelGeneratorSchema), missing=[], required=False)
    coordinates = fields.List(fields.Nested(CoordinateSchema))
    call_trees = fields.Dict(keys=fields.Nested(AnalysisTypeSchema), values=fields.Nested(CallTreeSchema))
    analysistypes = fields.List(fields.Nested(AnalysisTypeSchema))

    def set_progress_bar(self, pbar):
        self.context['progress_bar'] = pbar

    @pre_load
    def add_progress(self, data, **kwargs):
        file_version = data.get(extradeep.__title__)
        if file_version:
            prog_version = Version(extradeep.__version__)
            file_version = Version(file_version)
            if prog_version < file_version:
                if prog_version.major != file_version.major or prog_version.minor != file_version.minor:
                    warnings.warn(
                        f"The loaded experiment was created with a newer version ({file_version}) of extradeep. "
                        f"This extradeep version ({prog_version}) might not work correctly with this experiment.")
                else:
                    logging.info(
                        f"The loaded experiment was created with a newer version ({file_version}) of extradeep. ")
        if 'progress_bar' in self.context:
            pbar = self.context['progress_bar']
            models = 0
            ms = data.get('measurements')
            if ms:
                for cp in ms.values():
                    for m in cp.values():
                        models += 1
                        pbar.total += len(m)
            pbar.total += models
            ms = data.get('modelers')
            if ms:
                pbar.total += len(ms)
                pbar.total += len(ms) * models
            pbar.update(0)
        return data

    def create_object(self):
        return Experiment()

    def postprocess_object(self, obj: Experiment):
        if 'progress_bar' in self.context:
            pbar = self.context['progress_bar']
        else:
            pbar = DUMMY_PROGRESS

        #for (callpath, metric, analysistype), measurement in obj.measurements.items():
        #    obj.add_callpath(callpath)
        #    obj.add_metric(metric)
        #    obj.add_analysistype(analysistype)
        #    for m in measurement:
        #        obj.add_coordinate(m.coordinate)
        #        m.callpath = callpath
        #        m.metric = metric
        #        m.analysistype = analysistype
        #    pbar.update()

        #obj.call_tree = io_helper.create_call_tree(obj.callpaths)

        temp_call_trees = {}
        analysistypes = obj.analysistypes
        for analysistype in analysistypes:
            callpaths = obj.callpaths[analysistype]
            call_tree = io_helper.create_call_tree(callpaths)
            temp_call_trees[analysistype] = call_tree

        obj.call_trees = temp_call_trees

        for modeler in obj.modelers:
            modeler.experiment = obj
            for key, model in modeler.models.items():
                model.measurements = obj.measurements[key]
            pbar.update()

        return obj
