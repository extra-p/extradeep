# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import zipfile
from zipfile import ZipFile

from marshmallow import ValidationError

from extradeep.entities.experiment import ExperimentSchema
from extradeep.util.exceptions import FileFormatError, RecoverableError
from extradeep.util.progress_bar import DUMMY_PROGRESS

EXPERIMENT_DATA_FILE = 'experiment.json'


def read_experiment(path, progress_bar=DUMMY_PROGRESS):
    progress_bar.total += 3
    schema = ExperimentSchema()
    schema.set_progress_bar(progress_bar)
    try:
        with ZipFile(path, 'r', allowZip64=True) as file:
            progress_bar.update()
            data = file.read(EXPERIMENT_DATA_FILE).decode("utf-8")
            progress_bar.update()
            try:
                experiment = schema.loads(data)
                progress_bar.update()
                return True, experiment
            except ValidationError as v_err:
                raise FileFormatError(str(v_err)) from v_err
    except (IOError, zipfile.BadZipFile) as err:
        raise RecoverableError(str(err)) from err


def write_experiment(experiment, path, progress_bar=DUMMY_PROGRESS):
    progress_bar.total += 3
    schema = ExperimentSchema()
    try:
        with ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as file:
            progress_bar.update()
            try:
                data = schema.dumps(experiment)
                progress_bar.update()
                file.writestr(EXPERIMENT_DATA_FILE, data)
                progress_bar.update()
            except ValidationError as v_err:
                raise FileFormatError(str(v_err)) from v_err
    except (IOError, FileNotFoundError, zipfile.BadZipFile) as err:
        raise RecoverableError(str(err)) from err
