# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extradeep.modelers.loader import load_modelers
from extradeep.modelers.single_parameter.basic import SingleParameterModeler as Default
#from extradeep.modelers.single_parameter.refining import RefiningModeler as Default

all_modelers = load_modelers(__path__, __name__)
all_modelers['Default'] = Default
