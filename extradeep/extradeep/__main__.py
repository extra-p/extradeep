#from extradeep.extradeep import extradeep_modeler
#from extradeep.extradeep import extradeep_evaluator

#extradeep_modeler.main(prog='extradeep_modeler')
#extradeep_evaluator.main(prog='extradeep_evaluator')

from extradeep.extradeep import extradeep
from extradeep.extradeep import extradeep_gui
from extradeep.extradeep import extradeep_instrumenter
from extradeep.extradeep import extradeep_plotter

extradeep.main(prog="extradeep")
extradeep_gui.main(prog="extradeep_gui")
extradeep_instrumenter.main(prog="extradeep_instrumenter")
extradeep_plotter.main(prog="extradeep_plotter")
