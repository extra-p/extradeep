# This file is part of the Extra-Deep software (https://github.com/extra-p/extrapdeep)
#
# Copyright (c) 2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extradeep.util.exceptions import CancelProcessError
from extradeep.util.progress_bar import ProgressBar
from extradeep.fileio.nsight_systems_file_reader import read_nsight_systems_files

from functools import partial
from threading import Event

from PySide2.QtCore import *  # @UnusedWildImport
from PySide2.QtWidgets import *  # @UnusedWildImport

class ParameterWidget(QWidget):

    def __init__(self, parent):
        super(ParameterWidget, self).__init__(parent)
        self.name = "Parameter"
        self.values = "1"

    def init_UI(self):
        layout = QFormLayout(self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setText(self.name)
        layout.addRow("Parameter name:", self.name_edit)

        self.values_edit = QLineEdit(self)
        self.values_edit.setText(self.values)
        layout.addRow("Values:", self.values_edit)

        self.setLayout(layout)

    def onNewValues(self):
        self.name_edit.setText(self.name)
        self.values_edit.setText(self.values)

class NsightSystemsReader(QDialog):

    def __init__(self, parent, dirName):
        super(NsightSystemsReader, self).__init__(parent)

        self.valid = False
        self.dir_name = dirName
        self.num_params = 1
        self.max_params = 3
        self.prefix = ""
        self.postfix = ""
        self.filename = "profile.cubex"
        self.repetitions = 1
        self.parameters = list()

        self._cancel_event = Event()

        self.init_UI()

    def init_UI(self):
        self.setWindowTitle("Import Settings")
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        main_layout = QFormLayout(self)
        layout = QFormLayout()
        self.controls_layout = layout

        self.scaling_choice = QComboBox(self)
        self.scaling_choice.addItem("weak")
        self.scaling_choice.addItem("strong")

        layout.addRow("Scaling type:", self.scaling_choice)

        self.progress_indicator = QProgressBar(self)
        self.progress_indicator.hide()
        layout.addRow(self.progress_indicator)

        main_layout.addRow(layout)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        main_layout.addRow(self.buttonBox)

        self.setLayout(main_layout)

    @Slot()
    def reject(self):
        self._cancel_event.set()
        super().reject()

    @Slot()
    def accept(self):

        self.scaling_type = self.scaling_choice.currentText()

        with ProgressBar(total=0, gui=True) as pbar:
            self._show_progressbar()
            pbar.display = partial(self._display_progress, pbar)
            pbar.sp = None

            # read the nsight systems .sqlite files
            try:
                self.experiment = read_nsight_systems_files(self.dir_name, self.scaling_type, pbar)
            except Exception as err:
                self.close()
                raise err

            if not self.experiment:
                QMessageBox.critical(self,
                                     "Error",
                                     "Could not read Nsight System .sqlite Files, may be corrupt!",
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                self.close()
                return
            self.valid = True

            super().accept()

    def _show_progressbar(self):
        self.controls_layout.setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.progress_indicator.show()

    def _display_progress(self, pbar: ProgressBar, msg=None, pos=None):
        if self._cancel_event.is_set():
            raise CancelProcessError()
        self.progress_indicator.setMaximum(pbar.total)
        self.progress_indicator.setValue(pbar.n)
        QApplication.processEvents()
        