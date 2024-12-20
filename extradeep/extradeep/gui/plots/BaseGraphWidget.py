# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from PySide2.QtWidgets import QSizePolicy
from matplotlib import patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from extradeep.gui.main_widget import MainWidget


class GraphDisplayWindow(FigureCanvas):
    def __init__(self, graphWidget, main_widget: MainWidget, width=5, height=4, dpi=100):
        self.graphWidget = graphWidget
        self.main_widget = main_widget
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        super().setSizePolicy(QSizePolicy.Expanding,
                              QSizePolicy.Expanding)
        super().updateGeometry()
        self.draw_figure()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fig.tight_layout()

    def redraw(self):
        rotation = self._save_rotation()
        self.fig.clear()
        self.draw_figure()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fig.tight_layout()
        self._restore_rotation(rotation)
        self.fig.canvas.draw_idle()

    def _save_rotation(self):
        return [(ax.elev, ax.azim) if isinstance(ax, Axes3D) else (None, None) for ax in self.fig.axes]

    def _restore_rotation(self, rotations):
        for ax, (elev, azim) in zip(self.fig.axes, rotations):
            if isinstance(ax, Axes3D):
                ax.view_init(elev, azim)

    @abstractmethod
    def draw_figure(self):
        ...

    # noinspection DuplicatedCode
    def _calculate_grid_parameters(self, maxX, maxY):
        number_of_pixels_x = 50
        number_of_pixels_y = 50

        pixel_gap_x = self.getPixelGap(0, maxX, number_of_pixels_x)
        pixel_gap_y = self.getPixelGap(0, maxY, number_of_pixels_y)
        return pixel_gap_x, pixel_gap_y

    @staticmethod
    def getPixelGap(lowerlimit, upperlimit, numberOfPixels):
        """
           This function calculate the gap in pixels based on number of pixels and max value
        """
        pixelGap = (upperlimit - lowerlimit) / numberOfPixels
        return pixelGap

    def calculate_z_optimized(self, X, Y, function):
        """
           This function evaluates the function passed to it.
        """
        xs, ys = X.reshape(-1), Y.reshape(-1)
        points = np.ndarray((len(self.main_widget.data_display.parameters), len(xs)))

        parameter_value_list = self.main_widget.data_display.getValues()
        for p, v in parameter_value_list.items():
            points[p] = v
        param1 = self.main_widget.data_display.getAxisParameter(0).id
        param2 = self.main_widget.data_display.getAxisParameter(1).id
        points[param1] = xs
        points[param2] = ys

        z_value = function.evaluate(points)
        return z_value

    def calculate_z_models(self, maxX, maxY, model_list, max_z=0):
        # define grid parameters based on max x and max y value
        pixelGap_x, pixelGap_y = self._calculate_grid_parameters(maxX, maxY)
        # Get the grid of the x and y values
        x = np.arange(1.0, maxX, pixelGap_x)
        y = np.arange(1.0, maxY, pixelGap_y)
        X, Y = np.meshgrid(x, y)
        # Get the z value for the x and y value
        z_List = list()
        Z_List = list()
        previous = np.seterr(invalid='ignore', divide='ignore')
        for model in model_list:
            function = model.hypothesis.function
            zs = self.calculate_z_optimized(X, Y, function)
            Z = zs.reshape(X.shape)
            z_List.append(zs)
            Z_List.append(Z)
            max_z = max(max_z, np.max(zs[np.logical_not(np.isinf(zs))]))
        np.seterr(**previous)
        for z, Z in zip(z_List, Z_List):
            z[np.isinf(z)] = max_z
            Z[np.isinf(Z)] = max_z
        return X, Y, Z_List, z_List

    def get_selected_models(self):
        selected_analysis_type = self.main_widget.getSelectedAnalysisType()
        selected_metric = self.main_widget.getSelectedMetric()
        selected_callpaths = self.main_widget.getSelectedCallpath()
        model_set = self.main_widget.getCurrentModel()
        if not selected_callpaths or model_set is None:
            return None, None
        model_set_models = model_set.models
        if not model_set_models:
            return None, None
        model_list = list()
        for selected_callpath in selected_callpaths:
            key = (selected_callpath.path, selected_metric, selected_analysis_type)
            if key in model_set_models:
                model = model_set_models[key]
                model_list.append(model)
        return model_list, selected_callpaths

    def draw_legend(self, ax_all, dict_callpath_color):
        fontSize = self.graphWidget.getFontSize()
        # draw legend
        patches = list()
        for key, value in dict_callpath_color.items():
            labelName = str(key.name)
            if labelName.startswith("_"):
                labelName = labelName[1:]
            patch = mpatches.Patch(color=value, label=labelName)
            patches.append(patch)
        leg = ax_all.legend(handles=patches, fontsize=fontSize,
                            loc="upper right", bbox_to_anchor=(1, 1))
        if leg:
            leg.set_draggable(True)

    def get_max(self, lower_max=2.0):
        # since we are drawing the plots with minimum axis value of 1 to avoid nan values,
        # so the first max-value of parameter could be 2 to calculate number of subdivisions
        maxX = self.graphWidget.getMaxX()
        maxY = self.graphWidget.getMaxY()
        # define min x and min y value
        if maxX < lower_max:
            maxX = lower_max
        if maxY < lower_max:
            maxY = lower_max
        return maxX, maxY


class BaseContourGraph(GraphDisplayWindow):
    @abstractmethod
    def draw_figure(self):
        ...

    def _calculate_grid_parameters(self, maxX, maxY):
        # define grid parameters based on max x and max y value
        numberOfPixels_x = 100
        numberOfPixels_y = 100
        pixelGap_x = self.getPixelGap(0, maxX, numberOfPixels_x)
        pixelGap_y = self.getPixelGap(0, maxY, numberOfPixels_y)
        return pixelGap_x, pixelGap_y