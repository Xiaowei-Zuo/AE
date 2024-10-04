import matplotlib
import pickle

matplotlib.use('Qt5Agg')  # To make it fast!
import sys
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import nidaqmx
from nidaqmx.constants import AcquisitionType
import pdb
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio, QCameraInfo
from PyQt5.QtWidgets import QMessageBox
from itertools import chain
import csv
import pyaudio
import wave
import mytime
import gui_support_display
from tkinter import Tk, filedialog
# import PIL.Image
from PIL import Image
from tkinter.filedialog import askopenfilename, askdirectory
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
print(input_audio_deviceInfos)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


def update_plot(sensor, plot):
    sensor.start_thread()  # q.put()
    plot.start_thread()  # q.get()

def update_button(btn, bgcolor, text):
    if bgcolor=='green':
        bgcolor='(0, 255, 0)'
    if bgcolor=='yellow':
        bgcolor = '(255, 255, 0)'
    if bgcolor=='red':
        bgcolor = '(255, 0, 0)'
    if bgcolor=='grey':
        bgcolor = '(156, 156, 156)'
    str_bgcolor = 'background-color: rgb'+bgcolor+';'

    string = str_bgcolor+'font: 63 15pt "Segoe UI Semibold";'

    btn.setStyleSheet(string)
    btn.setText(text)

class LIVE_PLOT_APP(QtWidgets.QMainWindow):
    # Define all initial values/states
    def __init__(self):
        # self.threadpool = QtCore.QThreadPool()
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('myui_display_ver2.ui', self)  # Load the user interface file
        self.resize(888, 600)

        self.define_plot_widgets()  # Define all the canvases: grouped together for easier adaption to ui
        self.define_all_guiRelated()  # Define all the guiRelated slots (buttons and lineEdits) for each sensor

        self.restrain_buttons_forSetUp()
        self.checkSetUp(self.guiRelated_ALL)

    def define_HY(self):  # Hydrophone #1
        self.HY = gui_support_display.audio(name="HY", chunk=4096, displayL=5120, device=1)
        self.plot_HY = gui_support_display.plotting(self.HY, self.canvas_HY, self.canvas_HY_2, 17, linewidth=.2, ymin=-2, ymax=2, tick=.5)
        pass

    def define_VB(self):  # Vibration Sensor
        self.VB = gui_support_display.NI(name="VB", device='Dev2', fs=40000, chunk=256, displayL=500, channel="ai0")
        self.plot_VB = gui_support_display.plotting(self.VB, self.canvas_VB, self.canvas_VB_2, 10, linewidth=1, ymin=-.8, ymax=0, tick=.1)
        # self.VB = gui_support_display.audio(name="VB", chunk=2048, displayL=20480, device=2)
        # self.plot_VB = gui_support_display.plotting(self.VB, self.canvas_VB, self.canvas_VB_2, 17)

    def define_AE(self):  # AE Sensor
        self.AE = gui_support_display.NI(name="AE", device='Dev2', fs=40000, chunk=256, displayL=500, channel="ai1")
        self.plot_AE = gui_support_display.plotting(self.AE, self.canvas_AE, self.canvas_AE_2, 7, linewidth=1, ymin=-.5, ymax=3, tick=.5, btn_status=self.pushButton_status, btn_loc=self.pushButton_location)

    def checkSetUp(self, guiRelated_ALL):  # If input selections are complete, proceed to defining all the sensors
        if True:
            self.define_HY()
            self.define_VB()
            self.define_AE()

            # self.connectToGUI(self.HY, self.plot_HY, self.guiRelated_HY)
            self.connectToGUI(self.VB, self.plot_VB, self.guiRelated_VB)
            self.connectToGUI(self.AE, self.plot_AE, self.guiRelated_AE)

            SM, EM, SPR, EPR, btn_status, btn_loc, label_status, label_loc = guiRelated_ALL

            plot_detect = self.plot_AE
            SM.clicked.connect(lambda: self.startMNT(self.guiRelated_ALL, plot_detect))
            EM.clicked.connect(lambda: self.endMNT(self.guiRelated_ALL, plot_detect))
            SPR.clicked.connect(lambda: self.startPRED(self.guiRelated_ALL, plot_detect))
            EPR.clicked.connect(lambda: self.endPRED(self.guiRelated_ALL, plot_detect))
            SM.setEnabled(True)
            SPR.setEnabled(True)

    def startMNT(self, guiRelated_ALL, plot_detect):
        SM, EM, SPR, EPR, btn_status, btn_loc, label_status, label_loc = guiRelated_ALL
        SM.setEnabled(False)
        EM.setEnabled(True)
        label_status.setEnabled(True)
        btn_status.setEnabled(True)
        update_button(btn_status, 'green', 'normal')
        plot_detect.detect=True

    def endMNT(self, guiRelated_ALL, plot_detect):
        SM, EM, SPR, EPR, btn_status, btn_loc, label_status, label_loc = guiRelated_ALL
        SM.setEnabled(True)
        EM.setEnabled(False)
        label_status.setEnabled(False)
        btn_status.setEnabled(False)
        update_button(btn_status, 'grey', 'N/A')
        plot_detect.detect = False

    def startPRED(self, guiRelated_ALL, plot_detect):
        SM, EM, SPR, EPR, btn_status, btn_loc, label_status, label_loc = guiRelated_ALL
        SPR.setEnabled(False)
        EPR.setEnabled(True)
        label_loc.setEnabled(True)
        btn_loc.setEnabled(True)
        update_button(btn_loc, 'green', 'normal')
        plot_detect.predict = True

    def endPRED(self, guiRelated_ALL, plot_detect):
        SM, EM, SPR, EPR, btn_status, btn_loc, label_status, label_loc = guiRelated_ALL
        SPR.setEnabled(True)
        EPR.setEnabled(False)
        label_loc.setEnabled(False)
        btn_loc.setEnabled(False)
        update_button(btn_loc, 'grey', 'N/A')
        plot_detect.predict = False

    def define_plot_widgets(self):
        # Defining all plot widgets
        self.canvas_HY = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_10.addWidget(self.canvas_HY, 1, 1, 1, 2)
        self.canvas_HY_2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_12.addWidget(self.canvas_HY_2, 1, 1, 1, 2)

        self.canvas_VB = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_14.addWidget(self.canvas_VB, 1, 1, 1, 2)
        self.canvas_VB_2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_16.addWidget(self.canvas_VB_2, 1, 1, 1, 2)

        self.canvas_AE = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_18.addWidget(self.canvas_AE, 1, 1, 1, 2)
        self.canvas_AE_2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_20.addWidget(self.canvas_AE_2, 1, 1, 1, 2)

    def define_all_guiRelated(self):
        self.guiRelated_ALL = [
            self.pushButton_startMonitoring,
            self.pushButton_endMonitoring,
            self.pushButton_startPrediction,
            self.pushButton_endPrediction,
            self.pushButton_status,
            self.pushButton_location,
            self.label_status,
            self.label_location,
        ]
        self.guiRelated_HY = [
            self.lineEdit_ymin_HY_1,
            self.lineEdit_ymax_HY_1,
            self.pushButton_minmax_HY_1,
            self.pushButton_startPlot_HY_1,
            self.pushButton_endPlot_HY_1,
            self.pushButton_startRec_HY_1,
            self.pushButton_endRec_HY_1,
            self.pushButton_save_HY_1,
            self.lineEdit_ymin_HY_2,
            self.lineEdit_ymax_HY_2,
            self.pushButton_minmax_HY_2,
            self.pushButton_startPlot_HY_2,
            self.pushButton_endPlot_HY_2,
            self.pushButton_startRec_HY_2,
            self.pushButton_endRec_HY_2,
            self.pushButton_save_HY_2,
        ]
        self.guiRelated_VB = [
            self.lineEdit_ymin_VB_1,
            self.lineEdit_ymax_VB_1,
            self.pushButton_minmax_VB_1,
            self.pushButton_startPlot_VB_1,
            self.pushButton_endPlot_VB_1,
            self.pushButton_startRec_VB_1,
            self.pushButton_endRec_VB_1,
            self.pushButton_save_VB_1,
            self.lineEdit_ymin_VB_2,
            self.lineEdit_ymax_VB_2,
            self.pushButton_minmax_VB_2,
            self.pushButton_startPlot_VB_2,
            self.pushButton_endPlot_VB_2,
            self.pushButton_startRec_VB_2,
            self.pushButton_endRec_VB_2,
            self.pushButton_save_VB_2,
        ]
        self.guiRelated_AE = [
            self.lineEdit_ymin_AE_1,
            self.lineEdit_ymax_AE_1,
            self.pushButton_minmax_AE_1,
            self.pushButton_startPlot_AE_1,
            self.pushButton_endPlot_AE_1,
            self.pushButton_startRec_AE_1,
            self.pushButton_endRec_AE_1,
            self.pushButton_save_AE_1,
            self.lineEdit_ymin_AE_2,
            self.lineEdit_ymax_AE_2,
            self.pushButton_minmax_AE_2,
            self.pushButton_startPlot_AE_2,
            self.pushButton_endPlot_AE_2,
            self.pushButton_startRec_AE_2,
            self.pushButton_endRec_AE_2,
            self.pushButton_save_AE_2,
        ]

    def restrain_buttons_forSetUp(self):
        sensors_gui = [self.guiRelated_ALL, self.guiRelated_HY, self.guiRelated_VB, self.guiRelated_AE]
        restrained_forSetUp = []
        for sensor in sensors_gui:
            restrained_forSetUp.extend(sensor)
        for item in restrained_forSetUp:
            item.setEnabled(False)

    ####################################General functions below#####################################
    def connectToGUI(self, sensor, plot, guiRelated):
        plot.timer.timeout.connect(lambda: update_plot(sensor, plot))

        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated

        MM.clicked.connect(lambda: self.minmax(
            plot,
            float(LE_ymin.text()),
            float(LE_ymax.text()),
        ))
        SP.clicked.connect(lambda: self.startPlot(plot, guiRelated))
        EP.clicked.connect(lambda: self.endPlot(plot, guiRelated))
        SR.clicked.connect(lambda: self.startRec(plot, guiRelated))
        ER.clicked.connect(lambda: self.endRec(plot, guiRelated))
        SV.clicked.connect(lambda: self.save(plot))

        MM.setEnabled(False)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(False)

    def minmax(self, plot, ymin, ymax):
        plot.ymin = ymin
        plot.ymax = ymax

    def startPlot(self, plot, guiRelated):
        plot.recording = False

        plot.timer.start(plot.updateInterval)  # Update plot every num(interval) milliseconds

        # self.pushButton_status.setStyleSheet(
        #     'background-color: rgb(156, 156, 156);'
        #     'color: rgb(255, 255, 255);font: 63 15pt "Segoe UI Semibold";background-color: rgb(255, 255, 0);')


        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(True)
        SR.setEnabled(False)
        ER.setEnabled(False)
        SV.setEnabled(False)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        SP_2.setEnabled(False)
        EP_2.setEnabled(True)
        SR_2.setEnabled(False)
        ER_2.setEnabled(False)
        SV_2.setEnabled(False)

    def endPlot(self, plot, guiRelated):
        plot.timer.stop()

        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(False)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        SP_2.setEnabled(True)
        EP_2.setEnabled(False)
        SR_2.setEnabled(True)
        ER_2.setEnabled(False)
        SV_2.setEnabled(False)

    def startRec(self, plot, guiRelated):
        plot.recorded = []  # Reset recorded
        plot.recording = True

        plot.timer.start(plot.updateInterval)

        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(False)
        SR.setEnabled(False)
        ER.setEnabled(True)
        SV.setEnabled(False)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        SP_2.setEnabled(False)
        EP_2.setEnabled(False)
        SR_2.setEnabled(False)
        ER_2.setEnabled(True)
        SV_2.setEnabled(False)

    def endRec(self, plot, guiRelated):
        plot.timer.stop()

        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(True)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        SP_2.setEnabled(True)
        EP_2.setEnabled(False)
        SR_2.setEnabled(True)
        ER_2.setEnabled(False)
        SV_2.setEnabled(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = LIVE_PLOT_APP()
    mainWindow.show()
    sys.exit(app.exec_())