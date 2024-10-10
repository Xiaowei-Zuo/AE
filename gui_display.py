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
import multiprocessing

input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
# print(input_audio_deviceInfos)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


def get_data(q):
    return q.get(block=True)


def update_plot(sensor, plot):
    sensor.DAQ()  # q.put()
    # plot.new_data = get_data(sensor.q) # q.get()
    plot.start_thread()


def update_button(btn, bgcolor, text, color='black', bold=True):
    if bgcolor == 'red':
        bgcolor = '(255, 0, 0)'
    if bgcolor == 'orange':
        bgcolor = '(255, 137, 2)'
    if bgcolor == 'yellow':
        bgcolor = '(255, 255, 0)'
    if bgcolor == 'green':
        bgcolor = '(0, 255, 0)'
    if bgcolor == 'blue':
        bgcolor = '(33, 111, 255)'
    if bgcolor == 'grey':
        bgcolor = '(156, 156, 156)'
    if bgcolor == 'dark grey':
        bgcolor = '(72, 72, 72)'
    if bgcolor == 'black':
        bgcolor = '(0, 0, 0)'
    str_bgcolor = 'background-color: rgb' + bgcolor + ';'

    if color == 'black':
        color = '(0, 0, 0)'
    if color == 'white':
        color = '(255, 255, 255)'
    str_color = 'color: rgb' +color + ';'

    string = str_bgcolor + 'font: 63 15pt "Segoe UI Semibold";' + str_color

    if bold:
        string += 'font - weight: bold;'

    btn.setStyleSheet(string)
    btn.setText(text)


class LIVE_PLOT_APP(QtWidgets.QMainWindow):
    # Define all initial values/states
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('myui_display_ver2.ui', self)  # Load the user interface file
        self.resize(888, 600)

        self.get_all_audio_devices()

        # self.p_DAQ_VB.start()
        self.HY = True
        self.VB = True
        self.AE = True
        # self.HY1 = True
        # self.HY2 = False
        # self.VB1 = True
        # self.VB2 = False
        # self.AE1 = True
        # self.AE2 = False
        self.setUp = True

        if self.HY:
            self.canvas_HY = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_10.addWidget(self.canvas_HY, 1, 1, 1, 2)
            self.canvas_HY_2 = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_12.addWidget(self.canvas_HY_2, 1, 1, 1, 2)

            self.HY = gui_support_display.audio(
                name="HY", device=2,
                chunk=2048, displayL=5000, )
            # self.HY = gui_support_display.NI(
            #     name="HY", device='myDAQ2', channel="audioInputLeft",
            #     fs=40000, chunk=256, displayL=2048, )
            self.plot_HY = gui_support_display.plotting(
                self.HY, self.canvas_HY, self.canvas_HY_2, 17,
                self.btn_displayDetSta, self.btn_displayDetLoc, self.btn_displayDetTip,
                self.btn_displayPrdSta, self.btn_displayPrdLoc, self.btn_displayPrdTip,
                linewidth=.5, color='yellow',
                ymin=-5, ymax=5, tick=1,
            )
            self.guiRelated_HY = [
                self.lineEdit_ymin_HY_1,
                self.lineEdit_ymax_HY_1,
                self.pushButton_minmax_HY_1,
                self.pushButton_auto_HY_1,
                self.pushButton_startPlot_HY_1,
                self.pushButton_endPlot_HY_1,
                self.pushButton_startRec_HY_1,
                self.pushButton_endRec_HY_1,
                self.pushButton_save_HY_1,
                self.lineEdit_ymin_HY_2,
                self.lineEdit_ymax_HY_2,
                self.pushButton_minmax_HY_2,
                self.pushButton_auto_HY_2,
                self.pushButton_startPlot_HY_2,
                self.pushButton_endPlot_HY_2,
                self.pushButton_startRec_HY_2,
                self.pushButton_endRec_HY_2,
                self.pushButton_save_HY_2,
            ]
            self.connectToGUI(self.HY, self.plot_HY, self.guiRelated_HY)

        if self.VB:
            self.canvas_VB = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_14.addWidget(self.canvas_VB, 1, 1, 1, 2)
            self.canvas_VB_2 = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_16.addWidget(self.canvas_VB_2, 1, 1, 1, 2)

            self.VB = gui_support_display.NI(
                name="VB", device='Dev2', channel="ai0",
                fs=40000, chunk=128, displayL=512,)
            self.plot_VB = gui_support_display.plotting(
                self.VB, self.canvas_VB, self.canvas_VB_2, 10,
                self.btn_displayDetSta, self.btn_displayDetLoc, self.btn_displayDetTip,
                self.btn_displayPrdSta, self.btn_displayPrdLoc, self.btn_displayPrdTip,
                linewidth=.5,
                ymin=-5, ymax=-3, tick=.5,
            )

            self.guiRelated_VB = [
                self.lineEdit_ymin_VB_1,
                self.lineEdit_ymax_VB_1,
                self.pushButton_minmax_VB_1,
                self.pushButton_auto_VB_1,
                self.pushButton_startPlot_VB_1,
                self.pushButton_endPlot_VB_1,
                self.pushButton_startRec_VB_1,
                self.pushButton_endRec_VB_1,
                self.pushButton_save_VB_1,
                self.lineEdit_ymin_VB_2,
                self.lineEdit_ymax_VB_2,
                self.pushButton_minmax_VB_2,
                self.pushButton_auto_VB_2,
                self.pushButton_startPlot_VB_2,
                self.pushButton_endPlot_VB_2,
                self.pushButton_startRec_VB_2,
                self.pushButton_endRec_VB_2,
                self.pushButton_save_VB_2,
            ]
            self.connectToGUI(self.VB, self.plot_VB, self.guiRelated_VB)

        if self.AE:
            self.canvas_AE = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_18.addWidget(self.canvas_AE, 1, 1, 1, 2)
            self.canvas_AE_2 = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_20.addWidget(self.canvas_AE_2, 1, 1, 1, 2)

            self.AE = gui_support_display.NI(
                name="AE", device='myDAQ1', channel="ai0",
                fs=40000, chunk=128, displayL=512,)
            self.plot_AE = gui_support_display.plotting(
                self.AE, self.canvas_AE, self.canvas_AE_2, 7,
                self.btn_displayDetSta, self.btn_displayDetLoc, self.btn_displayDetTip,
                self.btn_displayPrdSta, self.btn_displayPrdLoc, self.btn_displayPrdTip,
                linewidth=.5, color='light green',
                ymin=-5, ymax=10, tick=1,
            )

            self.guiRelated_AE = [
                self.lineEdit_ymin_AE_1,
                self.lineEdit_ymax_AE_1,
                self.pushButton_minmax_AE_1,
                self.pushButton_auto_AE_1,
                self.pushButton_startPlot_AE_1,
                self.pushButton_endPlot_AE_1,
                self.pushButton_startRec_AE_1,
                self.pushButton_endRec_AE_1,
                self.pushButton_save_AE_1,
                self.lineEdit_ymin_AE_2,
                self.lineEdit_ymax_AE_2,
                self.pushButton_minmax_AE_2,
                self.pushButton_auto_AE_2,
                self.pushButton_startPlot_AE_2,
                self.pushButton_endPlot_AE_2,
                self.pushButton_startRec_AE_2,
                self.pushButton_endRec_AE_2,
                self.pushButton_save_AE_2,
            ]
            self.connectToGUI(self.AE, self.plot_AE, self.guiRelated_AE)

        if self.setUp:
            plot_det = self.plot_VB
            plot_det.plot_excluded = self.plot_AE

            plot_prd = self.plot_AE
            plot_prd.plot_excluded = self.plot_VB

            self.btn_sDet.clicked.connect(lambda: self.startDet(plot_det))
            self.btn_eDet.clicked.connect(lambda: self.endDet(plot_det))

            self.btn_sPrd.clicked.connect(lambda: self.startPred(plot_prd))
            self.btn_ePrd.clicked.connect(lambda: self.endPred(plot_prd))

            self.btn_sDet.setEnabled(True)
            self.btn_eDet.setEnabled(False)
            self.btn_sPrd.setEnabled(True)
            self.btn_ePrd.setEnabled(False)

    def startDet(self, plot):
        self.btn_sDet.setEnabled(False)
        self.btn_eDet.setEnabled(True)

        plot.btn_displayDetSta = self.btn_displayDetSta
        plot.btn_displayDetLoc = self.btn_displayDetLoc
        plot.btn_displayDetTip = self.btn_displayDetTip

        update_button(plot.btn_displayDetSta, 'dark grey', '정상', 'white')
        update_button(plot.btn_displayDetLoc, 'dark grey', '정상', 'white')
        update_button(plot.btn_displayDetTip, 'dark grey', '정상', 'white')

        plot.detect = True

    def endDet(self, plot):
        self.btn_sDet.setEnabled(True)
        self.btn_eDet.setEnabled(False)

        update_button(plot.btn_displayDetSta, 'grey', 'N/A')
        update_button(plot.btn_displayDetLoc, 'grey', 'N/A')
        update_button(plot.btn_displayDetTip, 'grey', 'N/A')

        plot.detect = False

    def startPred(self, plot):
        self.btn_sPrd.setEnabled(False)
        self.btn_ePrd.setEnabled(True)

        plot.btn_displayPrdSta = self.btn_displayPrdSta
        plot.btn_displayPrdLoc = self.btn_displayPrdLoc
        plot.btn_displayPrdTip = self.btn_displayPrdTip

        update_button(plot.btn_displayPrdSta, 'green', 'normal')
        update_button(plot.btn_displayPrdLoc, 'green', 'normal')
        update_button(plot.btn_displayPrdTip, 'green', 'normal')

        plot.predict = True

    def endPred(self, plot):
        self.btn_sPrd.setEnabled(True)
        self.btn_ePrd.setEnabled(False)

        update_button(plot.btn_displayPrdSta, 'grey', 'N/A')
        update_button(plot.btn_displayPrdLoc, 'grey', 'N/A')
        update_button(plot.btn_displayPrdTip, 'grey', 'N/A')

        plot.predict = False

    ####################################General functions below#####################################
    def get_all_audio_devices(self):
        p = pyaudio.PyAudio()
        audio_info = p.get_host_api_info_by_index(0)
        numdevices = audio_info.get('deviceCount')
        self.audio_devices_list = []  # ['Choose your audio input']
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = p.get_device_info_by_host_api_device_index(0, i).get('name')
                print("Input Device id ", i, " - ", name)
                self.audio_devices_list.append(name)
        self.audio_prompt = ['Choose your input audio'] + self.audio_devices_list


    def connectToGUI(self, sensor, plot, guiRelated):
        plot.timer.timeout.connect(lambda: update_plot(sensor, plot))

        LE_ymin, LE_ymax, MM, AUTO, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, AUTO_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated

        MM.clicked.connect(lambda: self.minmax(
            plot,
            float(LE_ymin.text()),
            float(LE_ymax.text()),
        ))
        AUTO.clicked.connect(lambda: self.autoscale(plot))
        SP.clicked.connect(lambda: self.startPlot(plot, guiRelated))
        EP.clicked.connect(lambda: self.endPlot(plot, guiRelated))
        SR.clicked.connect(lambda: self.startRec(plot, guiRelated))
        ER.clicked.connect(lambda: self.endRec(plot, guiRelated))
        SV.clicked.connect(lambda: self.save(plot))

        MM.setEnabled(False)
        AUTO.setEnabled(False)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(False)

    def minmax(self, plot, ymin, ymax):
        # plot.auto = False
        # plot.auto_2 = False
        plot.ymin = ymin
        plot.ymax = ymax

    def autoscale(self, plot):
        # plot.auto = True
        # plot.ymin = None
        # plot.ymax = None
        # plot.canvas.axes.relim()
        # plot.canvas.axes.autoscale_view()
        self.enableAutoRange(axis='y')
        self.setAutoVisible(y=True)

    def startPlot(self, plot, guiRelated):
        plot.recording = False

        plot.timer.start(plot.updateInterval)  # Update plot every num(interval) milliseconds

        LE_ymin, LE_ymax, MM, AUTO, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, AUTO_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        AUTO.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(True)
        SR.setEnabled(False)
        ER.setEnabled(False)
        SV.setEnabled(False)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        AUTO_2.setEnabled(True)
        SP_2.setEnabled(False)
        EP_2.setEnabled(True)
        SR_2.setEnabled(False)
        ER_2.setEnabled(False)
        SV_2.setEnabled(False)

    def endPlot(self, plot, guiRelated):
        plot.timer.stop()

        LE_ymin, LE_ymax, MM, AUTO, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, AUTO_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        AUTO.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(False)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        AUTO_2.setEnabled(True)
        SP_2.setEnabled(True)
        EP_2.setEnabled(False)
        SR_2.setEnabled(True)
        ER_2.setEnabled(False)
        SV_2.setEnabled(False)

    def startRec(self, plot, guiRelated):
        plot.recorded = []  # Reset recorded
        plot.recording = True

        plot.timer.start(plot.updateInterval)

        LE_ymin, LE_ymax, MM, AUTO, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, AUTO_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        AUTO.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(False)
        SR.setEnabled(False)
        ER.setEnabled(True)
        SV.setEnabled(False)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        AUTO_2.setEnabled(True)
        SP_2.setEnabled(False)
        EP_2.setEnabled(False)
        SR_2.setEnabled(False)
        ER_2.setEnabled(True)
        SV_2.setEnabled(False)

    def endRec(self, plot, guiRelated):
        plot.timer.stop()

        LE_ymin, LE_ymax, MM, AUTO, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, AUTO_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        AUTO.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(True)
        LE_ymin_2.setEnabled(True)
        LE_ymax_2.setEnabled(True)
        MM_2.setEnabled(True)
        AUTO_2.setEnabled(True)
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
