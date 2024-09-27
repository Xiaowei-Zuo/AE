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
import gui_support_ver11
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


class LIVE_PLOT_APP(QtWidgets.QMainWindow):
    # Define all initial values/states
    def __init__(self):
        # self.threadpool = QtCore.QThreadPool()
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('myui_ver4.ui', self)  # Load the user interface file
        self.resize(888, 600)

        self.get_all_audio_devices()  # Define self.audio_devices_list
        self.get_all_NI_channels()  # Define self.NI_channels_list

        self.define_plot_widgets()  # Define all the canvases: grouped together for easier adaption to ui
        self.define_all_guiRelated()  # Define all the guiRelated slots (buttons and lineEdits) for each sensor

        # Restrain slots (except input selections and checkSetUp) from activating before proper setup
        self.restrain_buttons_forSetUp()

        # Input selection for each sensor
        self.comboBox_HY1.addItems(self.audio_prompt)
        # self.comboBox_HY2.addItems(self.audio_prompt)
        # self.comboBox_VB1.addItems(self.NI_prompt)

        # Check if all sensor inputs are selected
        self.pushButton_checkSetUp.clicked.connect(lambda: self.checkSetUp())

    def define_HY1(self, name):  # Hydrophone #1
        self.HY1 = gui_support_ver11.audio(name="HY1", chunk=4096, displayL=40960, device=self.audio_devices_list.index(name))
        self.HY1_plot = gui_support_ver11.plotting(self.HY1, self.HY1_canvas, self.HY1_canvasf, 17)
    #
    # def define_HY2(self, name):  # Hydrophone #1
    #     self.HY2 = gui_support_ver11.audio(name="HY2", chunk=4096, displayL=40960, device=self.audio_devices_list.index(name))
    #     self.HY2_plot = gui_support_ver11.plotting(self.HY2, self.HY2_canvas, self.HY2_canvasf, 19)
    #
    # def define_VB1(self, channel):  # Vibration Sensor #1
    #     self.VB1 = gui_support_ver11.NI(name="VB1", device='myDAQ1', fs=40000, chunk=4000, displayL=40000, channel=channel)
    #     self.VB1_plot = gui_support_ver11.plotting(self.VB1, self.VB1_canvas, self.VB1_canvasf, 10)

    def checkSetUp(self):  # If input selections are complete, proceed to defining all the sensors
        if (
                self.comboBox_HY1.currentText() in self.audio_devices_list
                # and
                # self.comboBox_HY2.currentText() in self.audio_devices_list
                # and
                # self.comboBox_VB1.currentText() in self.NI_channels_list
                and
                True
        ):

            self.pushButton_checkSetUp.setEnabled(False)
            self.comboBox_HY1.setEnabled(False)
            # self.comboBox_HY2.setEnabled(False)
            # self.comboBox_VB1.setEnabled(False)

            self.define_HY1(self.comboBox_HY1.currentText())
            # self.define_HY2(self.comboBox_HY2.currentText())
            # self.define_VB1(self.comboBox_VB1.currentText())

            self.connectToGUI(self.HY1, self.HY1_plot, self.HY1_guiRelated)
            # self.connectToGUI(self.HY2, self.HY2_plot, self.HY2_guiRelated)
            # self.connectToGUI(self.VB1, self.VB1_plot, self.VB1_guiRelated)

            self.sensor_set = [
                [self.HY1_plot, self.HY1_guiRelated],
                # [self.HY2_plot, self.HY2_guiRelated],
                # [self.VB1_plot, self.VB1_guiRelated],
            ]

            self.pushButton_ALL_startPlot.clicked.connect(lambda: self.startPlot_ALL())
            self.pushButton_ALL_endPlot.clicked.connect(lambda: self.endPlot_ALL())
            self.pushButton_ALL_startRec.clicked.connect(lambda: self.startRec_ALL())
            self.pushButton_ALL_endRec.clicked.connect(lambda: self.endRec_ALL())
            self.pushButton_ALL_save.clicked.connect(lambda: self.save_ALL())
            self.pushButton_ALL_startPlot.setEnabled(True)
            self.pushButton_ALL_endPlot.setEnabled(False)
            self.pushButton_ALL_startRec.setEnabled(True)
            self.pushButton_ALL_endRec.setEnabled(False)
            self.pushButton_ALL_save.setEnabled(False)
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Not ready!")
            msg.setIcon(QMessageBox.Critical)  # Critical:X, Warning:!, Information:i, Question:?
            msg.setInformativeText("Check if you have selected input for every sensor.")
            x = msg.exec_()

    def define_plot_widgets(self):
        # Defining all plot widgets
        self.HY1_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_2.addWidget(self.HY1_canvas, 1, 1, 1, 1)
        self.HY1_canvasf = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_2.addWidget(self.HY1_canvasf, 1, 2, 1, 1)
        #
        # self.HY2_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.ui.gridLayout_9.addWidget(self.HY2_canvas, 1, 1, 1, 1)
        # self.HY2_canvasf = MplCanvas(self, width=5, height=4, dpi=100)
        # self.ui.gridLayout_9.addWidget(self.HY2_canvasf, 1, 2, 1, 1)
        #
        # self.VB1_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.ui.gridLayout_15.addWidget(self.VB1_canvas, 1, 1, 1, 1)
        # self.VB1_canvasf = MplCanvas(self, width=5, height=4, dpi=100)
        # self.ui.gridLayout_15.addWidget(self.VB1_canvasf, 1, 2, 1, 1)

    def define_all_guiRelated(self):
        self.ALL_guiRelated = [
            self.pushButton_ALL_startPlot,
            self.pushButton_ALL_endPlot,
            self.pushButton_ALL_startRec,
            self.pushButton_ALL_endRec,
            self.pushButton_ALL_save,
        ]
        self.HY1_guiRelated = [
            self.pushButton_HY1_minmax,
            self.pushButton_HY1_startPlot,
            self.pushButton_HY1_endPlot,
            self.pushButton_HY1_startRec,
            self.pushButton_HY1_endRec,
            self.pushButton_HY1_save,
            self.lineEdit_HY1_ymin,
            self.lineEdit_HY1_ymax,
            self.lineEdit_HY1_tick,
        ]
        # self.HY2_guiRelated = [
        #     self.pushButton_HY2_minmax,
        #     self.pushButton_HY2_startPlot,
        #     self.pushButton_HY2_endPlot,
        #     self.pushButton_HY2_startRec,
        #     self.pushButton_HY2_endRec,
        #     self.pushButton_HY2_save,
        #     self.lineEdit_HY2_ymin,
        #     self.lineEdit_HY2_ymax,
        #     self.lineEdit_HY2_tick,
        # ]
        # self.VB1_guiRelated = [
        #     self.pushButton_VB1_minmax,
        #     self.pushButton_VB1_startPlot,
        #     self.pushButton_VB1_endPlot,
        #     self.pushButton_VB1_startRec,
        #     self.pushButton_VB1_endRec,
        #     self.pushButton_VB1_save,
        #     self.lineEdit_VB1_ymin,
        #     self.lineEdit_VB1_ymax,
        #     self.lineEdit_VB1_tick,
        # ]

    def restrain_buttons_forSetUp(self):
        # sensors_gui = [self.ALL_guiRelated, self.HY1_guiRelated, self.HY2_guiRelated, self.VB1_guiRelated]
        sensors_gui = [self.ALL_guiRelated, self.HY1_guiRelated]
        restrained_forSetUp = []
        for sensor in sensors_gui:
            restrained_forSetUp.extend(sensor)
        for item in restrained_forSetUp:
            item.setEnabled(False)

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

    def get_all_NI_channels(self):
        self.NI_channels_list = ['ai0']
        self.NI_prompt = ['Choose your input channel'] + self.NI_channels_list

    def connectToGUI(self, sensor, plot, guiRelated):
        plot.timer.timeout.connect(lambda: update_plot(sensor, plot))

        MM, SP, EP, SR, ER, SV, LE_ymin, LE_ymax, LE_tick = guiRelated

        MM.clicked.connect(lambda: self.minmax(
            plot,
            float(LE_ymin.text()),
            float(LE_ymax.text()),
            float(LE_tick.text()),
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

    def startPlot_ALL(self):
        self.pushButton_ALL_startPlot.setEnabled(False)
        self.pushButton_ALL_endPlot.setEnabled(True)
        self.pushButton_ALL_startRec.setEnabled(False)
        self.pushButton_ALL_endRec.setEnabled(False)
        self.pushButton_ALL_save.setEnabled(False)
        for item in self.sensor_set:
            plot, guiRelated = item
            self.startPlot(plot, guiRelated)

    def endPlot_ALL(self):
        self.pushButton_ALL_startPlot.setEnabled(True)
        self.pushButton_ALL_endPlot.setEnabled(False)
        self.pushButton_ALL_startRec.setEnabled(True)
        self.pushButton_ALL_endRec.setEnabled(False)
        self.pushButton_ALL_save.setEnabled(False)
        for item in self.sensor_set:
            plot, guiRelated = item
            self.endPlot(plot, guiRelated)

    def startRec_ALL(self):
        self.pushButton_ALL_startPlot.setEnabled(False)
        self.pushButton_ALL_endPlot.setEnabled(False)
        self.pushButton_ALL_startRec.setEnabled(False)
        self.pushButton_ALL_endRec.setEnabled(True)
        self.pushButton_ALL_save.setEnabled(False)
        for item in self.sensor_set:
            plot, guiRelated = item
            self.startRec(plot, guiRelated)

    def endRec_ALL(self):
        self.pushButton_ALL_startPlot.setEnabled(True)
        self.pushButton_ALL_endPlot.setEnabled(False)
        self.pushButton_ALL_startRec.setEnabled(True)
        self.pushButton_ALL_endRec.setEnabled(False)
        self.pushButton_ALL_save.setEnabled(True)
        for item in self.sensor_set:
            plot, guiRelated = item
            self.endRec(plot, guiRelated)

    def save_ALL(self):
        self.pushButton_ALL_startPlot.setEnabled(True)
        self.pushButton_ALL_endPlot.setEnabled(False)
        self.pushButton_ALL_startRec.setEnabled(True)
        self.pushButton_ALL_endRec.setEnabled(False)
        self.pushButton_ALL_save.setEnabled(False)
        for item in self.sensor_set:
            plot, _ = item
            self.save(plot)

    def minmax(self, plot, ymin, ymax, tick):
        plot.ymin = ymin
        plot.ymax = ymax
        plot.tick = tick

    def startPlot(self, plot, guiRelated):
        plot.recording = False

        plot.timer.start(plot.updateInterval)  # Update plot every num(interval) milliseconds

        MM, SP, EP, SR, ER, SV, LE_ymin, LE_ymax, LE_tick = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        LE_tick.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(True)
        SR.setEnabled(False)
        ER.setEnabled(False)
        SV.setEnabled(False)

    def endPlot(self, plot, guiRelated):
        plot.timer.stop()

        MM, SP, EP, SR, ER, SV, LE_ymin, LE_ymax, LE_tick = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        LE_tick.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(False)

    def startRec(self, plot, guiRelated):
        plot.recorded = []  # Reset recorded
        plot.recording = True

        plot.timer.start(plot.updateInterval)

        MM, SP, EP, SR, ER, SV, LE_ymin, LE_ymax, LE_tick = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        LE_tick.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(False)
        SR.setEnabled(False)
        ER.setEnabled(True)
        SV.setEnabled(False)

    def endRec(self, plot, guiRelated):
        plot.timer.stop()

        MM, SP, EP, SR, ER, SV, LE_ymin, LE_ymax, LE_tick = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        LE_tick.setEnabled(True)
        MM.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(True)

    def save(self, plot):
        name, recorded, fs = plot.sensor.name, plot.recorded, plot.sensor.fs

        print("Saving recorded data for:", name)
        Tk().withdraw()
        filenameNpath = filedialog.asksaveasfilename(
            title="Select a file for saving recording from "+name,
            filetypes=[("pickle files", "*.pickle")]
        )
        filenameNpath = filenameNpath + ".pickle"
        with open(filenameNpath, 'wb') as handle:
            pickle.dump(recorded, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Recording saved under:", filenameNpath)

        # with open(filenameNpath, 'rb') as handle:
        #     data = pickle.load(handle)
            # # Plot the fft
            # filename = Path(filenameNpath).stem
            # gui_support_ver11.quickfft(data, fs, filename)


app = QtWidgets.QApplication(sys.argv)
mainWindow = LIVE_PLOT_APP()
mainWindow.show()
sys.exit(app.exec_())
