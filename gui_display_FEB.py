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
import gui_sup_display_FEB
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
import time

# input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
# print(input_audio_deviceInfos)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


def get_data(q):
    return q.get(block=True)
#
# def update_data(QDG):
#     QDG.start_thread()

def update_plot(sensor, plot):
    start_time = time.time()

    sensor.start_thread()  # q.put()
    # plot.new_data = get_data(sensor.q) # q.get()
    plot.start_thread()
    sensor.end_thread()
    plot.end_thread()

    end_time = time.time()
    print("time elapsed:", end_time - start_time)
#
# def stop_update_plot(sensor, plot):
#     sensor.end_thread()
#     plot.end_thread()


def update_MLresults(ML):
    ML.start_thread()

def update_button(btn, bgcolor, text, color='black', bold=True, fontsize='15pt'):
    if bgcolor == 'red':
        bgcolor = '(255, 0, 0)'
    if bgcolor == 'orange':
        bgcolor = '(255, 137, 2)'
    if bgcolor == 'yellow':
        bgcolor = '(255, 255, 0)'
    if bgcolor == 'green':
        bgcolor = '(0, 170, 0);'
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

    # if fontsize = '15pt':
    fontsize = 'font: 63 '+ fontsize + '"Segoe UI Semibold";'

    # string = str_bgcolor + 'font: 63 15pt "Segoe UI Semibold";' + str_color + 'font - weight: bold;'
    string = str_bgcolor + fontsize + str_color + 'font - weight: bold;'

    # if bold:
    #     string += 'font - weight: bold;'

    btn.setStyleSheet(string)
    btn.setText(text)


class LIVE_PLOT_APP(QtWidgets.QMainWindow):
    # Define all initial values/states
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('myui_display_FEB.ui', self)  # Load the user interface file
        self.resize(888, 600)

        # self.p_DAQ_VB.start()

        self.HY = True
        self.VB = True
        self.AE = False
        self.setUp = True

        if self.HY:
            self.canvas_HY = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_10.addWidget(self.canvas_HY, 1, 1, 1, 2)

            self.HY = gui_sup_display_FEB.NI(
                name="HY", device='myDAQ1', channel="audioInputLeft",
                fs=100*1000, chunk=2048, displayL=65536,
                input_min=-2.0, input_max=2.0)
            self.plot_HY = gui_sup_display_FEB.plotting(
                self.HY, self.canvas_HY, 17,
                linewidth=.9, color='yellow',
                ymin=-2, ymax=2, tick=.2,)
            self.guiRelated_HY = [
                self.lineEdit_ymin_HY_1,
                self.lineEdit_ymax_HY_1,
                self.pushButton_minmax_HY_1,
                # self.pushButton_auto_HY_1,
                self.pushButton_startPlot_HY_1,
                self.pushButton_endPlot_HY_1,
                self.pushButton_startRec_HY_1,
                self.pushButton_endRec_HY_1,
                self.pushButton_save_HY_1,]
            self.connectToGUI(self.HY, self.plot_HY, self.guiRelated_HY)

            self.plot_HY.timer.timeout.connect(lambda: update_plot(self.HY, self.plot_HY))

            self.QDG_HY = gui_sup_display_FEB.QDG(self.HY, updateInterval=17)
            # self.QDG_HY.timer.timeout.connect(lambda: update_data(self.QDG_HY))

        if self.VB:
            self.canvas_VB = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_14.addWidget(self.canvas_VB, 1, 1, 1, 2)

            self.VB = gui_sup_display_FEB.NI(
                name="VB", device='Dev2', channel="ai0",
                fs=24*1000, chunk=2048, displayL=15728,)
            # self.getData_VB = gui_sup_display_FEB.getData(self.VB.q, 10)
            # self.p_DAQ_VB = multiprocessing.Process(target=gui_sup_display_FEB.myDAQ, args=(self.VB,))
            self.plot_VB = gui_sup_display_FEB.plotting(
                self.VB, self.canvas_VB,10,
                linewidth=.7,
                ymin=-5, ymax=5, tick=.5,)

            self.guiRelated_VB = [
                self.lineEdit_ymin_VB_1,
                self.lineEdit_ymax_VB_1,
                self.pushButton_minmax_VB_1,
                # self.pushButton_auto_VB_1,
                self.pushButton_startPlot_VB_1,
                self.pushButton_endPlot_VB_1,
                self.pushButton_startRec_VB_1,
                self.pushButton_endRec_VB_1,
                self.pushButton_save_VB_1,]
            self.connectToGUI(self.VB, self.plot_VB, self.guiRelated_VB)

            self.plot_VB.timer.timeout.connect(lambda: update_plot(self.VB, self.plot_VB))

        if self.AE:
            self.canvas_AE = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.gridLayout_18.addWidget(self.canvas_AE, 1, 1, 1, 2)

            self.AE = gui_sup_display_FEB.NI(
                name="AE", device='myDAQ1', channel="ai0",
                fs=40000, chunk=256, displayL=1024,)
            self.plot_AE = gui_sup_display_FEB.plotting(
                self.AE, self.canvas_AE, self.canvas_AE_2, 7,
                self.btn_displayDetSta, self.btn_displayDetLoc, self.btn_displayDetTip,
                self.btn_displayPrdSta, self.btn_displayPrdLoc, self.btn_displayPrdTip,
                linewidth=.5, color='light green',
                ymin=-5, ymax=10, tick=1,)

            self.guiRelated_AE = [
                self.lineEdit_ymin_AE_1,
                self.lineEdit_ymax_AE_1,
                self.pushButton_minmax_AE_1,
                # self.pushButton_auto_AE_1,
                self.pushButton_startPlot_AE_1,
                self.pushButton_endPlot_AE_1,
                self.pushButton_startRec_AE_1,
                self.pushButton_endRec_AE_1,
                self.pushButton_save_AE_1,]
            self.connectToGUI(self.AE, self.plot_AE, self.guiRelated_AE)

        if self.setUp:
            self.ML = gui_sup_display_FEB.ML(self.btn_displayDetSta, plot_HY=self.plot_HY)

            self.btn_sDet.clicked.connect(lambda: self.startDet(self.QDG_HY, self.ML))
            self.btn_eDet.clicked.connect(lambda: self.endDet(self.QDG_HY, self.ML))

            self.btn_sDet.setEnabled(True)
            self.btn_eDet.setEnabled(False)

            print("finished setup")

            self.ML.timer.timeout.connect(lambda: update_MLresults(self.ML))
            #
            # self.plot_HY.timer.start(self.plot_HY.updateInterval)
            # self.plot_VB.timer.start(self.plot_VB.updateInterval)



    def startDet(self, QDG_HY, ML):
        self.btn_sDet.setEnabled(False)
        self.btn_eDet.setEnabled(True)

        update_button(self.btn_displayDetSta, 'green', '정상', 'white', fontsize='20pt')
        update_button(self.btn_displayDetLoc, 'green', '정상', 'white')

        # QDG_HY.timer.start(QDG_HY.updateInterval)
        ML.timer.start(ML.updateInterval)


        print("started detection")

    def endDet(self, QDG_HY, ML):
        self.btn_sDet.setEnabled(True)
        self.btn_eDet.setEnabled(False)

        ML.timer.stop()
        QDG_HY.timer.stop()

        update_button(self.btn_displayDetSta, 'grey', 'N/A')
        update_button(self.btn_displayDetLoc, 'grey', 'N/A')

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

        # LE_ymin, LE_ymax, MM, AUTO, SP, EP, SR, ER, SV, LE_ymin_2, LE_ymax_2, MM_2, AUTO_2, SP_2, EP_2, SR_2, ER_2, SV_2 = guiRelated
        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, = guiRelated

        MM.clicked.connect(lambda: self.minmax(
            plot,
            float(LE_ymin.text()),
            float(LE_ymax.text()),
        ))
        # AUTO.clicked.connect(lambda: self.autoscale(plot))
        SP.clicked.connect(lambda: self.startPlot(plot, guiRelated))
        EP.clicked.connect(lambda: self.endPlot(plot, guiRelated))
        # SR.clicked.connect(lambda: self.startRec(plot, guiRelated))
        # ER.clicked.connect(lambda: self.endRec(plot, guiRelated))
        SV.clicked.connect(lambda: self.save(plot))

        MM.setEnabled(False)
        # AUTO.setEnabled(False)
        SP.setEnabled(True)
        EP.setEnabled(False)
        # SR.setEnabled(True)
        # ER.setEnabled(False)
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

        # plot.timer.start(plot.updateInterval)  # Update plot every num(interval) milliseconds

        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        # AUTO.setEnabled(True)
        SP.setEnabled(False)
        EP.setEnabled(True)
        SR.setEnabled(False)
        ER.setEnabled(False)
        SV.setEnabled(False)

    def endPlot(self, plot, guiRelated):
        plot.timer.stop()

        LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV, = guiRelated
        LE_ymin.setEnabled(True)
        LE_ymax.setEnabled(True)
        MM.setEnabled(True)
        # AUTO.setEnabled(True)
        SP.setEnabled(True)
        EP.setEnabled(False)
        SR.setEnabled(True)
        ER.setEnabled(False)
        SV.setEnabled(False)
    #
    # def startRec(self, QDG, plot, guiRelated):
    #     plot.recorded = []  # Reset recorded
    #     plot.recording = True
    #
    #     plot.timer.start(plot.updateInterval)
    #
    #     LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV = guiRelated
    #     LE_ymin.setEnabled(True)
    #     LE_ymax.setEnabled(True)
    #     MM.setEnabled(True)
    #     # AUTO.setEnabled(True)
    #     SP.setEnabled(False)
    #     EP.setEnabled(False)
    #     SR.setEnabled(False)
    #     ER.setEnabled(True)
    #     SV.setEnabled(False)
    #
    # def endRec(self, QDG, plot, guiRelated):
    #     plot.timer.stop()
    #
    #     LE_ymin, LE_ymax, MM, SP, EP, SR, ER, SV = guiRelated
    #     LE_ymin.setEnabled(True)
    #     LE_ymax.setEnabled(True)
    #     MM.setEnabled(True)
    #     # AUTO.setEnabled(True)
    #     SP.setEnabled(True)
    #     EP.setEnabled(False)
    #     SR.setEnabled(True)
    #     ER.setEnabled(False)
    #     SV.setEnabled(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = LIVE_PLOT_APP()
    mainWindow.show()
    sys.exit(app.exec_())
