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
#
#
# input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)
# print(input_audio_deviceInfos)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


# def update_plot(sensor, plot):
#     sensor.start_thread()  # q.put()
#     plot.start_thread()  # q.get()


class LIVE_PLOT_APP(QtWidgets.QMainWindow):
    # Define all initial values/states
    def __init__(self):
        # self.threadpool = QtCore.QThreadPool()
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('myui_display_ver2.ui', self)  # Load the user interface file
        self.resize(888, 600)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = LIVE_PLOT_APP()
    mainWindow.show()
    sys.exit(app.exec_())
