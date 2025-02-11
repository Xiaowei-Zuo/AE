# import serial
import time
# import cv2, os, queue
import os
import numpy as np
from threading import Thread
import multiprocessing
from datetime import datetime, timedelta
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants
# import pypylon.pylon as py
# from harvesters.core import Harvester
from pyqtgraph import mkPen

import sys
import matplotlib
import pandas as pd
# from tensorflow_estimator.python.estimator.canned.timeseries.saved_model_utils import predict_continuation

matplotlib.use('Qt5Agg')
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

from itertools import chain
import csv

import pyaudio
import wave
import gui_display_FEB

import mytime

from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import Tk, filedialog

# import pyfftw


def myDAQ(device):
    device.DAQ()


class NI():
    def __init__(
            self, name, device, channel,
            fs, chunk, displayL,
            input_min=-10.0, input_max=10.0):
        self.name = name  # "VB"
        self.device = device
        self.channel = channel
        self.fs = fs
        self.chunk = chunk
        self.history = np.zeros((displayL,))
        self.input_min = input_min
        self.input_max = input_max

        self.q = multiprocessing.Queue()

    def start_thread(self):
        self.thread = Thread(target=self.DAQ)
        self.thread.start()
        self.thread.join()

    # def stop_thread(self):
    #     self.thread.join()

    def DAQ(self):
        # while True:
        with nidaqmx.Task() as task:


            channel=task.ai_channels.add_ai_voltage_chan(
                f"{self.device}/{self.channel}",
                min_val=self.input_min, max_val=self.input_max)
            # if self.name=="HY":
            #     channel.ai_min = -2.0
            #     channel.ai_max = 2.0
            task.timing.cfg_samp_clk_timing(
                rate=self.fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.chunk)
            data = task.read(number_of_samples_per_channel=self.chunk)  # list
            data = np.array(data) #*10
            self.q.put(data)
            # print("NI DAQ qsize:", self.q.qsize())
                # print(self.q.get())
                # print(self.q.get())


class audio():
    def __init__(
            self, name, device,
            chunk, displayL):
        self.name = name  # "HY"
        self.fs = 44100  # Default for audio
        self.chunk = chunk
        self.history = np.zeros((displayL,))

        self.q = multiprocessing.Queue()

        self.p = pyaudio.PyAudio()
        # print("audio device:", device)
        self.inStream = self.p.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.fs,
                                    input=True,
                                    frames_per_buffer=self.chunk,
                                    input_device_index=device)

    def start_thread(self):
        self.thread = Thread(target=self.DAQ)
        self.thread.start()
        self.thread.join()

    # def stop_thread(self):
    #     self.p.close(self.inStream)
    #     self.thread.join()

    def start_DAQ(self):
        p = multiprocessing.Process(target=self.DAQ)

    def DAQ(self):
        # while True:
        audioString = self.inStream.read(self.chunk)
        data = np.fromstring(audioString, dtype=np.int16)/1000
        data = data.tolist()
        self.q.put(data)
        # print("audio DAQ qsize:", self.q.qsize())


class getData():
    def __init__(self, q, updateInterval):
        self.new_data = None
        self.q = q

        self.timer = QtCore.QTimer()
        self.updateInterval = updateInterval

        # new_data = self.q.get(block=True)
        # self.get_all_queue_result()

    def get_all_queue_result(self):  # Empty q
        self.new_data = self.q.get()
        while not self.q.empty():
            self.new_data += self.q.get()


class plotting():
    def __init__(
            self, sensor, canvas, updateInterval,
            linewidth=.2, color='white',
            ymin=None, ymax=None, tick=.5,
            plot_excluded = None,
            canvas_2=None,
    ):
        self.sensor = sensor
        self.q = sensor.q
        # self.new_data = None

        self.plot_excluded = plot_excluded

        self.detect = False
        self.detected = False
        self.startTime_det = float('inf')

        # self.loc = round(np.random.rand() * 10, 1)
        self.detLoc = 1.2
        self.prdLoc = 3.7

        self.history = sensor.history

        self.fs = sensor.fs

        self.canvas = canvas

        self.linewidth = linewidth

        self.color=color
        if self.color=='white':
            self.color=(1,1,1)
        if self.color=='yellow':
            self.color=(1, 0.984, 0)
        if self.color=='light green':
            self.color=(0, 1, 0.29)

        self.ref_plot = None

        self.timer = QtCore.QTimer()
        self.updateInterval = updateInterval

        self.recording = False
        self.recorded = []

        self.auto = False
        self.ymin = ymin
        self.ymax = ymax
        # self.tick = (ymax-ymin)/10
        self.tick = tick

        self.second_plot = False

    def start_thread(self):
        self.thread = Thread(target=self.start)
        self.thread.start()
        self.thread.join()

    def start(self):
        block = True  # Works for vibra if True
        # try:
        self.new_data = self.q.get(block=block)
        # print(len(new_data))
        print("shape:", np.array(self.new_data).shape)


            ###########################
        shift = len(self.new_data)
        self.history = np.roll(self.history, -shift, axis=0)
        self.history[-shift:, ] = self.new_data
        self.y = self.history[:]
        self.canvas.axes.set_facecolor((0, 0, 0))

        if self.recording:
            self.recorded.extend(self.new_data)

        if self.ref_plot is None:
            plot_refs = self.canvas.axes.plot(self.y, color=self.color, linewidth=self.linewidth)
            self.ref_plot = plot_refs[0]
        else:
            self.ref_plot.set_ydata(self.y)

            ###########################################
        # except queue.Empty:
        #     print("empty")

        # Time domain
        self.canvas.axes.yaxis.grid(True, linestyle='--')
        start, end = self.canvas.axes.get_ylim()
        self.canvas.axes.yaxis.set_ticks(np.arange(start, end, self.tick))
        self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        # if self.ymin and self.ymax:
        self.canvas.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)
        # else:
        #     self.canvas.axes.relim()
        #     self.canvas.axes.autoscale_view()
        self.canvas.draw()


class ML():
    def __init__(
            self,
            btn_displayDetSta,
            plot_HY=None, plot_VB=None, plot_AE=None,
            updateInterval=19,
    ):
        self.btn_displayDetSta = btn_displayDetSta
        self.plot_HY = plot_HY
        self.plot_VB = plot_VB
        self.plot_AE = plot_AE

        self.timer = QtCore.QTimer()
        self.updateInterval = updateInterval

    def start_thread(self):
        self.thread = Thread(target=self.start)
        self.thread.start()
        self.thread.join()

    def start(self):
        print("in ML function")
        print(np.mean(self.plot_VB.new_data))
        if np.mean(self.plot_VB.new_data) > -0.3:
            gui_display_FEB.update_button(self.btn_displayDetSta, 'red', '정상', 'white')
        else:
            gui_display_FEB.update_button(self.btn_displayDetSta, 'green', '정상', 'white')