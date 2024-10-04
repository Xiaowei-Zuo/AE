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
import gui_display

import mytime

from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import Tk, filedialog

# import pyfftw


class NI():
    def __init__(self, name, device, channel, fs, chunk, displayL):
        self.name = name  # "VB"
        self.device = device
        self.channel = channel
        self.fs = fs
        self.chunk = chunk
        self.history = np.zeros((displayL,))

        self.q = queue.Queue()

    def start_thread(self):
        self.thread = Thread(target=self.DAQ)
        self.thread.start()
        self.thread.join()

    # def stop_thread(self):
    #     self.thread.join()

    def DAQ(self):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f"{self.device}/{self.channel}")
            task.timing.cfg_samp_clk_timing(self.fs,
                                            sample_mode=AcquisitionType.CONTINUOUS,
                                            samps_per_chan=self.chunk)
            data = task.read(number_of_samples_per_channel=self.chunk)
            self.q.put(data)


class audio():
    def __init__(self, name, device, chunk, displayL):
        self.name = name  # "HY"
        self.fs = 44100  # Default for audio
        self.chunk = chunk
        self.history = np.zeros((displayL,))

        self.q = queue.Queue()

        self.p = pyaudio.PyAudio()
        print("audio device:", device)
        self.inStream = self.p.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.fs,
                                    input=True,
                                    frames_per_buffer=self.chunk,
                                    input_device_index=device)

    def start_thread(self):
        self.thread = Thread(target=self.audio_DAQ)
        self.thread.start()
        self.thread.join()

    # def stop_thread(self):
    #     self.p.close(self.inStream)
    #     self.thread.join()

    def audio_DAQ(self):
        audioString = self.inStream.read(self.chunk)
        data = np.fromstring(audioString, dtype=np.int16)/1000
        data = data.tolist()
        self.q.put(data)


class plotting():
    def __init__(self, sensor, canvas, canvas_2, updateInterval, linewidth, ymin, ymax, tick, detect=None, btn_status=None, predict=None, btn_loc=None):
        self.sensor = sensor
        self.q = sensor.q

        self.detect = detect
        self.btn_status = btn_status
        self.predict = predict
        self.predicted = False
        self.btn_loc = btn_loc
        self.loc = round(np.random.rand() * 10, 2)

        self.history = sensor.history

        self.fs = sensor.fs

        self.canvas = canvas

        self.linewidth = linewidth

        self.ref_plot = None


        self.timer = QtCore.QTimer()
        self.updateInterval = updateInterval

        self.recording = False
        self.recorded = []

        self.ymin = ymin
        self.ymax = ymax
        self.tick = tick

        self.second_plot = True
        if self.second_plot:
            self.history_2 = None
            self.canvas_2 = canvas_2
            self.ref_plot_2 = None

    def start_thread(self):
        self.thread = Thread(target=self.start)
        self.thread.start()
        self.thread.join()

    def start(self):
        block = True  # Works for vibra if True
        try:
            new_data = self.q.get(block=block)
            # print("shape:|", np.array(new_data).shape)

            if self.detect:
                if any(ele > 1.5 for ele in new_data):
                    gui_display.update_button(self.btn_status, 'red', 'LEAK DETECTED!')
                else:
                    gui_display.update_button(self.btn_status, 'green', 'normal')
            if self.predict:
                # self.loc = round(np.random.rand() * 10, 2)
                if any(ele > 1.5 for ele in new_data):
                    # if self.loc==None:
                    text = str(self.loc)+'m'
                    # self.predicted = True
                    gui_display.update_button(self.btn_loc, 'red', text)
                else:
                    gui_display.update_button(self.btn_loc, 'green', 'normal')
            ###########################
            shift = len(new_data)
            if self.second_plot:
                self.history_2 = self.history
            self.history = np.roll(self.history, -shift, axis=0)
            self.history[-shift:, ] = new_data
            self.y = self.history[:]
            self.canvas.axes.set_facecolor((0, 0, 0))

            if self.recording:
                self.recorded.extend(new_data)

            if self.ref_plot is None:
                plot_refs = self.canvas.axes.plot(self.y, color=(1, 1, 1), linewidth=self.linewidth)
                self.ref_plot = plot_refs[0]
            else:
                self.ref_plot.set_ydata(self.y)

            if self.second_plot:
                self.y_2 = self.history_2[:]
                self.canvas_2.axes.set_facecolor((0, 0, 0))
                if self.ref_plot_2 is None:
                    plot_refs_2 = self.canvas_2.axes.plot(self.y_2, color=(1, 1, 1), linewidth=self.linewidth)
                    self.ref_plot_2 = plot_refs_2[0]
                else:
                    self.ref_plot_2.set_ydata(self.y_2)
            ###########################################
        except queue.Empty:
            print("empty")

        # Time domain
        self.canvas.axes.yaxis.grid(True, linestyle='--')
        start, end = self.canvas.axes.get_ylim()
        self.canvas.axes.yaxis.set_ticks(np.arange(start, end, self.tick))
        self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #     self.canvas.axes.relim()
        #     self.canvas.axes.autoscale_view()
        self.canvas.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)
        self.canvas.draw()

        if self.second_plot:
            self.canvas_2.axes.yaxis.grid(True, linestyle='--')
            start, end = self.canvas_2.axes.get_ylim()
            self.canvas_2.axes.yaxis.set_ticks(np.arange(start, end, self.tick))
            self.canvas_2.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            #     self.canvas.axes.relim()
            #     self.canvas.axes.autoscale_view()
            self.canvas_2.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)
            self.canvas_2.draw()

        # Freq domain
        # L = len(self.history)
        # T = 1/self.fs
        # fft_values = np.fft.fft(self.y)
        # fft_magnitude = np.abs(fft_values) / L  # Normalize the magnitude
        # fft_freq = np.fft.fftfreq(L, T)
        # self.canvasf.axes.plot(fft_freq[:L // 2], fft_magnitude[:L // 2] * 2)
        # self.canvasf.draw()
        # self.canvasf.cla()

        # X = fft(self.history)
        # N = len(X)
        # n = np.arange(N)
        # T = N / self.fs
        # freq = n / T
        # self.canvasf.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
        # self.canvasf.xlabel('Freq (Hz)')
        # self.canvasf.ylabel('FFT Amplitude |X(freq)|')
        # # self.canvasf.xlim(0, 10)


def quickplot(data, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), data, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(filename)
    plt.show()


def quickfft(data, fs):
    spacing = 1/fs
    N = len(data)  # Number of samples

    # When the input is purely real, the output is Hermitian-symmetric
    a = pyfftw.empty_aligned(len(data), dtype='complex128', n=16)
    a[:] = data
    # yf = rfft(data)  # Half of the conjugate pairs
    yf = pyfftw.interfaces.numpy_fft.rfft(a)  # Same as np.fft.rfft(a)
    print("fft len:", len(yf))
    freq = np.fft.rfftfreq(N, spacing)  # Frequencies
    print("freq len:", len(freq))

    return freq, yf


def plotfft(data, fs, freq, yf, folder_path, name, lower_f=None, upper_f=None):
    N = len(data)

    if lower_f and upper_f:
        idx_L = np.searchsorted(freq, lower_f)
        idx_U = np.searchsorted(freq, upper_f+0.1, side='left')  # Assume upper_f > 1Hz

        yf = yf[idx_L:idx_U]
        freq = freq[idx_L:idx_U]

    amp = np.abs(yf) / N  # Normalize the magnitude
    print("FFT is computed.")

    plt.figure(figsize=(12, 6))

    plt.subplot(122)
    plt.plot(range(len(data)), data, 'r')
    plt.xlabel('Time (1/'+str(int(fs))+' s)')
    plt.ylabel('Amplitude')
    plt.title(name)

    plt.subplot(121)
    plt.stem(freq, amp, 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Normalized Amplitude: |X(freq)| / N')
    plt.title(name)

    plt.tight_layout()

    # Tk().withdraw()
    # folder_path = askdirectory(title="Select a folder for saving FFT figure")
    filenameNpath = folder_path + "/" + name + '.png'
    plt.savefig(filenameNpath)
    print('figure saved:', filenameNpath)

    # plt.show()
    plt.close()