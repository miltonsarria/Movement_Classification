import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import lfilter, resample, firwin, hilbert
from scipy.fftpack import fft
import math
import pickle
from tools import *


data_file ="data/M1_flexion/P3S1D1M1C1.txt"
fil_dc = True
alpha  = 0.89
fs     = 1000
data0 = np.loadtxt(data_file)
data =  lfilter([1,-1], [1, -alpha], data0)
data[:40] = 0
t = np.arange(data.size)/fs
plt.plot(t,data)
plt.show()

