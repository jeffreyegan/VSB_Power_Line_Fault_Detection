import os
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import scipy
import pywt
import peakutils
from statsmodels.robust import mad
import collections
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import leastsq


def discete_wavelet_transform(signal, wavelet="db4", level=1):  # Discrete Wavelet Transform on the Signal to Remove Noise

    # Discrete Wavelet Transform Types (largely left for reference)
    Discrete_Meyer = ["dmey"]
    Daubechies = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11", "db12", "db13", "db14", "db15", "db16", "db17", "db18", "db19", "db20"]
    Symlets = ["sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10", "sym11", "sym12", "sym13", "sym14", "sym15", "sym16", "sym17", "sym18", "sym19", "sym20"]
    Coiflet = ["coif1", "coif2", "coif3", "coif4", "coif5"]
    Biorthogonal = ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8"]
    Reverse_Biorthogonal = ["rbio1.1", "rbio1.3", "rbio1.5", "rbio1.2", "rbio1.4", "rbio1.6", "rbio1.8", "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", "rbio3.9", "rbio4.4", "rbio5.5", "rbio6.8"]

    # Discrete Wavelet Transform on the Signal to Remove Noise
    coeff = pywt.wavedec(signal, wavelet, mode="per")  # calculate the wavelet coefficients
    sigma = mad(coeff[-level])  # calculate a threshold
    uthresh = sigma * np.sqrt(2*np.log(len(signal)))  # changing this threshold also changes the behavior, but hasn't been adjusted much
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    dwt_signal = pywt.waverec(coeff, wavelet, mode="per")  # reconstruct the signal using the thresholded coefficients
    return dwt_signal


def fit_sinusoid(signal):  # Fit a sinusoidal function to the data
    t = np.linspace(0, 2*np.pi, len(signal))  # data covers one period, thus 2pi
    guess_mean = np.mean(signal)
    guess_std = 3*np.std(signal)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_freq = 50
    guess_amp = 20
    optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - signal
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
    return est_amp*np.sin(est_freq*t+est_phase) + est_mean  # signal_fit


def find_pd_probable(signal_fit, condition):
    first_derivative = np.gradient(signal_fit)
    return [i for i, elem in enumerate(first_derivative) if condition(elem)]  #PD Prob Region
    

def detrend_signal(signal, high_prob_idx):  # Detrend Sinusoidal Behavior to "flatten" signal
    x = np.diff(signal, n=1, axis=-1)  # Calculate the n-th discrete difference along the given axis
    if max(high_prob_idx) == len(x):
        high_prob_idx = high_prob_idx[0:-1]
    #return x[high_prob_idx]  # Detrended PD Prob Region
    
    temp = x[high_prob_idx]
    if len(temp) >= 400000:
        return temp[0:400000]
    else:
        temp2 = np.zeros((400000))
        temp2[0:len(temp)] = temp
        return temp2


def norm_int8_vals(ts, min_data=-128, max_data=127, range_needed=(-1,1)): # Normalizes int8 data samples to (-1, 1)
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:    
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]
