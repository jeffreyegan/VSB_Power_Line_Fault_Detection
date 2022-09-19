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


#  Feature Extraction Functions
def calculate_entropy(signal):
    counter_values = collections.Counter(signal).most_common()
    probabilities = [elem[1]/len(signal) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return np.array([entropy])


def calculate_statistics(signal):
    statistics = np.array([])
    mean = np.nanmean(signal)
    statistics = np.append(statistics, mean)
    std = np.nanstd(signal)
    statistics = np.append(statistics, std)
    std_top = mean + std
    statistics = np.append(statistics, std_top)
    std_bot = mean - std
    statistics = np.append(statistics, std_bot)
    var = np.nanvar(signal)
    statistics = np.append(statistics, var)
    rms = np.nanmean(np.sqrt(signal**2))
    statistics = np.append(statistics, rms)
    percentiles = np.nanpercentile(signal, [0, 1, 25, 50, 75, 99, 100]) 
    statistics = np.append(statistics, percentiles)
    max_range = percentiles[-1] - percentiles[0]
    statistics = np.append(statistics, max_range)
    relative_percentiles = percentiles - mean
    statistics = np.append(statistics, relative_percentiles)
    return statistics
 

def calculate_crossings(signal):
    zero_crossing_indices = np.nonzero(np.diff(np.array(signal) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(signal) > np.nanmean(signal)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return np.array([no_zero_crossings, no_mean_crossings])  #no_mean_crossings a very promising feature


def find_all_peaks(signal, threshold=0.7, min_distance=0):
    #peaks = peakutils.indexes(1.0*(signal), thres=threshold, min_dist=min_distance)
    peaks = np.argwhere(signal > 1.0*threshold)
    #valleys = peakutils.indexes(-1.0*(signal), thres=threshold, min_dist=min_distance)
    valleys = np.array(np.argwhere(signal < -1.0*threshold))
    pois = np.sort(np.concatenate((peaks, valleys)))
    peak_indexes = []
    for pk in pois:
        #peak_indexes.append(pk)
        peak_indexes.append(pk[0])
    return np.sort(peak_indexes), peaks, valleys


def calculate_peak_widths(peak_idxs):
    tmp_w = 1
    widths = []
    for idx in range(1,len(peak_idxs)):
        if peak_idxs[idx]-peak_idxs[idx-1] < 3:
            tmp_w +=1
        else:
            widths.append(tmp_w)
            tmp_w = 1
    widths.append(tmp_w)
    min_width = min(np.array(widths))
    max_width = max(np.array(widths))
    mean_width = np.nanmean(np.array(widths))
    num_true_peaks = len(widths)
    return min_width, max_width, mean_width, num_true_peaks


def cancel_false_peaks(signal, peak_indexes):
    false_peak_indexes = []
    max_sym_distance = 10  #
    max_pulse_train = 500  # 
    max_height_ratio = 0.25  # 
    for pk in range(len(peak_indexes)-1):
        if not peak_indexes[pk] in false_peak_indexes:
            if (signal[peak_indexes[pk]] > 0 and signal[peak_indexes[pk+1]] < 0) and (peak_indexes[pk+1] - peak_indexes[pk]) < max_sym_distance:  # opposite polarity and within symmetric check distance
                if min(abs(signal[peak_indexes[pk]]),abs(signal[peak_indexes[pk+1]]))/max(abs(signal[peak_indexes[pk]]),abs(signal[peak_indexes[pk+1]])) > max_height_ratio:  # ratio of opposing polarity check
                    scrub = list(x for x in range(len(peak_indexes)) if peak_indexes[pk] <= peak_indexes[x] <= peak_indexes[pk]+max_pulse_train)  # build pulse train
                    for x in scrub:
                        false_peak_indexes.append(peak_indexes[x])

            if (signal[peak_indexes[pk]] < 0 and signal[peak_indexes[pk+1]] > 0) and (peak_indexes[pk+1] - peak_indexes[pk]) < max_sym_distance:
                if min(abs(signal[peak_indexes[pk]]),abs(signal[peak_indexes[pk+1]]))/max(abs(signal[peak_indexes[pk]]),abs(signal[peak_indexes[pk+1]])) > max_height_ratio:
                    scrub = list(x for x in range(len(peak_indexes)) if peak_indexes[pk] <= peak_indexes[x] <= peak_indexes[pk]+max_pulse_train)
                    for x in scrub:
                        false_peak_indexes.append(peak_indexes[x])
    return false_peak_indexes


def cancel_high_amp_peaks(signal, peak_indexes, false_peak_indexes):
    threshold = 50/128  # amplitude threshld for determining high amplitude peaks for cancellation
    #peaks = peakutils.indexes(1.0*(signal), thres=0.80, min_dist=0)
    peaks = np.argwhere(signal > 1.0*threshold)
    #valleys = peakutils.indexes(-1.0*(signal), thres=0.80, min_dist=0)
    valleys = np.argwhere(signal < -1.0*threshold)
    hi_amp_pk_indexes = np.sort(np.concatenate((peaks, valleys)))
    for pk_idx in hi_amp_pk_indexes:
        if not pk_idx[0] in false_peak_indexes:
            false_peak_indexes.append(pk_idx[0])
    return false_peak_indexes


def cancel_flagged_peaks(peak_indexes, false_peak_indexes):
    true_peak_indexes = list(set(peak_indexes) - set(false_peak_indexes))
    true_peak_indexes.sort()
    return true_peak_indexes


def low_high_peaks(signal, true_peak_indexes, hi_idx, lo_idx):
    if np.size(np.array(hi_idx))> 0:
        lhr = 1.0*np.size(np.array(lo_idx))/np.size(np.array(hi_idx))
    else:
        lhr = 0.0
    hi_true = 0
    lo_true = 0
    for x in true_peak_indexes:
        if signal[x] > 0.0:
            hi_true += 1
        else:
            lo_true += 1
    if hi_true > 0:
        lhrt = 1.0*lo_true/hi_true
    else:
        lhrt = 0.0
    #return np.array([hi_idx, lo_idx, lhr, hi_true, lo_true, lhrt])
    return np.array([lhr, hi_true, lo_true, lhrt])


def calculate_peaks(signal, true_peak_indexes):  # Peak Characteristics on True Peaks
    peak_values = signal[true_peak_indexes]
    num_detect_peak = len(true_peak_indexes)
    if num_detect_peak > 0:
        min_height = min(peak_values)
        max_height = max(peak_values)
        mean_height = np.nanmean(peak_values)
        min_width, max_width, mean_width, num_true_peaks = calculate_peak_widths(true_peak_indexes)
        return np.array([min_height, max_height, mean_height, min_width, max_width, mean_width, num_detect_peak, num_true_peaks])
    else:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0])
