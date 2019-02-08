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

# Loads the meta data file that will map signal data to labels
def load_metadata(source_meta):  
    df = pd.read_csv(source_meta)
    min_id = min(df.signal_id)
    max_id = max(df.signal_id)
    return df, min_id, max_id


# Create Feature Data Frame to Store Extracted Features
def create_feature_matrix():
    feature_matrix_columns = ["signal_id", "entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "fault"]
    feature_matrix = pd.DataFrame([], columns=feature_matrix_columns)
    return feature_matrix, feature_matrix_columns


# Returns a dataframe with the three phase measurement time series
def load_sample(source_data, idx_start, idx_stop):  
    df = pq.read_pandas(source_data, columns=[str(i) for i in range(idx_start, idx_stop + 1)]).to_pandas()
    return df


def discete_wavelet_transform(signal, wavelet="db4", level=1):  # Discrete Wavelet Transform on the Signal to Remove Noise
    coeff = pywt.wavedec(signal, wavelet, mode="per")  # calculate the wavelet coefficients
    sigma = mad(coeff[-level])  # calculate a threshold
    uthresh = sigma * np.sqrt(2*np.log(len(signal)))  # changing this threshold also changes the behavior, but hasn't been adjusted much
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    dwt_signal = pywt.waverec(coeff, wavelet, mode="per")  # reconstruct the signal using the thresholded coefficients
    return dwt_signal


def detrend_signal(signal):  # Detrend Sinusoidal Behavior to "flatten" signal
    return np.diff(signal, n=1, axis=-1)  # Calculate the n-th discrete difference along the given axis


#  Feature Extraction Functions
def calculate_entropy(signal):
    counter_values = collections.Counter(signal).most_common()
    probabilities = [elem[1]/len(signal) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(signal):
    n5 = np.nanpercentile(signal, 5)
    n25 = np.nanpercentile(signal, 25)
    n75 = np.nanpercentile(signal, 75)
    n95 = np.nanpercentile(signal, 95)
    median = np.nanpercentile(signal, 50)
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    var = np.nanvar(signal)
    rms = np.nanmean(np.sqrt(signal**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 

def calculate_crossings(signal):
    zero_crossing_indices = np.nonzero(np.diff(np.array(signal) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(signal) > np.nanmean(signal)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]  # TODO no_mean_crossings a very promising feature!!!!


def find_all_peaks(signal, threshold=0.7, min_distance=0):
    #peaks = peakutils.indexes(1.0*(signal), thres=threshold, min_dist=min_distance)
    peaks = np.argwhere(signal > 1.0*threshold).tolist()
    #valleys = peakutils.indexes(-1.0*(signal), thres=threshold, min_dist=min_distance)
    valleys = np.array(np.argwhere(signal < -1.0*threshold)).tolist()
    pois = np.sort(np.concatenate((peaks, valleys)))
    peak_indexes = []
    for pk in pois:
        peak_indexes.append(pk[0])
    return np.sort(peak_indexes)


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
    threshold = 40  # amplitude threshld for determining high amplitude peaks for cancellation
    #peaks = peakutils.indexes(1.0*(signal), thres=0.80, min_dist=0)
    peaks = np.argwhere(signal > 1.0*threshold).tolist()
    #valleys = peakutils.indexes(-1.0*(signal), thres=0.80, min_dist=0)
    valleys = np.argwhere(signal < -1.0*threshold).tolist()
    hi_amp_pk_indexes = np.sort(np.concatenate((peaks, valleys)))
    for pk_idx in hi_amp_pk_indexes:
        if not pk_idx[0] in false_peak_indexes:
            false_peak_indexes.append(pk_idx[0])
    return false_peak_indexes


def cancel_flagged_peaks(peak_indexes, false_peak_indexes):
    true_peak_indexes = list(set(peak_indexes) - set(false_peak_indexes))
    true_peak_indexes.sort()
    return true_peak_indexes


def calculate_peaks(signal, true_peak_indexes):  # Peak Characteristics on True Peaks
    peak_values = signal[true_peak_indexes]
    num_detect_peak = len(true_peak_indexes)
    if num_detect_peak > 0:
        min_height = min(peak_values)
        max_height = max(peak_values)
        mean_height = np.nanmean(peak_values)
        min_width, max_width, mean_width, num_true_peaks = calculate_peak_widths(true_peak_indexes)
        return [min_height, max_height, mean_height, min_width, max_width, mean_width, num_detect_peak, num_true_peaks]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0]


def get_features(signal, signal_id, threshold, min_distance): # Extract features from the signal and build an array of them
    peak_indexes = find_all_peaks(signal, threshold, min_distance)
    print("Now processing signal_id: "+str(signal_id)+" with peak detection threshold at "+str(threshold)+" yielding "+str(len(peak_indexes))+" peaks at "+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    false_peak_indexes = cancel_false_peaks(signal, peak_indexes)
    false_peak_indexes = cancel_high_amp_peaks(signal, peak_indexes, false_peak_indexes)
    true_peak_indexes = cancel_flagged_peaks(peak_indexes, false_peak_indexes)

    entropy = calculate_entropy(signal)
    crossings = calculate_crossings(signal)
    statistics = calculate_statistics(signal)
    peaks = calculate_peaks(signal, true_peak_indexes)
    return [entropy] + crossings + statistics + peaks


def vsb_feature_extraction(source_meta, source_data, data_type, dwt_type, peak_threshold, peak_min_distance):  
    df_meta, min_id, max_id = load_metadata(source_meta)  # retrieve relevant meta data for training or test
    feature_matrix, feature_matrix_columns = create_feature_matrix()  # create new feature matrix data frame

    for signal_id in range(min_id, max_id+1):
        df_signal = pq.read_pandas(source_data, columns=[str(i) for i in range(signal_id, signal_id + 1)]).to_pandas()
        raw_signal = df_signal[str(signal_id)]  # Raw Signal from Data
        dwt_signal = discete_wavelet_transform(raw_signal, wavelet=dwt_type, level=1)  # Denoise the Raw Signal with Discrete Wavelet Transform       
        dwt_detrend_signal = detrend_signal(dwt_signal)  # Detrend the Transformed Signal to "Flatten" it
        signal_features = get_features(dwt_detrend_signal, signal_id, peak_threshold, peak_min_distance)  # Perform Feature Extraction on the Transformed, "Flattened" Signal
        
        if data_type.lower() == "train":  # Stage Feature Array for Addition to Feature Matrix
            df_features = pd.DataFrame([[signal_id] + signal_features + [df_meta.target[df_meta.signal_id == signal_id].values[0]]], columns=feature_matrix_columns)
        else:  # for test data
            df_features = pd.DataFrame([[signal_id] + signal_features + [np.NaN]], columns=feature_matrix_columns)
        feature_matrix = feature_matrix.append(df_features, ignore_index=True)  # Append Feature Matrix Data Frame

    # After processing and extracting features from each signal in the test set, save feature matrix
    feature_matrix.to_csv("extracted_features/"+data_type+"_features_thresh_"+str(peak_threshold)+"_"+dwt_type+".csv", sep=",")



# Discrete Wavelet Transform Types
Discrete_Meyer = ["dmey"]
Daubechies = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11", "db12", "db13", "db14", "db15", "db16", "db17", "db18", "db19", "db20"]
Symlets = ["sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10", "sym11", "sym12", "sym13", "sym14", "sym15", "sym16", "sym17", "sym18", "sym19", "sym20"]
Coiflet = ["coif1", "coif2", "coif3", "coif4", "coif5"]
Biorthogonal = ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8"]
Reverse_Biorthogonal = ["rbio1.1", "rbio1.3", "rbio1.5", "rbio1.2", "rbio1.4", "rbio1.6", "rbio1.8", "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", "rbio3.9", "rbio4.4", "rbio5.5", "rbio6.8"]


dwt_type = "db4"  # Wavelets chosen for processing 
#thresholds = [0.71, 0.69, 0.67, 0.65, 0.63, 0.61]  # Thresholds for peakutils.indexes() function
thresholds = [10.0, 7.0, 5.0]  # Thresholds for np.argwhere(signal <> threshold) method of detecting peaks
peak_min_distance = 0  # minumum distance required between peak detections
run_test_data = False

# Raw Data Sources
if run_test_data:
    source_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/test.parquet"
    source_meta = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/metadata_test.csv"
    data_type = "test"
else:
    source_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/train.parquet"
    source_meta = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/metadata_train.csv"
    data_type = "train"


for peak_threshold in thresholds:
    print("Starting signal processing and feature extraction on "+data_type+" data with the "+dwt_type+" transform and threshold = "+str(peak_threshold)+" at "+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    vsb_feature_extraction(source_meta, source_data, data_type, dwt_type, peak_threshold, peak_min_distance)
print("Done! at "+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))