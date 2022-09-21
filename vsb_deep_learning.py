
import os
from typing_extensions import dataclass_transform
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle

#import torch
#import torch.nn as nn
#import torch.nn.functional as F

from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, roc_curve, auc, classification_report, recall_score, precision_recall_curve

from vsb_signal_processing import discete_wavelet_transform, fit_sinusoid, find_pd_probable, detrend_signal, norm_int8_vals
from vsb_feature_extraction import *


class VSB_Dataset():
    def __init__(self, data_type="train", batch_size=36):

        self.data_type = str(data_type).lower()
        self.source_meta = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/metadata_"+self.data_type+".csv"
        self.source_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/"+self.data_type+".parquet"
        self.load_meta()

        try:
            self.X = np.load(os.path.join("extracted_features", "X_"+self.data_type+".npy"))
            self.y = np.load(os.path.join("extracted_features", "y_"+self.data_type+".npy"))
            print("Loaded Pre-Processed Feature Data")
        except:
            print("Pre-Processing Signals...")
            self.sig_len = 400000
            self.X, self.y = self.process_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_meta(self):  # Loads the meta data file that will map signal data to labels
        self.meta = pd.read_csv(self.source_meta)
        self.min_measurement = min(self.meta.id_measurement)
        self.max_measurement = max(self.meta.id_measurement)
        self.min_signal_id= min(self.meta.signal_id)
        self.max_signal_id = max(self.meta.signal_id)


    def feature_extraction(self, signal, n_dim=160):
        bucket_size = int(self.sig_len / n_dim)
        features = []

        for i in range(0, self.sig_len, bucket_size):
            signal_chunk = signal[i:i+bucket_size]
            feature_array = np.array(())  # intermediate array, maintained per bucket

            # calculate basic signal statistics
            feature_array = np.append(feature_array, calculate_entropy(signal_chunk))  # entropy
            feature_array = np.append(feature_array, calculate_statistics(signal_chunk))  # statistics
            feature_array = np.append(feature_array, calculate_crossings(signal_chunk))  # calculate crossings

            # peaks processing
            peak_indexes, hi_idx, lo_idx = find_all_peaks(signal_chunk, threshold=4/128, min_distance=0)
            false_peak_indexes = cancel_false_peaks(signal_chunk, peak_indexes)
            false_peak_indexes = cancel_high_amp_peaks(signal_chunk, peak_indexes, false_peak_indexes)
            true_peak_indexes = cancel_flagged_peaks(peak_indexes, false_peak_indexes)
            peaks = calculate_peaks(signal_chunk, true_peak_indexes)
            low_high_stats = low_high_peaks(signal_chunk, true_peak_indexes, hi_idx, lo_idx)
            feature_array = np.append(feature_array, peaks) 
            feature_array = np.append(feature_array, low_high_stats) 

            features.append(feature_array)

        return np.asarray(features)


    def signal_processing(self, raw_signal, dwt_type="db4"):
        norm_signal = norm_int8_vals(raw_signal)  # Normalizes int8 data samples from (-128, 127) to (-1, 1)
        dwt_signal = discete_wavelet_transform(norm_signal, wavelet=dwt_type, level=1)  # Denoise the Raw Signal with Discrete Wavelet Transform
        fit_signal = fit_sinusoid(dwt_signal)  # Fit a sinusoidal function to the de-noised signal data
        high_prob_region = find_pd_probable(fit_signal, lambda e: e>0)  # Identify the PD-Probable Region
        dwt_detrend_pd_reg_signal = detrend_signal(dwt_signal, high_prob_region)  # Detrend the PD-Probable Region of the Transformed Signal to "Flatten" it
        ts = dwt_detrend_pd_reg_signal  # final time series of signal data before feature extraction
        return ts


    def process_data(self):
        signal_data = pq.read_pandas(self.source_data, columns=[str(i) for i in range(self.min_signal_id, self.max_signal_id)]).to_pandas()
        X = [] #np.array(ndmin=2)
        y = []
        for measurement_id in tqdm(range(self.min_measurement, self.max_measurement)):
            X_features = []
            for phase in [0,1,2]:
                target = self.meta[(self.meta['id_measurement']==measurement_id) & (self.meta['phase']==phase)]['target'].iloc[0]
                signal_id = self.meta[(self.meta['id_measurement']==measurement_id) & (self.meta['phase']==phase)]['signal_id'].iloc[0]
                if phase == 0:
                    y.append(target)
                processed_signal = self.signal_processing(signal_data[str(signal_id)])
                features = self.feature_extraction(processed_signal)
                X_features.append(features)
            X_features = np.concatenate(X_features, axis=1)
            X.append(X_features)
        X = np.asarray(X)
        y = np.asarray(y)
        print(X.shape, y.shape)
        np.save(os.path.join("extracted_features", "X_"+self.data_type+".npy"), X)
        np.save(os.path.join("extracted_features", "y_"+self.data_type+".npy"), y)
        return X, y


data = VSB_Dataset(data_type="train")

print(np.shape(data.X))
print(np.shape(data.y))
