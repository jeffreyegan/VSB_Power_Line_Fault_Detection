import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

from vsb_signal_processing import discete_wavelet_transform, fit_sinusoid, find_pd_probable, detrend_signal, norm_int8_vals
from vsb_feature_extraction import *


class VSB_Dataset():
    def __init__(self, data_type="train", batch_size=16):

        self.data_type = str(data_type).lower()
        self.source_meta = os.path.join("source_data","metadata_"+self.data_type+".csv")
        self.source_data = os.path.join("source_data",self.data_type+".parquet")
        self.load_meta()

        self.batch_size = batch_size

        try:
            self.X = np.load(os.path.join("extracted_features", "X_"+self.data_type+"_features.npy"))
            self.y = np.load(os.path.join("extracted_features", "y_"+self.data_type+"_features.npy"))
            print("Loaded Pre-Processed Feature Data")
        except:
            print("Pre-Processing Signals...")
            self.sig_len = 400000
            self.X, self.y = self.process_data()

        print("Splitting and Normalizing Feature Data...")
        self.split_normalize()
        print("Building Loader...")
        self.make_loader(num_workers=2)


    def load_meta(self):  # Loads the meta data file that will map signal data to labels
        self.meta = pd.read_csv(self.source_meta)
        self.min_measurement = min(self.meta.id_measurement)
        self.max_measurement = max(self.meta.id_measurement)
        self.min_signal_id= min(self.meta.signal_id)
        self.max_signal_id = max(self.meta.signal_id)


    def signal_processing(self, raw_signal, dwt_type="db4"):
        norm_signal = norm_int8_vals(raw_signal)  # Normalizes int8 data samples from (-128, 127) to (-1, 1)
        dwt_signal = discete_wavelet_transform(norm_signal, wavelet=dwt_type, level=1)  # Denoise the Raw Signal with Discrete Wavelet Transform
        fit_signal = fit_sinusoid(dwt_signal)  # Fit a sinusoidal function to the de-noised signal data
        high_prob_region = find_pd_probable(fit_signal, lambda e: e>0)  # Identify the PD-Probable Region
        dwt_detrend_pd_reg_signal = detrend_signal(dwt_signal, high_prob_region)  # Detrend the PD-Probable Region of the Transformed Signal to "Flatten" it
        ts = dwt_detrend_pd_reg_signal  # final time series of signal data before feature extraction
        return ts


    def feature_extraction(self, signal, n_part=400):
        length = len(signal)
        n_feat = 36  #TODO len of all features used
        pool = np.int32(np.ceil(length/n_part))
        feature_array = np.zeros((n_part, n_feat))
        for j, i in enumerate(range(0,length, pool)):
            if i+pool < length:
                k = signal[i:i+pool]
            else:
                k = signal[i:]
            feature_array[j, 0] = calculate_entropy(k)  # entropy
            feature_array[j, 1:22] = calculate_entropy(k)# statistics
            feature_array[j, 22:24] = calculate_crossings(k)  # calculate crossings

            # peaks processing
            peak_indexes, hi_idx, lo_idx = find_all_peaks(k, threshold=4/128, min_distance=0)
            false_peak_indexes = cancel_false_peaks(k, peak_indexes)
            false_peak_indexes = cancel_high_amp_peaks(k, peak_indexes, false_peak_indexes)
            true_peak_indexes = cancel_flagged_peaks(peak_indexes, false_peak_indexes)
            feature_array[j, 24:32] = calculate_peaks(k, true_peak_indexes) 
            feature_array[j, 32:36] = low_high_peaks(k, true_peak_indexes, hi_idx, lo_idx)
        return feature_array


    def process_data(self):
        signal_data = pq.read_pandas(self.source_data, columns=[str(i) for i in range(self.min_signal_id, self.max_signal_id)]).to_pandas()
        X = []
        y = []
        for i in tqdm(range(self.min_signal_id, self.max_signal_id)):
            #idx = self.meta.loc[self.meta.signal_id==i, 'signal_id'].values.tolist()
            processed_signal = self.signal_processing(signal_data[str(i)])
            y.append(self.meta.loc[self.meta.signal_id==i, 'target'].values)
            X.append(self.feature_extraction(processed_signal, n_part=400))

        
        X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1])
        X = np.transpose(X, [0,2,1]) # Make X shape (batch size, channels, time steps) 
        y = np.array(y).reshape(-1,1)
        print(X.shape, y.shape)
        np.save(os.path.join("extracted_features", "X_"+self.data_type+"_features.npy"), X)
        np.save(os.path.join("extracted_features", "y_"+self.data_type+"_features.npy"), y)
        return X, y


    def split_normalize(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
        (train_idx, val_idx) = next(sss.split(self.X, self.y))

        self.X_train, self.X_val = self.X[train_idx], self.X[val_idx]
        self.y_train, self.y_val = self.y[train_idx], self.y[val_idx]
        print("Train/Val Splits:")
        print(self.X_train.shape)
        print(self.X_val.shape)
        print(self.y_train.shape)
        print(self.y_val.shape)

        scalers = {}
        for i in range(self.X_train.shape[1]):
            scalers[i] = MinMaxScaler(feature_range=(-1, 1))
            self.X_train[:, i, :] = scalers[i].fit_transform(self.X_train[:, i, :]) 

        for i in range(self.X_val.shape[1]):
            self.X_val[:, i, :] = scalers[i].transform(self.X_val[:, i, :]) 
        return


    def make_loader(self, num_workers=2):
        self.loader = DataLoader(list(zip(self.X_train,self.y_train)), batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        return