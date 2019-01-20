

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np



def load_feature_data(file_name):
    df = pd.read_csv(file_name)
    return df

def split_data(features, labels, random_state_value):
    from sklearn.model_selection import train_test_split
    # Using standard split of 80-20 training to testing data split ratio and fixing random_state=1 for repeatability
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=random_state_value)
    return x_train, x_test, y_train, y_test

def matthews_corr_coef(c_matrix):  # Use 2x2 Confusion Matrix to Calculate Matthews Correlation Coefficient
    TP = c_matrix[0][0]  # True Positives
    TN = c_matrix[0][1]  # True Negavitves
    FP = c_matrix[1][0]  # False Positives
    FN = c_matrix[1][1]  # False Negatives
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print(MCC)
    return MCC

def score_classifier(truth, predictions):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    from sklearn.metrics import confusion_matrix
    m_accuracy = accuracy_score(truth, predictions)
    m_recall = recall_score(truth, predictions)
    m_precision = precision_score(truth, predictions)
    m_f1 = f1_score(truth, predictions)
    c_matrix = confusion_matrix(truth, predictions)
    return m_accuracy, m_recall, m_precision, m_f1, c_matrix

def classification_random_forest(features, labels, n_value, random_seed):
    from sklearn.ensemble import RandomForestClassifier
    x_train, x_test, y_train, y_test = split_data(features, labels, random_seed)
    classifier = RandomForestClassifier(n_estimators=n_value)  # Create Gaussian Classifier
    classifier.fit(x_train, y_train.values.ravel())
    y_predicted = classifier.predict(x_test)
    m_accuracy, m_recall, m_precision, m_f1, c_matrix = score_classifier(y_test, y_predicted.ravel())
    mcc = matthews_corr_coef(c_matrix)
    return m_accuracy, m_recall, m_precision, m_f1, mcc


# Monte Carlo Trial Random Seeds
random_states = list(range(1, 51, 5))  # 10 seeds
num_estimators_range = [15, 20, 25, 30]

# Discrete Wavelet Transform Types
Discrete_Meyer = ["dmey"]
Daubechies = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11", "db12", "db13", "db14", "db15", "db16", "db17", "db18", "db19", "db20"]
Symlets = ["sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10", "sym11", "sym12", "sym13", "sym14", "sym15", "sym16", "sym17", "sym18", "sym19", "sym20"]
Coiflet = ["coif1", "coif2", "coif3", "coif4", "coif5"]
Biorthogonal = ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8"]
Reverse_Biorthogonal = ["rbio1.1", "rbio1.3", "rbio1.5", "rbio1.2", "rbio1.4", "rbio1.6", "rbio1.8", "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", "rbio3.9", "rbio4.4", "rbio5.5", "rbio6.8"]
dwt_types = Discrete_Meyer + Coiflet + Daubechies[1:4] + Symlets[1:4] + Daubechies[5:6] # DWTs used to extract features so far

# Run Monte Carlo Trials
monte_df_cols = ["dwt_type", "random_seed", "num_estimators", "accuracy", "recall", "precision", "f1_score", "matthews_corr_coef"]
monte_df = pd.DataFrame([], columns=monte_df_cols)

for dwt in dwt_types:
    print("Starting monte carlo trials for the "+dwt+" transform at "+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for number_estimators in num_estimators_range:
        for seed in random_states:
            file_name = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/train_features_"+dwt+".csv"
            df = load_feature_data(file_name)
            features = df[["entropy", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings"]]
            labels = df[["fault"]]
            m_accuracy, m_recall, m_precision, m_f1, mcc = classification_random_forest(features, labels, number_estimators, seed)
            trial_results = pd.DataFrame([[dwt, seed, number_estimators, m_accuracy, m_recall, m_precision, m_f1, mcc]], columns=monte_df_cols)
            monte_df = monte_df.append(trial_results, ignore_index=True)
monte_df.to_csv("random_forest_monte_carlo_trials.csv", sep=",")
print("Done! at "+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))