
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

def load_feature_data(file_name):
    df = pd.read_csv(file_name)
    return df


def classification_random_forest_model(features, labels, n_value):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=n_value)  # Create Gaussian Classifier
    classifier.fit(features, labels.values.ravel())
    #y_predicted = classifier.predict(x_test)
    return classifier

def classification_random_forest_predict(model, x_test):
    y_predicted = model.predict(x_test)
    return y_predicted


# Train Model with Full Set
dwt = "db4"
number_estimators = 50

training_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/train_features_"+dwt+".csv"
df_train = pd.read_csv(training_data)
features = df_train[["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]]
labels = df_train[["fault"]]
classifier = classification_random_forest_model(features, labels, number_estimators)


# Make Predictions
test_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/test_features_"+dwt+".csv"
test_meta = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/source_data/metadata_test.csv"
df_test = pd.read_csv(test_data).drop(['Unnamed: 0'],axis=1)
test_features = df_test[["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]]
predicted_faults = classification_random_forest_predict(classifier, test_features)
df_test["fault"] = predicted_faults


# Make Submission File
submission_filename = "prediction_submission_"+dwt+".csv"

#submission = df_test[["signal_id", "fault"]]
#submission.to_csv(submission_filename, sep=",", index="False")

f_o = open(submission_filename, "w+")
f_o.write("signal_id,target\n")
for idx in range(len(df_test)):
    signal_id = df_test["signal_id"][idx]
    fault = df_test["fault"][idx]
    f_o.write(str(signal_id)+","+str(fault)+"\n")

f_o.close()