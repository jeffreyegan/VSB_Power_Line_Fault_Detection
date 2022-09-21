import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


def load_feature_data(file_name):
    df = pd.read_csv(file_name)
    return df


def split_data(features, labels, random_state_value=1):
    from sklearn.model_selection import train_test_split
    # Using standard split of 80-20 training to testing data split ratio and fixing random_state=1 for repeatability
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=random_state_value)
    return x_train, x_test, y_train, y_test
    

def run_rf(data_file, random_state_value):
    df = load_feature_data(data_file)
 
    # Error with Hi_count and Lo_count ... listing full list of peaks with line breaks?
    features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "low_high_ratio", "hi_true", "lo_true", "low_high_ratio_true"]
    target = ["fault"]

    x_train, x_test, y_train, y_test = split_data(df[features], df[target], random_state_value)  # Split Data

    #classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=2, oob_score=False, random_state=0, verbose=0, warm_start=False)
    classifier = RandomForestClassifier(n_estimators=10, random_state=random_state_value)

    classifier.fit(x_train, y_train)

    y_predicted = classifier.predict(x_test)

    mcc.append(matthews_corrcoef(y_test, y_predicted))

    return classifier, y_test, y_predicted, mcc




data_file = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/"+"train_featuresHiLo_thresh_4.5_db4.csv"

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
random_state_values = list(range(0,4000, 40))
mcc_array = np.zeros((len(random_state_values), len(np.linspace(0.01, 0.99, 99))))
m_idx = 0
for random_state_value in random_state_values:
    classifier, y_test, y_pred_probs = run_light_gbm(data_file, random_state_value)
    mcc = survey_thresholds(y_test, y_pred_probs)
    mcc_array[m_idx,:] = mcc
    print("MCC for Fault Detection Threshold = 0.91: "+str(mcc[92]))
    m_idx+=1
plt.plot(np.linspace(0.01, 0.99, 99), np.median(mcc_array, axis=0), color="#2a2a2a", label="Median")
plt.plot(np.linspace(0.01, 0.99, 99), np.percentile(mcc_array, 25, axis=0), '--', color="#2a2a2a", label="25th Percentile")
plt.legend()
plt.grid()
plt.xlabel("Fault Classification Threshold")
plt.ylabel("Matthews Correlation Coefficient")
plt.title("Peak Detection Threshold : 4.5 , "+str(len(random_state_values))+" Runs with 80% Training Data")
plt.savefig("plots/mcc_vs_faultThresh_RF_cv_learn0.025_split80.png", bbox_inches='tight')