import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef

from sklearn.model_selection import GridSearchCV


def load_feature_data(file_name):
    df = pd.read_csv(file_name)
    return df


def split_data(features, labels, random_state_value=1):
    from sklearn.model_selection import train_test_split
    # Using standard split of 80-20 training to testing data split ratio and fixing random_state=1 for repeatability
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=random_state_value)
    return x_train, x_test, y_train, y_test


# Metric Used in this Competition
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Matthews Correlation Coefficient (MCC)
def mcc(truth, predictions):
    return 'MCC', matthews_corrcoef(truth, predictions), True

def run_light_gbm(data_file):
    print('Loading data...')
    df = load_feature_data(data_file)

    features = df[["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]]
    labels = df[["fault"]]

    x_train, x_test, y_train, y_test = split_data(features, labels, 2019)  # Split Data

    data_training = lgb.Dataset(x_train, label=y_train)  # Data set used to train model
    training_iterations = 10 # Number of iterations used to train model
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        #'metric': mcc,
        #'early_stopping_rounds': 5,
        'num_leaves': 31,
        'min_data': 50,
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
    }

    print('Starting training...')
    clf = lgb.train(params, data_training, training_iterations)  # Train model
    
    print('Starting predicting...')
    #y_predicted = clf.predict(x_test, num_iteration=clf.best_iteration_)
    y_pred_probs = clf.predict(x_test) # output is a list of probabilities
    y_predicted = []

    for y in y_pred_probs:  # use probabilities to assign binary classification
        if y >= 0.5:
            y_predicted.append(1)
        else:
            y_predicted.append(0)



    # Evaulation / Performance Scoring
    m_mcc = mcc(y_test, y_predicted)
    print("Matthews Correlation Coefficient score for predictions: "+str(m_mcc))

    # Important Features
    #print('Feature importances:', list(clf.feature_importances_))


data_file = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/"+"train_features_thresh_0.69_db4.csv"
run_light_gbm(data_file)