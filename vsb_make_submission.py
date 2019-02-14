
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from matplotlib import pyplot as plt


def split_data(features, labels, random_state_value=1):
    from sklearn.model_selection import train_test_split
    # Using standard split of 80-20 training to testing data split ratio and fixing random_state=1 for repeatability
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.95, test_size=0.05, random_state=random_state_value)
    return x_train, x_test, y_train, y_test


def classification_light_gbm_model(df_train):
    print('Loading data...')

    features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]
    target = ["fault"]
    x_train, x_test, y_train, y_test = split_data(df_train[features], df_train[target], 2019)  # Split Data

    print("preparing validation datasets")
    xgdata = lgb.Dataset(df_train[features], df_train[target])
    xgtrain = lgb.Dataset(x_train, y_train)
    xgtest = lgb.Dataset(x_test, y_test)

    evals_results = {}

    dtrain = lgb.Dataset(x_train, label=y_train)  # Data set used to train model
    training_iterations = 10 # Number of iterations used to train model
    metrics = 'auc'

    lgb_params = {
        'objective': 'binary',
        'metric': metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'scale_pos_weight':99, # because training data is extremely unbalanced 
        'boosting_type': 'gbdt',
        'boost_from_average': False
    }

    print("Training...")
    classifier = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgtest], 
                     valid_names=['train','test'], 
                     evals_result=evals_results, 
                     num_boost_round=1000,
                     early_stopping_rounds=50,
                     verbose_eval=True, 
                     feval=None)

    
    n_estimators = classifier.best_iteration
    y_pred_probs = classifier.predict(x_test, n_estimators)

    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.hist(y_pred_probs, bins=100, color=blues[1])
    plt.ylabel("Occurrences")
    plt.xlabel("Probability of Assigning Fault")
    plt.savefig("plots/fault_probs_lgbm_submit.png", bbox_inches='tight')

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['test'][metrics][n_estimators-1])
    return classifier


def predict_light_gbm_model(classifier, df_test, threshold):
    print('Predicting...')
    test_features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]
    n_estimators = classifier.best_iteration
    y_pred_probs = classifier.predict(df_test[test_features], n_estimators)

    y_predicted = []
    for y in y_pred_probs:  # use probabilities to assign binary classification
        if y >= threshold:
            y_predicted.append(1)
        else:
            y_predicted.append(0) 
    return y_predicted


# Train Model with Full Set, Return Classifier Model
dwt = "db4"
peak_thresh = "5.0"

training_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/train_features_thresh_"+peak_thresh+"_"+dwt+".csv"
df_train = pd.read_csv(training_data)
classifier = classification_light_gbm_model(df_train)  # Light GBM


# Make Predictions
test_data = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/test_features_thresh_"+peak_thresh+"_"+dwt+".csv"
df_test = pd.read_csv(test_data).drop(['Unnamed: 0'],axis=1)

fault_detection_threshold = 0.91
predicted_faults = predict_light_gbm_model(classifier, df_test, fault_detection_threshold)
df_test["fault"] = predicted_faults


# Make Submission File
submission_filename = "submissions/prediction_submission_"+peak_thresh+"_"+dwt+"_"+str(fault_detection_threshold)+"fdt_.csv"

f_o = open(submission_filename, "w+")
f_o.write("signal_id,target\n")
for idx in range(len(df_test)):
    signal_id = df_test["signal_id"][idx]
    fault = df_test["fault"][idx]
    f_o.write(str(signal_id)+","+str(fault)+"\n")

f_o.close()