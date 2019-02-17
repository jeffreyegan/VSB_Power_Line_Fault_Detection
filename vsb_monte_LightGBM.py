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
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.95, test_size=0.05, random_state=random_state_value)
    return x_train, x_test, y_train, y_test


# Metric Used in this Competition
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Matthews Correlation Coefficient (MCC)
def mcc(truth, predictions):
    return 'MCC', matthews_corrcoef(truth, predictions), True

def run_light_gbm(data_file, random_state_value):
    print('Loading data...')
    df = load_feature_data(data_file)

    #features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]
    #features = ["min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]
    features = ["signal_id", "entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "hi_count", "lo_count", "low_high_ratio", "hi_true", "lo_true", "low_high_ratio_true"]
    target = ["fault"]

    x_train, x_test, y_train, y_test = split_data(df[features], df[target], random_state_value)  # Split Data

    print("preparing validation datasets")
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
                     verbose_eval=False, 
                     feval=None)

    
    n_estimators = classifier.best_iteration
    y_pred_probs = classifier.predict(x_test, n_estimators)

    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    #fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    #plt.hist(y_pred_probs, bins=100, color=blues[1])
    #plt.ylabel("Occurrences")
    #plt.xlabel("Probability of Assigning Fault")
    #plt.savefig("plots/fault_probs_lgbm.png", bbox_inches='tight')

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['test'][metrics][n_estimators-1])

    return classifier, y_test, y_pred_probs



def survey_thresholds(y_test, y_pred_probs):

    thresholds = np.linspace(0.01, 0.99, 99)
    mcc = []
    for threshold in thresholds:
        y_predicted = []
        for y in y_pred_probs:  # use probabilities to assign binary classification
            if y >= threshold:
                y_predicted.append(1)
            else:
                y_predicted.append(0)
        mcc.append(matthews_corrcoef(y_test, y_predicted))
        #print("MCC: "+str(matthews_corrcoef(y_test, y_predicted)))

    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    #fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(thresholds, mcc, color=blues[1], alpha=0.2)
    #plt.ylabel("Fault Classification Threshold")
    #plt.xlabel("Matthews Correlation Coefficient Score")
    #plt.show()
    #plt.savefig("plots/mcc_vs_faultThresh_lgbm.png", bbox_inches='tight')
    return mcc


data_file = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/"+"train_features_thresh_5.0_db4.csv"

fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
random_state_values = list(range(0,4000, 40))
for random_state_value in random_state_values:
    classifier, y_test, y_pred_probs = run_light_gbm(data_file, random_state_value)
    mcc = survey_thresholds(y_test, y_pred_probs)
    print("MCC for Fault Detection Threshold = 0.91: "+str(mcc[92]))
plt.xlabel("Fault Classification Threshold")
plt.ylabel("Matthews Correlation Coefficient")
plt.title("Peak Detection Threshold : 5.0 , "+str(len(random_state_values))+" Runs with 95% Training Data")
plt.savefig("plots/mcc_vs_faultThresh_lgbm_7feats_Peaks5.0_split95.png", bbox_inches='tight')