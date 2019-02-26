# https://towardsdatascience.com/working-with-highly-imbalanced-datasets-in-machine-learning-projects-c70c5f2a7b16
# https://github.com/msahamed/handle_imabalnce_class/blob/master/imbalance_datasets_machine_learning.ipynb

# https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

import numpy as np
import pandas as pd
import warnings

## Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

## Sklearn Libraries
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report, recall_score, precision_recall_curve


def split_data(features, labels, random_state_value=1):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=random_state_value)
    return x_train, x_test, y_train, y_test

class Create_ensemble(object):
    def __init__(self, n_splits, base_models):
        self.n_splits = n_splits
        self.base_models = base_models

    def predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        no_class = len(np.unique(y))

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                     random_state = random_state).split(X, y))

        train_proba = np.zeros((X.shape[0], no_class))
        test_proba = np.zeros((T.shape[0], no_class))
        
        train_pred = np.zeros((X.shape[0], len(self.base_models)))
        test_pred = np.zeros((T.shape[0], len(self.base_models)* self.n_splits))
        f1_scores = np.zeros((len(self.base_models), self.n_splits))
        recall_scores = np.zeros((len(self.base_models), self.n_splits))
        
        test_col = 0
        for i, clf in enumerate(self.base_models):
            
            for j, (train_idx, valid_idx) in enumerate(folds):

                print(j)
                
                X_train = X[train_idx]
                Y_train = y[train_idx]
                X_valid = X[valid_idx]
                Y_valid = y[valid_idx]
                
                clf.fit(X_train, Y_train)
                
                valid_pred = clf.predict(X_valid)
                recall  = recall_score(Y_valid, valid_pred, average='macro')
                f1 = f1_score(Y_valid, valid_pred, average='macro')
                
                recall_scores[i][j] = recall
                f1_scores[i][j] = f1
                
                train_pred[valid_idx, i] = valid_pred
                test_pred[:, test_col] = clf.predict(T)
                test_col += 1
                
                ## Probabilities
                valid_proba = clf.predict_proba(X_valid)
                train_proba[valid_idx, :] = valid_proba
                test_proba  += clf.predict_proba(T)
                
                print( "Model- {} and CV- {} recall: {}, f1_score: {}".format(i, j, recall, f1))
                
            test_proba /= self.n_splits
            
        return train_proba, test_proba, train_pred, test_pred





# Load Data
data_file = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/"+"train_featuresHiLo_thresh_4.5_db4.csv"
df = pd.read_csv(data_file)
features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "low_high_ratio", "hi_true", "lo_true", "low_high_ratio_true"]
target = ["fault"]



random_states = [1]
for random_state in random_states:
    x_train, x_test, y_train, y_test = split_data(df[features], df[target], random_state)


    #class_weight = dict({1:1.9, 2:35, 3:180})
    class_weight = dict({1:1.9, 2:180})

    rdf = RandomForestClassifier(bootstrap=True, class_weight=class_weight, criterion='gini',
                max_depth=8, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=4, min_samples_split=10,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
                oob_score=False,
                random_state=random_state,
                verbose=0, warm_start=False)

    base_models = [rdf]
    n_splits = 5
    lgb_stack = Create_ensemble(n_splits = n_splits, base_models = base_models)        

    #xtrain = train.drop(['label'], axis=1)
    #ytrain = train['label'].values
    # ytrain = label_binarize(Y, classes=[0, 1, 2])

    train_proba, test_proba, train_pred, test_pred = lgb_stack.predict(x_train, y_train, x_test)

    mcc = matthews_corrcoef(test_pred, y_test)
    print(mcc)

    #print('1. The F-1 score of the model {}\n'.format(f1_score(y_train, train_pred, average='macro')))
    #print('2. The recall score of the model {}\n'.format(recall_score(y_train, train_pred, average='macro')))
    #print('3. Classification report \n {} \n'.format(classification_report(y_train, train_pred)))
    #print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_train, train_pred)))