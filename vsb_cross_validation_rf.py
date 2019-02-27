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
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report, recall_score, precision_recall_curve


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
                
                X_train = X[train_idx]
                Y_train = y[train_idx]
                X_valid = X[valid_idx]
                Y_valid = y[valid_idx]
                
                clf.fit(X_train, Y_train)
                
                valid_pred = clf.predict(X_valid)
                recall  = recall_score(Y_valid, valid_pred, average='macro')
                f1 = f1_score(Y_valid, valid_pred, average='macro')
                mcc = matthews_corrcoef(Y_valid, valid_pred)
                
                recall_scores[i][j] = recall
                f1_scores[i][j] = f1
                
                train_pred[valid_idx, i] = valid_pred
                test_pred[:, test_col] = clf.predict(T)
                test_col += 1
                
                ## Probabilities
                valid_proba = clf.predict_proba(X_valid)
                train_proba[valid_idx, :] = valid_proba
                test_proba  += clf.predict_proba(T)
                
                print( "Model- {} and CV- {} recall: {}, f1_score: {}, mcc: {}".format(i, j, recall, f1, mcc))
                
            test_proba /= self.n_splits
            
        return train_proba, test_proba, train_pred, test_pred

def re_predict(data, threshods):

    argmax = np.argmax(data)

    ## If the argmax is 2 (class-3) then ovbiously return this highest label
    if argmax == 2: 
        return (argmax +1)

    # If argmax is 1 (class-2) there is a chnace that, label is class-2 if
    # the probability of the class is greater than the threshold otherwise obviously
    # return this highest label (class-3)
    elif argmax == 1:
        if data[argmax] >= threshods[argmax] : 
            return (argmax +1)
        else:
            return (argmax +2)

    # If the argmax is 0 (class-1) then there are chances that label is class-1 if
    # the probability of the class is greater than the threshold otherwise label can be
    # either next two highest labels (class-2 or class-3). To determine the exact class
    # class, we have to consider four cases.
    # case A : if class_2_prob >= threshold and class_3_prob < threshold then pick class-2
    # case B : if class_3_prob >= threshold and class_2_prob < threshold then pick class-3
    # case C : if class_2_prob < threshold and class_3_prob < threshold then pick class-1
    # case D : if class_2_prob > threshold and class_3_prob > threshold then pick class-3

    elif argmax == 0:

        if data[argmax] >= threshods[argmax] : 
            return (argmax +1)
        else:
            # case A : if class_2_prob >= threshold and class_3_prob < threshold then pick class-2
            if data[argmax + 1] >= threshods[argmax + 1] and data[argmax + 2] < threshods[argmax + 2]:
                return (argmax + 2)

            # case B : if class_3_prob >= threshold and class_2_prob < threshold then pick class-3
            if data[argmax + 2] >= threshods[argmax + 2] and data[argmax + 1] < threshods[argmax + 1]:
                return (argmax + 3)

            # case C : if class_2_prob < threshold and class_3_prob < threshold then pick class-1
            if data[argmax + 1] < threshods[argmax + 1] and data[argmax + 2] < threshods[argmax + 2]:
                return (argmax + 1)

            # case D : if class_2_prob > threshold and class_3_prob > threshold then pick class-3
            if data[argmax + 1] > threshods[argmax + 1] and data[argmax + 2] > threshods[argmax + 2]:
                return (argmax + 3)


# Load Data
features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "low_high_ratio", "hi_true", "lo_true", "low_high_ratio_true"]
target = ["fault"]

data_file = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/"+"train_featuresHiLo_thresh_4.5_db4.csv"
df_train = pd.read_csv(data_file)
train = df_train[features + target]

data_file = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/"+"test_featuresHiLo_thresh_4.5_db4.csv"
df_test = pd.read_csv(data_file)
test = df_test[features]



random_states = [1]
for random_state in random_states:
    class_weight = dict({0:1.9, 1:180})

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

    xtrain = train.drop(['fault'], axis=1)
    ytrain = train['fault'].values

    train_proba, test_proba, train_pred, test_pred = lgb_stack.predict(xtrain, ytrain, test)

    print('1. The F-1 score of the model {}\n'.format(f1_score(ytrain, train_pred, average='macro')))
    print('2. The recall score of the model {}\n'.format(recall_score(ytrain, train_pred, average='macro')))
    print('3. The Matthews Correlation Coefficient: {}\n'.format(matthews_corrcoef(ytrain, train_pred)))
    print('4. Classification report \n {} \n'.format(classification_report(ytrain, train_pred)))
    print('5. Confusion matrix \n {} \n'.format(confusion_matrix(ytrain, train_pred)))
    
    # histogram of predicted probabilities
    plt.figure(figsize=(12, 4))
    nclasses = 2
    for i in range(nclasses):
        
        plt.subplot(1, nclasses, i+1)
        plt.hist(train_proba[:, i], bins=20, histtype='bar', rwidth=0.95)
        plt.xlim(0,1)
        plt.title('Predicted class-{} probabilities'.format(i+1))
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    y = label_binarize(ytrain, classes=[0, 1])
    _, _, th1 = roc_curve(y[:, 0], train_proba[:, 0])
    _, _, th2 = roc_curve(y[:, 0], train_proba[:, 1])

    print(np.median(th1))
    print(np.median(th2))


    threshold = [0.47, 0.15]
    new_pred = []
    for i in range(train_pred.shape[0]):
        new_pred.append(re_predict(train_proba[i, :], threshold))
    
    print('1. The F-1 score of the model {}\n'.format(f1_score(ytrain, new_pred, average='macro')))
    print('2. The recall score of the model {}\n'.format(recall_score(ytrain, new_pred, average='macro')))
    print('3. The Matthews Correlation Coefficient: {}\n'.format(matthews_corrcoef(ytrain, new_pred)))
    print('4. Classification report \n {} \n'.format(classification_report(ytrain, new_pred)))
    print('5. Confusion matrix \n {} \n'.format(confusion_matrix(ytrain, new_pred)))