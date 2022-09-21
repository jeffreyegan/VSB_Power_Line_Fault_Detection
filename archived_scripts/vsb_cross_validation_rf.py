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
            
        return train_proba, test_proba, train_pred, test_pred, clf

def re_predict(data, threshods):
    argmax = np.argmax(data)
    if argmax == 1:
        if data[argmax] >= threshods[argmax] : 
            return 1
        else:
            return 0
    else:  # argmax == 0 
        if data[argmax] >= threshods[argmax] : 
            return 0
        else:
            return 1


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
    class_weight = dict({0:0.5, 1:2.0})

    rdf = RandomForestClassifier(bootstrap=True, class_weight=class_weight, criterion='gini',
            max_depth=8, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
            oob_score=False,
            random_state=random_state, verbose=0, warm_start=False)

    base_models = [rdf]
    n_splits = 5
    lgb_stack = Create_ensemble(n_splits = n_splits, base_models = base_models)        

    xtrain = train.drop(['fault'], axis=1)
    ytrain = train['fault'].values

    train_proba, test_proba, train_pred, test_pred, clf = lgb_stack.predict(xtrain, ytrain, test)

    print(train_pred[0])
    print(test_pred[0])
    print('\nPerformance Metrics after Weighted Random Forest Cross Validation')
    print('1. The F-1 score of the model {}\n'.format(f1_score(ytrain, train_pred, average='macro')))
    print('2. The recall score of the model {}\n'.format(recall_score(ytrain, train_pred, average='macro')))
    print('3. The Matthews Correlation Coefficient: {}\n'.format(matthews_corrcoef(ytrain, train_pred)))
    print('4. Classification report \n {} \n'.format(classification_report(ytrain, train_pred)))
    print('5. Confusion matrix \n {} \n'.format(confusion_matrix(ytrain, train_pred)))
    

    # histogram of predicted probabilities
    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]
    plt.figure(figsize=(12, 4))
    nclasses = 2
    titles = ["Probabilities for No Partial Discharge Fault Present", "Probabilities for Partial Discharge Fault Present"]
    for i in range(nclasses):
        plt.subplot(1, nclasses, i+1)
        plt.hist(train_proba[:, i], bins=50, histtype='bar', rwidth=0.95, color=blues[1])
        plt.xlim(0,1)
        plt.title(titles[i])
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # histogram of important features
    imp = clf.feature_importances_
    imp, features = zip(*sorted(zip(imp, features)))
    plt.figure(figsize=(12, 4))
    plt.barh(range(len(features)), imp, color=blues[1], align="center")
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance of Features")
    plt.ylabel("Features")
    plt.title("Importance of Each Feature in Classifier Model")
    plt.show()


    y = label_binarize(ytrain, classes=[0, 1])
    _, _, th1 = roc_curve(y[:, 0], train_proba[:, 0])
    _, _, th2 = roc_curve(y[:, 0], train_proba[:, 1])
    print('\nMedian Detection Thresholds for Fault Detection')  # use for setting reprediction thresholds
    print(np.median(th1))
    print(np.median(th2))


    threshold = [0.5, 0.1]
    new_pred = []
    for i in range(train_pred.shape[0]):
        new_pred.append(re_predict(train_proba[i, :], threshold))
    print('\nPerformance Metrics after Over Prediction')
    print('1. The F-1 score of the model {}\n'.format(f1_score(ytrain, new_pred, average='macro')))
    print('2. The recall score of the model {}\n'.format(recall_score(ytrain, new_pred, average='macro')))
    print('3. The Matthews Correlation Coefficient: {}\n'.format(matthews_corrcoef(ytrain, new_pred)))
    print('4. Classification report \n {} \n'.format(classification_report(ytrain, new_pred)))
    print('5. Confusion matrix \n {} \n'.format(confusion_matrix(ytrain, new_pred)))



test_pred = np.median(test_pred, axis=1).astype(int)
df_test["fault"] = test_pred

# Make Submission File
submission_filename = "submissions/prediction_submission_cv.csv"

f_o = open(submission_filename, "w+")
f_o.write("signal_id,target\n")
for idx in range(len(df_test)):
    signal_id = df_test["signal_id"][idx]
    fault = df_test["fault"][idx]
    f_o.write(str(signal_id)+","+str(fault)+"\n")
f_o.close()