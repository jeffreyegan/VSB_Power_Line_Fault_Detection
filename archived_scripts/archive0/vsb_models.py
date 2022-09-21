

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import collections
from matplotlib import pyplot as plt
import seaborn as sns


def load_feature_data(file_name):
    df = pd.read_csv(file_name)
    return df

def split_data(features, labels, seed=1):
    from sklearn.model_selection import train_test_split
    # Using standard split of 80-20 training to testing data split ratio and fixing random_state=1 for repeatability
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=seed)
    return x_train, x_test, y_train, y_test

def score_classifier(truth, predictions):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    from sklearn.metrics import confusion_matrix
    m_accuracy = accuracy_score(truth, predictions)
    m_recall = recall_score(truth, predictions)
    m_precision = precision_score(truth, predictions)
    m_f1 = f1_score(truth, predictions)
    c_matrix = confusion_matrix(truth, predictions)
    #print(accuracy_score(truth, predictions))
    #print(recall_score(truth, predictions))
    #print(precision_score(truth, predictions))
    #print(f1_score(truth, predictions))
    return m_accuracy, m_recall, m_precision, m_f1, c_matrix

def matthews_corr_coef(c_matrix):  # Use 2x2 Confusion Matrix to Calculate Matthews Correlation Coefficient
    #TP = c_matrix[0][0]  # True Positives
    #TN = c_matrix[0][1]  # True Negavitves
    #FP = c_matrix[1][0]  # False Positives
    #FN = c_matrix[1][1]  # False Negatives
    #print(c_matrix)

    TP = c_matrix[1][1]  # True Positives
    TN = c_matrix[0][0]  # True Negavitves
    FP = c_matrix[1][0]  # False Positives
    FN = c_matrix[0][1]  # False Negatives

    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    #print(MCC)
    return MCC


def classification_k_nearest(features, labels, k):
    from sklearn.neighbors import KNeighborsClassifier
    x_train, x_test, y_train, y_test = split_data(features, labels)
    time_in = time.time()
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train.values.ravel())
    y_predicted = classifier.predict(x_test)
    m_accuracy, m_recall, m_precision, m_f1, c_matrix = score_classifier(y_test, y_predicted.ravel())
    m_mcc = matthews_corr_coef(c_matrix)
    #print('Time elapsed: '+str(time.time() - time_in)+'')
    return m_accuracy, m_recall, m_precision, m_f1, m_mcc


def classification_support_vector_machine(features, labels, kernel_value, gamma_value):
    from sklearn.svm import SVC
    x_train, x_test, y_train, y_test = split_data(features, labels)
    time_in = time.time()
    classifier = SVC(kernel=kernel_value, gamma=gamma_value)
    classifier.fit(x_train, y_train.values.ravel())
    y_predicted = classifier.predict(x_test)
    m_accuracy, m_recall, m_precision, m_f1, c_matrix = score_classifier(y_test, y_predicted.ravel())
    #score = classifier.score(x_test, y_test)
    m_mcc = matthews_corr_coef(c_matrix)
    #print('Time elapsed: ' + str(time.time() - time_in) + '')
    return m_accuracy, m_recall, m_precision, m_f1, m_mcc


def classification_random_forest(features, labels, n_value, random_seed):
    from sklearn.ensemble import RandomForestClassifier
    x_train, x_test, y_train, y_test = split_data(features, labels, random_seed)
    classifier = RandomForestClassifier(n_estimators=n_value)  # Create Gaussian Classifier
    classifier.fit(x_train, y_train.values.ravel())
    y_predicted = classifier.predict(x_test)
    m_accuracy, m_recall, m_precision, m_f1, c_matrix = score_classifier(y_test, y_predicted.ravel())
    m_mcc = matthews_corr_coef(c_matrix)
    return m_accuracy, m_recall, m_precision, m_f1, m_mcc


def vsb_models(filename):
    df = load_feature_data(filename)
    features = df[["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks"]]
    labels = df[["fault"]]

    accuracy = []
    recall = []
    precision = []
    f1 = []
    mcc = []
    k_values = list(range(3,27,2))
    for k in k_values:
        m_accuracy, m_recall, m_precision, m_f1, m_mcc = classification_k_nearest(features, labels, k)
        accuracy.append(m_accuracy)
        recall.append(m_recall)
        precision.append(m_precision)
        f1.append(m_f1)
        mcc.append(m_mcc)

    plot_labels = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
    blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "#0E1E2B"]

    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(k_values, accuracy, '-', label=plot_labels[0], color=blues[0])
    plt.plot(k_values, recall, '--', label=plot_labels[1], color=blues[1])
    plt.plot(k_values, precision, '-.', label=plot_labels[2], color=blues[2])
    plt.plot(k_values, f1, ':', label=plot_labels[3], color=blues[3])
    plt.legend(loc='lower right')
    plt.title('K Nearest Neighbors Classification - Performance vs k Value')
    plt.xlabel('k Value')
    plt.ylabel('Classifier Scores')
    plt.savefig("performance_knn.png", bbox_inches='tight')

    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(k_values, mcc, '-', label="Matthews Correlation Coefficient", color=blues[1])
    plt.legend(loc='lower right')
    plt.title('K Nearest Neighbors Classification - Matthews Correlation Coefficient Performance vs k Value')
    plt.xlabel('k Value')
    plt.ylabel('Matthews Correlation Coefficient')
    plt.savefig("mcc_knn.png", bbox_inches='tight')


    accuracy = []
    recall = []
    precision = []
    f1 = []
    mcc = []
    gamma_values = np.linspace(0.05, 100, 500)
    kernel_value = "rbf"
    for gamma in gamma_values:
        print(gamma)
        m_accuracy, m_recall, m_precision, m_f1, m_mcc = classification_support_vector_machine(features, labels, kernel_value, gamma)
        accuracy.append(m_accuracy)
        recall.append(m_recall)
        precision.append(m_precision)
        f1.append(m_f1)
        mcc.append(m_mcc)
    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(gamma_values, accuracy, '-', label=plot_labels[0], color=blues[0])
    plt.plot(gamma_values, recall, '--', label=plot_labels[1], color=blues[1])
    plt.plot(gamma_values, precision, '-.', label=plot_labels[2], color=blues[2])
    plt.plot(gamma_values, f1, ':', label=plot_labels[3], color=blues[3])
    plt.legend(loc='lower right')
    plt.title('Support Vector Machine Classification - RBF Kernel Performance vs Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Classifier Scores')
    plt.savefig("performance_svm.png", bbox_inches='tight')

    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(gamma_values, mcc, '-', label="Matthews Correlation Coefficient", color=blues[1])
    plt.legend(loc='lower right')
    plt.title('K Nearest Neighbors Classification - Matthews Correlation Coefficient Performance vs Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Matthews Correlation Coefficient')
    plt.savefig("mcc_svm.png", bbox_inches='tight')


    accuracy = []
    recall = []
    precision = []
    f1 = []
    mcc = []
    n_values = list(range(5,250,5))
    seed = 2019
    for n_value in n_values:
        print(n_value)
        m_accuracy, m_recall, m_precision, m_f1, m_mcc = classification_random_forest(features, labels, n_value, seed)
        accuracy.append(m_accuracy)
        recall.append(m_recall)
        precision.append(m_precision)
        f1.append(m_f1)
        mcc.append(m_mcc)
    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(n_values, accuracy, '-', label=plot_labels[0], color=blues[0])
    plt.plot(n_values, recall, '--', label=plot_labels[1], color=blues[1])
    plt.plot(n_values, precision, '-.', label=plot_labels[2], color=blues[2])
    plt.plot(n_values, f1, ':', label=plot_labels[3], color=blues[3])
    plt.legend(loc='lower right')
    plt.title('Random Forest Classification - Performance vs Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Classifier Scores')
    plt.savefig("performance_rf.png", bbox_inches='tight')

    fig=plt.figure(figsize=(14, 8), dpi= 120, facecolor='w', edgecolor='k')
    plt.plot(n_values, mcc, '-', label="Matthews Correlation Coefficient", color=blues[1])
    plt.legend(loc='lower right')
    plt.title('Random Forest Classification - Matthews Correlation Coefficient Performance vs Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Matthews Correlation Coefficient')
    plt.savefig("mcc_rf.png", bbox_inches='tight')


filename = "/home/jeffrey/repos/VSB_Power_Line_Fault_Detection/extracted_features/train_features_db4.csv"
vsb_models(filename)