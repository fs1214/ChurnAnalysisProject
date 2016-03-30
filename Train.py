__author__ = 'fangyunsun'

import DataPreprocess as DP
import Models as Mod
import copy
import sys
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def construct_X_y(df, threshold):
    y = copy.deepcopy(df.ChurnRate)
    y[y > threshold] = 1
    y[y <= threshold] = 0
    X = df.drop(['ChurnRate','ClientName','Industry','RecentContractType','FirstContractType','StartDate','EndDate'], 1)
    return X, y

def obtain_best_pair(accu_dict):
    max_accu  = 0
    for key, value in accu_dict.iteritems():
        if max_accu < np.mean(value):
            max_accu = np.mean(value)
            max_para = key
    return max_para, max_accu

def train_LR_models(X_train, y_train, k, cs, class_weight, evaluation_methods):
    precision_accus, recall_accus, fscore_accus = Mod.LRValAUC(X_train[:,1:], y_train, k, cs, class_weight)
    if evaluation_methods == 'precision':
        best_params, mean_accu = obtain_best_pair(precision_accus)
    elif evaluation_methods == 'recall':
        best_params, mean_accu = obtain_best_pair(recall_accus)
    else:
        best_params, mean_accu = obtain_best_pair(fscore_accus)
    return best_params, mean_accu

def evaluation_LR_models(X_train, X_test,y_train,y_test, best_params, class_weight):
    lr = linear_model.LogisticRegression(C=best_params[0], penalty='l{}'.format(best_params[1]),class_weight = class_weight)
    lr.fit(X_train[:,1:], y_train)
    y_pred = lr.predict(X_test[:,1:])
    accu = (y_test == y_pred).mean()
    precision, recall, fscore, threshold = precision_recall_fscore_support(y_test,y_pred)
    return y_pred, accu, precision[1],recall[1],fscore[1]

def possible_saving(y_pred, y_test, X_test):
    res_dict = {'ClientID':[], 'AvgMonthlyBilling':[]}
    for i in np.arange(len(y_test)):
        if y_pred[i] == 1:
            if y_test[i] == 1:
                res_dict['ClientID'].append(int(X_test[i,0]))
                res_dict['AvgMonthlyBilling'].append(X_test[i,2])
    return res_dict

if __name__ == "__main__":
    filename = sys.argv[1]
    evaluation_methods = sys.argv[2]
    df = DP.read_file(filename)
    processed_df = DP.clean_data(df)
    #processed_df.to_csv('clean_data.csv')
    X, y = construct_X_y(processed_df, 0.5)
    X_train, X_test,y_train,y_test = train_test_split(X.values,y.values, test_size = 0.3,random_state = 42)

    #Logistic Regression
    k = 10
    cs = [10**i for i in np.arange(-5,5,0.5)]
    class_weight = {0:0.25,1:0.75}
    best_params, mean_accu = train_LR_models(X_train, y_train, k, cs, class_weight, evaluation_methods)
    y_pred_lr, accu, precision, recall, fscore = evaluation_LR_models(X_train, X_test,y_train,y_test, best_params, class_weight)
    lr_saving_res = possible_saving(y_pred_lr, y_test, X_test)
    print accu, precision, recall, fscore, lr_saving_res

