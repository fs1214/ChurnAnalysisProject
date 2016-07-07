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
import pickle
import matplotlib.pyplot as plt
import csv

"""
Given a threshold, if churn rate is more than that, we regard this customer as churn(y = 1); otherwise not churn(y = 0).
X is constructed by removing some features from the original dataset.
"""
def construct_X_y(df, threshold):
    y = copy.deepcopy(df.ChurnRate)
    y[y > threshold] = 1
    y[y <= threshold] = 0
    X = df.drop(['ChurnRate','ClientName','MaxLineCount','Industry','RecentContractType','FirstContractType','NumOfFCTickets','NumOfHDTickets',
                 'NumOfSATickets','FCLinesAffected','HDLinesAffected','SALinesAffected','FCAvgResolvingDays',
                 'HDAvgResolvingDays','SAAvgResolvingDays', 'StartDate','EndDate','IndOfAlrChurned'], 1)
    return X, y


"""
Choose the pair with the largest average value from dictionary
"""
def obtain_best_pair(accu_dict):
    max_accu  = 0
    for key, value in accu_dict.iteritems():
        if max_accu < np.mean(value):
            max_accu = np.mean(value)
            max_para = key
    return max_para, max_accu


"""
Train logistic regression on training set and find the best parameter by tuning parameter c and class_weight.
parameters:
    X_train: predictors of training set 
    y_train: target variable of trainging set
    k: k fold cross validation
    cs: a list of parameter c, which controls the regularization strength of logistic regression.
    class_weight: assign different weight to each class
    evaluation_methods: precision, recall, fscore
return:
    best_params: parameters which achieve highest performance on the training set
    mean_accu: the highest precision/recall/fscore score on the training set
"""
def train_LR_models(X_train, y_train, k, cs, class_weight, evaluation_methods):
    precision_accus, recall_accus, fscore_accus = Mod.LRValAUC(X_train[:,1:], y_train, k, cs, class_weight)
    if evaluation_methods == 'precision':
        best_params, mean_accu = obtain_best_pair(precision_accus)
    elif evaluation_methods == 'recall':
        best_params, mean_accu = obtain_best_pair(recall_accus)
    else:
        best_params, mean_accu = obtain_best_pair(fscore_accus)
    return best_params, mean_accu

"""
Use the best parameters from cross validation and evaluate the model performance on test set.
"""
def evaluation_LR_models(X_train, X_test,y_train,y_test, best_params, class_weight):
    
    #Fill the best parameters into the model
    lr = linear_model.LogisticRegression(C=best_params[0], penalty='l{}'.format(best_params[1]),class_weight = class_weight)
    lr.fit(X_train[:,1:], y_train)
    
    #Save the best trained model into 'best_lr.pkl', which could be deployed for new data set.
    with open('best_lr.pkl', 'wb') as f:
        pickle.dump(lr, f)
    print '==> Save model'
    
    #Get predictions from test data and also show the overall classification rate(accu), precision/recall/fscore only on churn customers. 
    y_pred = lr.predict(X_test[:,1:])
    accu = (y_test == y_pred).mean()
    precision, recall, fscore, threshold = precision_recall_fscore_support(y_test,y_pred)
    return y_pred, accu, precision[1],recall[1],fscore[1]

"""
Train random forest on training set and find the best parameter by tuning n_estimators and max_depth, class_weight.
parameters:
    X_train: predictors of training set 
    y_train: target variable of trainging set
    k: k fold cross validation
    n_estimators: the number of trees in the forest
    max_depth_list: a list of max_depth(the maximum depth of tree) values
    class_weight: assign different weight to each class
    evaluation_methods: precision, recall, fscore
return:
    best_params: parameters which achieve highest performance on the training set
    mean_accu: the highest precision/recall/fscore score on the training set
"""
def train_RF_models(X_train, y_train, k, n_estimators, max_depth_list, class_weight, evaluation_methods):
    precision_accus, recall_accus, fscore_accus = Mod.RFValAUC(X_train[:,1:], y_train, k, n_estimators, max_depth_list, class_weight)
    if evaluation_methods == 'precision':
        best_params, mean_accu = obtain_best_pair(precision_accus)
    elif evaluation_methods == 'recall':
        best_params, mean_accu = obtain_best_pair(recall_accus)
    else:
        best_params, mean_accu = obtain_best_pair(fscore_accus)
    return best_params, mean_accu


"""
Use the best parameters from cross validation and evaluate the model performance on test set.
"""
def evaluation_RF_models(X_train, X_test,y_train,y_test, best_params, class_weight, feature_names):
    rf = RandomForestClassifier(n_estimators=best_params[0] ,max_depth = best_params[1], class_weight=class_weight)
    rf.fit(X_train[:,1:], y_train)
    
    #Save the best trained model into 'best_lr.pkl', which could be deployed for new data set.
    with open('best_rf.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print '==> Save model'
    
    y_pred = rf.predict(X_test[:,1:])
    accu = (y_test == y_pred).mean()
    precision, recall, fscore, threshold = precision_recall_fscore_support(y_test,y_pred)
    
    #Plot top 20 important features
    plot_RF_feature_importance(feature_names, rf.feature_importances_, 20)
    return y_pred, accu, precision[1],recall[1],fscore[1]

"""
Plot the graph showing top n important features according to random forest attributes
"""
def plot_RF_feature_importance(feature_names, feature_importances, n):
    fi_dict = dict(zip(feature_names[1:],feature_importances))
    sorted_fi = sorted(fi_dict.items(),key = lambda x: x[1],reverse=True)
    
    header = [sorted_fi[i][0] for i in range(len(sorted_fi))]
    values = [sorted_fi[i][1] for i in range(len(sorted_fi))]

    fig, ax = plt.subplots()
    width = 0.6
    ax.bar(np.arange(1,n+1),values[:n],width, align = 'center')
    ax.set_xticks(np.arange(1,n+1))
    ax.set_xticklabels(header[:n], rotation = 90,fontsize = 10)
    plt.xlabel('Features')
    plt.ylabel('Feature Importances')
    plt.title('Top {} Features'.format(n))
    
    plt.savefig("TopFeatures.png", bbox_inches='tight')
    
"""
Calculate potential monthly revenue based on the churn customers we correctly recognize through predictions 
"""
def possible_saving(y_pred, y_test, X_test):
    res_dict = {'ClientID':[], 'AvgMonthlyBilling':[]}
    for i in np.arange(len(y_test)):
        if y_pred[i] == 1:
            if y_test[i] == 1:
                res_dict['ClientID'].append(int(X_test[i,0]))
                res_dict['AvgMonthlyBilling'].append(X_test[i,2])
    return res_dict

if __name__ == "__main__":
    
    #Write the result into txt file
    res_file = open("result.txt", 'a')
    
    #In the terminal, enter 'python Train.py filename evaluation_methods'
    filename = sys.argv[1]
    evaluation_methods = sys.argv[2]
    
    #Read file
    df = DP.read_file(filename)
    
    #Clean data
    processed_df = DP.clean_data(df)
    #processed_df.to_csv('clean_data.csv')
    
    #Construct X and y, split into training and test set
    X, y = construct_X_y(processed_df, 0.5)
    X_train, X_test,y_train,y_test = train_test_split(X.values,y.values, test_size = 0.3,random_state = 42)
    
    #Record all the predictors' name
    feature_names = X.columns.values

    #Logistic Regression
    """
    k = 10
    cs = [10**i for i in np.arange(-4,5,1)]
    class_weight = {0:0.3, 1:0.7}
    best_params, mean_accu = train_LR_models(X_train, y_train, k, cs, class_weight, evaluation_methods)
    params_str = "Best paramters: C = {}, norm = l{}, class_weight = {}, measurement = {}".format(best_params[0], best_params[1], class_weight, evaluation_methods)
    y_pred_lr, accu, precision, recall, fscore = evaluation_LR_models(X_train, X_test,y_train,y_test, best_params, class_weight)
    lr_saving_res = possible_saving(y_pred_lr, y_test, X_test)
    res_str = "Result: accuracy = {}, precision = {}, recall = {}, fscore = {}, num of TP = {}, PossibleMonthlyBilling = {}"\
        .format(accu, precision, recall, fscore, len(lr_saving_res['ClientID']), sum(lr_saving_res['AvgMonthlyBilling']))

    print params_str
    print res_str
    
    f = open('ChurnClient.csv', "wb")
    writer = csv.writer(f)
    writer.writerow(['ClientID','AvgMonthlyBilling'])
    for i in range(len(lr_saving_res['ClientID'])):
        writer.writerow([lr_saving_res['ClientID'][i],lr_saving_res['AvgMonthlyBilling'][i]])
    f.close()
    
    res_file.write("\n{}\n{}\n".format(params_str, res_str))
    res_file.close()
    """
    
    #Random Forest
    
    n_estimators = [100,500,900]
    max_depth_list = [10,20,30]
    k = 10
    class_weight = {0:0.5,1:0.5}
    best_params, mean_accu = train_RF_models(X_train, y_train, k, n_estimators, max_depth_list, class_weight, evaluation_methods)
    params_str = "Best paramters: n_estimators = {}, max_depth = {}, class_weight = {}, measurement = {}".format(best_params[0], best_params[1], class_weight, evaluation_methods)
    y_pred_rf, accu, precision, recall, fscore = evaluation_RF_models(X_train, X_test,y_train,y_test, best_params, class_weight,feature_names)
    rf_saving_res = possible_saving(y_pred_rf, y_test, X_test)
    res_str = "Result: accuracy = {}, precision = {}, recall = {}, fscore = {}, num of TP = {}, PossibleMonthlyBilling = {}"\
        .format(accu, precision, recall, fscore, len(rf_saving_res['ClientID']), sum(rf_saving_res['AvgMonthlyBilling']))
    
    f = open('ChurnClient.csv', "wb")
    writer = csv.writer(f)
    writer.writerow(['ClientID','AvgMonthlyBilling'])
    for i in range(len(rf_saving_res['ClientID'])):
        writer.writerow([rf_saving_res['ClientID'][i],rf_saving_res['AvgMonthlyBilling'][i]])
    f.close()
    
    
    print params_str
    print res_str
    
    
    res_file.write("\n{}\n{}\n".format(params_str, res_str))
    res_file.close()
    