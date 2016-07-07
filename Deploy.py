__author__ = 'fangyunsun'

import DataPreprocess as DP
import Models as Mod
import Train as Tr
import copy
import sys
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import csv

if __name__ == "__main__":
    
    #In the terminal, enter 'python Deploy.py filename modelname'
    filename = sys.argv[1]
    modelname = sys.argv[2]
    
    #Read new data set
    df = DP.read_file(filename)
    
    #Clean data
    processed_df = DP.clean_data(df)
    
    #Construct X and y, which stand for predictors and response variable 
    X, y = Tr.construct_X_y(processed_df, 0.5)
    
    #Load trained model from Train.py
    f = open(modelname,'rb')
    mod = pickle.load(f)
    f.close()
    
    #Get predictions and evaluation metrics for new records
    y_pred = mod.predict(X.values[:,1:])
    accu = (y.values == y_pred).mean()
    precision, recall, fscore, threshold = precision_recall_fscore_support(y.values,y_pred)
    potential_saving = Tr.possible_saving(y_pred, y.values, X.values)
    res_str = "Result: accuracy = {}, precision = {}, recall = {}, fscore = {}, num of TP = {}, PossibleMonthlyBilling = {}"\
        .format(accu, precision[1], recall[1], fscore[1], len(potential_saving['ClientID']), sum(potential_saving['AvgMonthlyBilling']))
    
    f = open('ChurnClient.csv', "wb")
    writer = csv.writer(f)
    writer.writerow(['ClientID','AvgMonthlyBilling'])
    for i in range(len(potential_saving['ClientID'])):
        writer.writerow([potential_saving['ClientID'][i],potential_saving['AvgMonthlyBilling'][i]])
    f.close()
    
    print res_str
