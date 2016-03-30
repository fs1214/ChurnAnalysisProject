__author__ = 'fangyunsun'

from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def LRValAUC(X_tr, Y_tr, k, cs, class_weight):
    '''
    Perform k-fold cross validation on logistic regression, varies C and penalty Type (L1 or L2),
    returns a dictionary where key=c,value=[auc-c1, auc-c2, ...auc-ck].
    '''
    cv = KFold(n=X_tr.shape[0], n_folds = k)
    precision_aucs = {}
    recall_aucs = {}
    fscore_aucs = {}

    for train_index, test_index in cv:
        X_tr_f = X_tr[train_index]
        X_va_f = X_tr[test_index]
        Y_tr_f = Y_tr[train_index]
        Y_va_f = Y_tr[test_index]

        for c in cs:
            for norm in [1,2]:
                lr = linear_model.LogisticRegression(C=c, penalty='l{}'.format(norm),class_weight = class_weight)
                lr.fit(X_tr_f,Y_tr_f)
                precision, recall, fscore, threshold = precision_recall_fscore_support(Y_va_f,lr.predict(X_va_f))

                if (precision_aucs.has_key((c, norm))):
                    precision_aucs[(c, norm)].append(precision[1])
                    recall_aucs[(c, norm)].append(recall[1])
                    fscore_aucs[(c, norm)].append(fscore[1])

                else:
                    precision_aucs[(c, norm)] = [precision[1]]
                    recall_aucs[(c, norm)] = [recall[1]]
                    fscore_aucs[(c, norm)] = [fscore[1]]
    return precision_aucs, recall_aucs, fscore_aucs

def RFValAUC(X_tr, Y_tr, k, n_estimators, max_depth_list, class_weight):
    '''
    Perform k-fold cross validation on logistic regression, varies C and penalty Type (L1 or L2),
    returns a dictionary where key=c,value=[auc-c1, auc-c2, ...auc-ck].
    '''
    cv = KFold(n=X_tr.shape[0], n_folds = k)
    precision_aucs = {}
    recall_aucs = {}
    fscore_aucs = {}


    for train_index, test_index in cv:
        X_tr_f = X_tr[train_index]
        X_va_f = X_tr[test_index]
        Y_tr_f = Y_tr[train_index]
        Y_va_f = Y_tr[test_index]

        for n in n_estimators:
            for max_depth in max_depth_list:
                rf_clf = RandomForestClassifier(n_estimators=n,max_depth = max_depth, class_weight=class_weight)
                rf_clf.fit(X_tr_f,Y_tr_f)
                precision, recall, fscore, threshold = precision_recall_fscore_support(Y_va_f,rf_clf.predict(X_va_f))

                if (precision_aucs.has_key((n, max_depth))):
                    precision_aucs[(n, max_depth)].append(precision[1])
                    recall_aucs[(n, max_depth)].append(recall[1])
                    fscore_aucs[(n, max_depth)].append(fscore[1])

                else:
                    precision_aucs[(n, max_depth)] = [precision[1]]
                    recall_aucs[(n, max_depth)] = [recall[1]]
                    fscore_aucs[(n, max_depth)] = [fscore[1]]
    return precision_aucs, recall_aucs, fscore_aucs
