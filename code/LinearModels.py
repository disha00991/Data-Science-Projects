import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, hinge_loss, accuracy_score
import csv
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

lamda_array = [0.0001, 0.01, 1, 10, 100]
def question_9(X_trn, y_trn, X_tst):
    #get k folds (k=5)
    kf = KFold(n_splits=5)
    predictors = [] #stores 10 predictors
    lam_errors = [] ##[[logistic loss, hinge loss], [, ], ..]
    print("Lambda  |  Logistic Loss  |  Hinge Loss")
    for lam in lamda_array:

        fold_scores_log = []
        fold_scores_hinge = []
        clf_log = LogisticRegression(random_state=1, penalty='l2', C=1/(2*lam), solver='sag')
        clf_hinge = LinearSVC(random_state=1, penalty='l2', C=1/(2*lam))
        for train_index, test_index in kf.split(X_trn):
            Xk_train, Xk_test = X_trn[train_index], X_trn[test_index]
            yk_train, yk_test = y_trn[train_index], y_trn[test_index]

            clf_log.fit(Xk_train, yk_train)
            clf_hinge.fit(Xk_train, yk_train)
            fold_scores_log.append(clf_log.score(Xk_test, yk_test))
            fold_scores_hinge.append(clf_hinge.score(Xk_test, yk_test))
        
        predictors.append(clf_log)
        predictors.append(clf_hinge)

        logistic_loss = 1 - np.mean(np.array(fold_scores_log))
        hinge_loss = 1 - np.mean(np.array(fold_scores_hinge))
        print(f"{lam}  |  {logistic_loss}  |  {hinge_loss}")

    #finding losses
    find_classification_errors(predictors, X_trn, y_trn)
    find_logistic_losses(predictors, X_trn, y_trn)
    find_hinge_losses(predictors, X_trn, y_trn)

def find_classification_errors(predictors, X_trn, y_trn):
    print("Table 1:")
    print("Lambda  |            Classification Error      ")

    for lam in lam_errors:
        for i, predictor in enumerate(predictors):
            y_pred = get_predictions(predictor, X_trn)
            loss = 1 - accuracy_score(y_trn, y_pred)
            print(f"   {lam}","Log" if i%2==0 else "Hinge","     |        {loss}      ")
    print("----------------------------------------------------")

def find_logistic_losses(predictors, X_trn, y_trn):
    print("Table 2:")
    print("Lambda  |            Logistic Loss       ")
    for lam in lam_errors:
        for i, predictor in enumerate(predictors):
            y_pred = get_predictions(predictor, X_trn)
            loss = log_loss(y_trn, y_pred)
            print(f"   {lam}","Log" if i%2==0 else "Hinge","     |        {loss}      ")
    print("----------------------------------------------------")

def find_hinge_losses(predictors, X_trn, y_trn):
    print("Table 3:")
    print("Lambda  |            Hinge Loss       ")
    for lam in lam_errors:
        for i, predictor in enumerate(predictors):
            y_pred = get_predictions(predictor, X_trn)
            loss = hinge_loss(y_trn, y_pred)
            print(f"   {lam}","Log" if i%2==0 else "Hinge","     |        {loss}      ")
    print("----------------------------------------------------")

def question_10(X_trn, y_trn, X_tst):
    #get k folds (k=5)
    kf = KFold(n_splits=5)
    print("no of splits: ", kf.get_n_splits(X_trn))

    fold_scores = []
    classifier = lgoggClassifier(random_state=1, penalty='l2', C=1/(2*bessttt)) ##best depth = 6 returned by question 4
    for train_index, test_index in kf.split(X_trn):
        Xk_train, Xk_test = X_trn[train_index], X_trn[test_index]
        yk_train, yk_test = y_trn[train_index], y_trn[test_index]

        classifier.fit(Xk_train, yk_train)
        fold_scores.append(classifier.score(Xk_test, yk_test))        
    
    print("Estimated generalization error: ", 1 - np.mean(np.array(fold_scores)))
    classifier.fit(X_trn, y_trn)

    predictions = get_predictions(classifier, X_tst)
    write_csv(predictions, 'kfold.csv')

def get_predictions(predictor, xtest):
    return np.argmax(np.array(predictor.decision_function(xtest)), axis=1)

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])      


