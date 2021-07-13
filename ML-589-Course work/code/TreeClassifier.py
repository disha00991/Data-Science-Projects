import numpy as np
import csv
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

def question_4(X_trn, y_trn, X_tst):
    #get k folds (k=5)
    kf = KFold(n_splits=5)
    print("no of splits: ",kf.get_n_splits(X_trn))

    max_depth = [1, 3, 6, 9, 12, 14]
    depth_errors = []
    print("Depth  |  Error")
    for depth in max_depth:

        fold_scores = []
        classifier = DecisionTreeClassifier(random_state=1, max_depth=depth)
        for train_index, test_index in kf.split(X_trn):
            Xk_train, Xk_test = X_trn[train_index], X_trn[test_index]
            yk_train, yk_test = y_trn[train_index], y_trn[test_index]

            classifier.fit(Xk_train, yk_train)
            fold_scores.append(classifier.score(Xk_test, yk_test))
        
        error = 1 - np.mean(np.array(fold_scores))
        print(f"{depth}  |  {error}")
        depth_errors.append(error)

    best_depth = max_depth[np.argmin(np.array(depth_errors))]
    print(f"best depth : {best_depth}") 

def question_5(X_trn, y_trn, X_tst):
    #get k folds (k=5)
    kf = KFold(n_splits=5)
    print("no of splits: ", kf.get_n_splits(X_trn))

    fold_scores = []
    classifier = DecisionTreeClassifier(random_state=1, max_depth=6) ##best depth = 6 returned by question 4
    for train_index, test_index in kf.split(X_trn):
        Xk_train, Xk_test = X_trn[train_index], X_trn[test_index]
        yk_train, yk_test = y_trn[train_index], y_trn[test_index]

        classifier.fit(Xk_train, yk_train)
        fold_scores.append(classifier.score(Xk_test, yk_test))        
    
    print("Estimated generalization error: ", 1 - np.mean(np.array(fold_scores)))
    classifier.fit(X_trn, y_trn)

    predictions = classifier.predict(X_tst)
    write_csv(predictions, 'kfold.csv')

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y])      


