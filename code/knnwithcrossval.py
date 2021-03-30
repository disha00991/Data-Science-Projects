from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import csv

stuff=np.load("../report_src/data.npz")
X_trn = stuff["X_trn"]
y_trn = stuff["y_trn"]
X_tst = stuff["X_tst"]

def question_7(X_trn, y_trn, X_tst):
    val = [1,3,5,7,9,11]

    #get k folds (k=5)
    kf = KFold(n_splits=5,shuffle=True, random_state=None)

    for k in val:
        fold_scores = []
        err=[]
        for train_index, test_index in kf.split(X_trn):
            
            Xk_train, Xk_test = X_trn[train_index], X_trn[test_index]
            yk_train, yk_test = y_trn[train_index], y_trn[test_index]

            # instantiate learning model for every value of k
            knn = KNeighborsClassifier(n_neighbors=k)

            # fitting the model
            knn.fit(Xk_train, yk_train)
    
            # predict the response
            pred = knn.predict(Xk_test)

            # predict the response
            err.append(1 - (accuracy_score(yk_test, pred)))

        fold_scores.append(sum(err)/len(err))
        for i in range(len(fold_scores)):
            print("Classification error for",k,":",fold_scores[0])

def question_8(X_trn, y_trn, X_tst):
    # instantiate learning model for every value of k
    knn = KNeighborsClassifier(n_neighbors=11)

    # fitting the model
    knn.fit(X_trn, y_trn)
    
    # predict the response
    pred = knn.predict(X_tst)

    write_csv(pred, 'knn.csv')

def write_csv(y_pred, filename):
    """Write a 1d numpy array to a Kaggle-compatible .csv file"""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Category'])
        for idx, y in enumerate(y_pred):
            csv_writer.writerow([idx, y]) 

question_8(X_trn, y_trn, X_tst)     


    