import numpy as np
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
        classifier = ''
        for train_index, test_index in kf.split(X_trn):
            Xk_train, Xk_test = X_trn[train_index], X_trn[test_index]
            yk_train, yk_test = y_trn[train_index], y_trn[test_index]

            classifier = train_tree_classifier(Xk_train, yk_train, depth)
            fold_scores.append(classifier.score(Xk_test, yk_test))
        
        error = 1 - np.mean(np.array(fold_scores))
        print(f"{depth}  |  {error}")
        depth_errors.append(error)

    best_depth = max_depth[np.argmin(np.array(depth_errors))]
    print(f"best depth : {best_depth}")         

def train_tree_classifier(X_trn, y_trn, max_depth):
    classifier = DecisionTreeClassifier(random_state=1, max_depth=max_depth)
    classifier.fit(X_trn, y_trn)
    return classifier

