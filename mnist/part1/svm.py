import numpy as np
from sklearn.svm import LinearSVC
# https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-iii-5dff33fa015d
# https://scikit-learn.org/stable/modules/svm.html#id18
# original paper:
# http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf

# Who are the support vectors?
# Support vector is a sample that is incorrectly classified or a sample close to a boundary.
# In SVM, only support vectors has an effective impact on model training,
# that is saying removing non support vector has no effect on the model at all.


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    linsvc = LinearSVC()
    linsvc.fit(train_x, train_y)
    pred_y = linsvc.predict(test_x)
    return (pred_y)



def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    linsvc = LinearSVC(multi_class="ovr")
    linsvc.fit(train_x, train_y)
    pred_y = linsvc.predict(test_x)
    return (pred_y)


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

# plot examples:
# svm = SVC(kernel="linear")
# svm.fit(...)
# plot_classifier(X, y, svm, lims(11,15,0,6))
