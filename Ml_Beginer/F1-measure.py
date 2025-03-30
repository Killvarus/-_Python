import numpy
def precision(y_true, y_pred):

    TP = (y_true * y_pred).sum()
    FP = (y_pred * (1 - y_true)).sum()
    # TN = ((1 - y_pred) * (1 - y_true)).sum()
    # FN = ((1 - y_pred) * y_true).sum()
    return(TP/(TP+FP))
def recall(y_true, y_pred):
    TP = (y_true * y_pred).sum()
    # FP = (y_pred * (1 - y_true)).sum()
    # TN = ((1 - y_pred) * (1 - y_true)).sum()
    FN = ((1 - y_pred) * y_true).sum()
    return(TP/(TP+FN))

def f1(y_true, y_pred):
    return 2*(precision(y_true,y_pred)*recall(y_true, y_pred)/(precision(y_true,y_pred)+recall(y_true, y_pred)))