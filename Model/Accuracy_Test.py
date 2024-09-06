######################################################
########## TESTING DATA OUPTUT AND THE MODEL ACCURACY###

def accuracy(i,j):
    #i = num testeted samples forcasted correctly
    #j =  total test samples
    num = i/j
    formatted_num = "{:.2f}".format(num)
    return formatted_num


def TS(ha,ma,fa):
    num = (ha/(ha+fa+ma))
    formatted_num = "{:.2f}".format(num)
    return formatted_num

def MAR(ma,ha):
    num =(ma/(ma+ha))
    formatted_num = "{:.2f}".format(num)
    return formatted_num

def precision(ha,fa):
    num = (ha/(ha+fa))
    formatted_num = "{:.2f}".format(num)
    return formatted_num


def calculateval(y_pred,y_test):
 # here I need to create a function that that calculates thefunctions to calculate the accuracy.    
    return "Null"