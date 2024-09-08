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

def SAR(ha,ts):
    num = (ha/ts)
    formatted_num = "{:.2f}".format(num)
    return formatted_num


def calculateval(y_pred,y_test):
 # here I need to create a function that that calculates thefunctions to calculate the accuracy.
    hitalarms = 0
    false_alarms = 0
    miss_alarms = 0
    correct_alarms = 0
    test_samples = len(y_pred)
    correct_test_samples=0
    if len(y_pred)!= len(y_test):
        print("Note the model has not produced the same amount of predictions than what is in test")
        return 0
    for p, o in zip(y_pred, y_test): 
        if p > 0 and o > 0:
            hitalarms += 1
        elif  p > 0 and o == 0:
            false_alarms += 1
        elif p == 0 and o > 0:
            miss_alarms +=1
        elif  p == 0 and o == 0:
            correct_alarms+=1
    #print("Hit Alarms:"+str(hitalarms))
    #print("False Alarms: "+str(false_alarms))
    #print("Miss Alarms: "+str(miss_alarms))
    #print("Correct Alarms:"+str(correct_alarms))
    
    print("Completed model calculations \n")
    correct_test_samples = hitalarms + correct_alarms
    acc = accuracy(correct_test_samples,test_samples)
    ThreatS = TS(hitalarms,miss_alarms,false_alarms)
    Summary_AR= SAR(hitalarms,test_samples)
    Missing_ar = MAR(miss_alarms,hitalarms)
    prec = precision(hitalarms,false_alarms)
    print(f"Accuracy: {acc}, Threat Score: {ThreatS}, Summary AR: {Summary_AR}, Missing AR: {Missing_ar}, Precision: {prec}")
    return 1 #acc,ThreatS,Summary_AR,Missing_ar,prec

