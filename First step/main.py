#Import Modules where they are coded. Reason For Split to manage time and tasks
import sys
from sklearn.model_selection import train_test_split
from Model import Input
from Model import ML
from Model import Accuracy_Test as AT
import pandas as pd
import numpy as np
from scipy.stats import pearsonr



def Model_TestingKNN():
    #Testing KNN
    try:
        file_path = 'Dataset/filtered_data_x2.csv'
        file_path2 = 'Dataset/filtered_data_y2.csv'
                    # Read the CSV file, 
        x =pd.read_csv(file_path)
        y= pd.read_csv(file_path2)
    except Exception as e:
                        # Catch issue with file naming.
                print(f"Issue when  attempting to read files {e}")
                sys.exit(1) #gracefull exit
        # Drop the target column and convert the remaining columns to float

    y = y.astype(float) #convert y to float and place in its own List
        # Columns to convert to float
    x = x.astype(float)
    # Convert the X columns to float
    x,y=Input.data_cleaning(x,y)
    #print(x)
    #print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_test_norm,x_train_norm = Input.data_norm(x_train,x_test)
    df = pd.DataFrame(y_test)
    print(df)
    df.to_csv('Actual_rainfall.csv', index=False)
    ML.KNN(x_train_norm,y_train,x_test_norm,y_test,4)#Changes KNN over each run



def Model_TestingMLR():
    #Testing KNN
    try:
        file_path = 'Dataset/filtered_data_x2.csv'
        file_path2 = 'Dataset/filtered_data_y2.csv'
                    # Read the CSV file, 
        x =pd.read_csv(file_path)
        y= pd.read_csv(file_path2)
    except Exception as e:
                        # Catch issue with file naming.
                print(f"Issue when  attempting to read files {e}")
                sys.exit(1) #gracefull exit
        # Drop the target column and convert the remaining columns to float
    y = y.astype(float) #convert y to float and place in its own List
        # Columns to convert to float
    x = x.astype(float)
    # Convert the X columns to float
    x,y=Input.data_cleaning(x,y) 
    #print(x)
    #print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_test_norm,x_train_norm = Input.data_norm(x_train,x_test)
    ML.MLR(x_test_norm,y_test,x_train_norm,y_train)

Model_TestingMLR()
Model_TestingKNN()