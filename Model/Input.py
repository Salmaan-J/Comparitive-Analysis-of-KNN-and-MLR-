#########Imports##########
import sqlite3
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import math
################################
### Global Variables###########

################ ###################
###### DATA INPUT ##########
def Read_CSV():  
    try:
        file_path = 'Dataset/Final_dataset.csv'
        # Read the CSV file, 
        df = pd.read_csv(file_path)
    except Exception as e:
            # Catch issue with file naming.
            print(f"Issue when  attempting to read files {e}")
            sys.exit(1) #gracefull exit
    # Drop the target column and convert the remaining columns to float
    y = df['Basel Precipitation Total'].astype(float) #convert y to float and place in its own List
    x = df.drop(columns=['Basel Precipitation Total'])
        # Columns to convert to float
    for columns in x.columns:
        x[columns] = x[columns].astype(float)
        # Convert the X columns to float
    print("ReadCSV complete")
    return x, y

###################################################################
##################### DATA PREPROCESSING ###############################

def data_cleaning(x, y,balance):
    # THE goal of this class is to allow me to balance the dataset according to either rain or no rain.
    # Allows API control to allow for finer control and easier to test
    while True:
        try:
            balance = int(balance)
            break
        except ValueError:
            print("\nInvalid input. Please enter an integer.")
    # Initializations
    #X values
    x_norain = pd.DataFrame(columns=x.columns)
    x_norain_balanced = pd.DataFrame(columns=x.columns)
    x_balanced = pd.DataFrame(columns=x.columns)
    x_rain = pd.DataFrame(columns=x.columns)
    #Y values
    y_balanced = pd.Series(dtype='float64', name='Basel Precipitation Total')
    y_norain_balanced = pd.Series(dtype='float64', name='Basel Precipitation Total')
    y_norain = pd.Series(dtype='float64', name='Basel Precipitation Total')
    y_rain = pd.Series(dtype='float64', name='Basel Precipitation Total')
    #k = 0  # Used for debugging 
    x_null = x.isnull().sum()
    y_null = y.isnull().sum()
    # Execution
    # Calculate the number of null values in each column of x
    all_zero = (x_null == 0).all()
    if not all_zero:
        print("Some columns in the X dataframe have missing values.")
        return "Check console"
    all_zero = (y_null == 0).all()
    if not all_zero:
        print("Some columns in the Y dataframe have missing values.")
        return "Check console"
    # Determine the number of samples to balance the dataset. Doing this can help me build the ability to change this while testing in future
    # Note this section is where we start splitting between rain and no Rain, Then reconstruct the dataset with a balance or bias towards rain or no rain. Might use another method
    # Iterate over the target Series to split data UPDATE************* was using an iterative method that was  bad and slow. 
    # much more efficient. Splits with row number
    rain_checker = y > 0  # This uses boolean indexing to compare values within the series Y and return the index. Taking out of the existing frame then pushing it to a new frame.
    norain_checker = y == 0
    y_rain = y[rain_checker]
    x_rain = x[rain_checker]
    y_norain = y[norain_checker]
    x_norain = x[norain_checker]

    # Creating a balanced dataset to test the model for rain.
    # Create an edit to the below that allows the user to place 0-1 that allows the user to indicate if bias to rain or bias to no rain.
   # print("data Cleaning")
    sample_size = len(y_rain)
    #print(sample_size)
    sample_size = sample_size - math.floor((sample_size / 100) * balance)# allowing me to create differnt size dataset 
   # print("data Cleaning") print(sample_size)  print(len(y_rain))

    x_norain_balanced = x_norain.sample(n=sample_size, random_state=1)
    y_norain_balanced = y_norain.loc[x_norain_balanced.index]

    x_balanced = pd.concat([x_rain, x_norain_balanced])
    y_balanced = pd.concat([y_rain, y_norain_balanced])
    # this section needs to be looked at as you are taking the no rain out and not adding rain.
    print("Complete Data Cleaning")
    return x_balanced, y_balanced


def data_norm(x_train,x_test):

# Feature Normalization
    scaler = MinMaxScaler()  # Initialize the MinMaxScaler for normalization
    #print(x_train.columns)
    x_train_norm = scaler.fit_transform(x_train)  # Fit the scaler on the training data and transform it
    x_test_norm = scaler.transform(x_test)  # Transform the testing data using the already fitted scaler
    print("Complete Data Normilization")
    x_train_norm= pd.DataFrame(x_train_norm, columns=x_train.columns)
    x_test_norm = pd.DataFrame(x_test_norm, columns=x_train.columns)
    #print(x_train_norm.columns)
    # add the ability to view
    return x_test_norm, x_train_norm  # Return the normalized testing and training data

def datasplt(x, y, size):
    seed=42 #hardcode for now Reason added multiple input is for easier management.
    while True:
        #print("Select the ratio of Test to Training Data (Between 1-100)")
        #size = input("Insert here: ").strip()
        try:
            size = int(size)
            if 1 <= size <= 100:
                break
            else:
                print("Insert a number only between 1 - 100")
        except:
            print("Insert a number only between 1 - 100")
    size = size/100
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=seed) #state = seed of 42:  70/30 split
    #print("Complete Data Splitting")
    return x_train, x_test, y_train, y_test

###############################

#Filtered data creatting
def corr_calculation():
    x,y=Read_CSV()
    correlation_with_y = pd.Series(index=x.columns)
    columns_to_drop = []
    columns_to_keep = pd.Series(index=x.columns)
    #corr = {}
    for column in x.columns:
        #print(column)
        correlation_with_y[column] = np.corrcoef(x[column], y)[0, 1]              
    columns_to_keep = correlation_with_y
    # Loop through the dictionary items (column and corresponding correlation value)
    for column, corrcof in correlation_with_y.items():
        # Check if the absolute value of the correlation is less than or equal to 0.1
        if abs(corrcof) <= 0.1:
            # Add the column to the list if the condition is met
            columns_to_drop.append(column)
            columns_to_keep.drop(column)
    # Drop these columns from x
    x_filtered = x.drop(columns=columns_to_drop)
    #print(f"Columns dropped due to low correlation: {columns_to_drop}")
    sorted_correlations = columns_to_keep.sort_values(ascending=False)
    #print(sorted_correlations)
    x_filtered.to_csv("filtered_data_x.csv", index=False)
    y.to_csv("filtered_data_y.csv", index=False)
    sorted_correlations.to_csv("sorted_correlations.csv", header=True)



###############################################################
#########RUNNING##############################################

def main():
    x,y= Read_CSV()
    x,y=data_cleaning(x,y)
   # x=filter_dataframe(x)
   # dataplot(x,y)
    x_train, x_test, y_train, y_test= datasplt(x,y)
    x_test_norm,x_train_norm = data_norm(x_train,x_test)
    print("Ready to train.")
    return x_test_norm,x_train_norm,y_train,y_test
