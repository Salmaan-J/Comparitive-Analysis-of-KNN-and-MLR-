#########Imports##########
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
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

def data_cleaning(x,y):
    # THE goal of this class is to allow me to balance the dataset according to either rain or no rain.
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
    #y_rain = pd.Series(dtype='float64', name='Basel Precipitation Total')
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
    # Iterate over the target Series to split data UPDATE************* was using an iterative method that was  bad and slow. 
    # much more efficient. Splits with row number
      # This uses boolean indexing to compare values within the series Y and return the index. Taking out of the existing frame then pushing it to a new frame.
    y_series = y.iloc[:, 0]  # This extracts the first column as a Series
    y_rain = y_series[y_series > 0]
    #print(y_rain)
    x_rain = x.loc[y_rain.index]
    #print(x_rain)
    y_norain = y_series[y_series == 0]
    x_norain = x.loc[y_norain.index]

    # Creating a balanced dataset to test the model for rain.
    # Create an edit to the below that allows the user to place 0-1 that allows the user to indicate if bias to rain or bias to no rain.
    # print("data Cleaning")
    sample_size = len(y_rain)
    #print('print Sample' +str(sample_size))
    #print("data Cleaning") print(sample_size)  print(len(y_rain))
    x_norain_balanced = x_norain.sample(n=sample_size, random_state=1)
    y_norain_balanced = y_norain.loc[x_norain_balanced.index]
    x_balanced = pd.concat([x_rain, x_norain_balanced])
    y_balanced = pd.concat([y_rain, y_norain_balanced])
    #y_balanced = np.where(y_balanced > 0, 1, 0)
    # this section needs to be looked at as you are taking the no rain out and not adding rain.
    print("Complete Data Cleaning")
    return x_balanced, y_balanced


def data_norm(x_train,x_test):

# Feature Normalization
    scaler = MinMaxScaler()  # Initialize the MinMaxScaler for normalization
    print(x_train.columns)
    x_train_norm = scaler.fit_transform(x_train)  # Fit the scaler on the training data and transform it
    x_test_norm = scaler.transform(x_test)  # Transform the testing data using the already fitted scaler
    print("Complete Data Normilization")
    joblib.dump(scaler, 'scaler.gz')
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

#Filtered data creating
def corr_calculation2():
    x,y=Read_CSV()
    correlation_with_y = pd.Series(index=x.columns)
    columns_to_drop = []
    columns_to_keep = pd.Series(index=x.columns)
    #corr = {}
    for column in x.columns:
        correlation_with_y[column] = np.corrcoef(x[column], y)[0, 1]         
    columns_to_keep = correlation_with_y
    # Loop through the dictionary items (column and corresponding correlation value)
    for column, corrcof in correlation_with_y.items():
        # remove values lower than 0.1
        if corrcof <= 0.1:
            # Add the column to the list if the condition is met
            print(corrcof)
            columns_to_drop.append(column)
            columns_to_keep.drop(column)
    # Drop these columns from x
    x_filtered = x.drop(columns=columns_to_drop)
    sorted_correlations = columns_to_keep.sort_values(ascending=False)
    #print(sorted_correlations)
    x_filtered.to_csv("filtered_data_x2.csv", index=False)
    y.to_csv("filtered_data_y2.csv", index=False)
    sorted_correlations.to_csv("sorted_correlations.csv", header=True)
    #Produces the Correlation coeff 
    
###############################################################
#########RUNNING##############################################



