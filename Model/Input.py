#########Imports##########
import sqlite3
import sys
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
        file_path = 'Dataset/Temp_dataset.csv'
        # Read the CSV file, 
        df = pd.read_csv(file_path)
    except Exception as e:
            # Catch issue with file naming.
            print(f"Issue when  attempting to read files {e}")
            sys.exit(1) #gracefull exit

       
    df.columns = df.columns.str.replace('ï»¿timestamp', 'timestamp') # Clean the first column name if it contains wierd name on file
        # Drop the target column and convert the remaining columns to float
    x = df.drop(columns=['Basel Precipitation Total'])
    y = df['Basel Precipitation Total'].astype(float) #convert y to float and place in its own List
        # Columns to convert to float
    columns_to_convert = [
        'Basel Temperature [2 m elevation corrected]',
        'Basel Relative Humidity [2 m]',
        'Basel Wind Speed [800 mb]',
        'Basel Wind Direction [800 mb]',
        'Basel Shortwave Radiation']

        # Convert the X columns to float
    x[columns_to_convert] = x[columns_to_convert].astype(float)
    print("ReadCSV complete")
    #print(len(x)) 
    #print(len(y)) 
        #Testing : print(x.columns)print(x.dtypes)print(y.dtypes)# Print the column names and data types to check
    return x, y

###################################################################
##################### DATA PREPROCESSING ###############################

def data_cleaning(x, y):
    # THE goal of this class is to allow me to balance the dataset according to either rain or no rain.
    # Allows API control to allow for finer control and easier to test
    while True:
        try:
            print("\nInsert the percentage of no rain in the dataset. Where 100 is half rain and half no rain.")
            balance = input()
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
    #x_balanced = x_balanced.drop(columns=['timestamp']) #Keeping here for now as Data filtering removes timestamps. Might add back later
    # this section needs to be looked at as you are taking the no rain out and not adding rain.
    print("Complete Data Cleaning")
    return x_balanced, y_balanced

def filter_dataframe(x): 
    #This aims to improve the interactive nature of this model
    added =[] #using to identify what fields should go into the dataset
    x_edit=pd.DataFrame() #new Dataset to store edited Dataset.
    print("\nSelect a fields to train model on:")# First line
    while True:
        print(added)#display whats added
        print("1. Basel Temperature [2 m elevation corrected]")
        print("2. Basel Relative Humidity [2 m]")
        print("3. Basel Wind Speed [800 mb]")
        print("4. Basel Wind Direction [800 mb]")
        print("5. Basel Shortwave Radiation")
        print("6. Done")
        choice = input("Enter your choice (1-6): ")
        #I am choosing the option to improve data data engineering by allowing on the fly 
        #ensure duplicate values are not inseted into the model
        if choice == '1' and 'Basel Temperature [2 m elevation corrected]' not in added: 
            added.append('Basel Temperature [2 m elevation corrected]')

        elif choice == '2' and 'Basel Relative Humidity [2 m]' not in added :
            print("\nBasel Relative Humidity [2 m]")
            added.append('Basel Relative Humidity [2 m]')

        elif choice == '3' and 'Basel Wind Speed [800 mb]' not in added:
            print("\nBasel Wind Speed [800 mb]")
            added.append('Basel Wind Speed [800 mb]')

        elif choice == '4' and 'Basel Wind Direction [800 mb]' not in added:
            added.append('Basel Wind Direction [800 mb]')
            print("\nBasel Wind Direction [800 mb]")

        elif choice == '5' and 'Basel Shortwave Radiation' not in added:
            print("\nBasel Shortwave Radiation")
            added.append('Basel Shortwave Radiation')

        elif choice == '6':
            print("\nEnd")
            break

        else:
            print("Invalid choice or Field selected already")
        print("Selected fields:",added)
    if len(added)>0:
        x_edit=x[added]
    else:
        print("No Fields were selected.")
        sys.exit("Issue splitting the file")
    return x_edit
def data_norm(x_train,x_test):

# Feature Normalization

    scaler = MinMaxScaler()  # Initialize the MinMaxScaler for normalization
    x_train_norm = scaler.fit_transform(x_train)  # Fit the scaler on the training data and transform it
    x_test_norm = scaler.transform(x_test)  # Transform the testing data using the already fitted scaler
    print("Complete Data Normilization")
    return x_test_norm, x_train_norm  # Return the normalized testing and training data

def datasplt(size,seed, x, y):
    seed=42 #hardcode for now Reason added multiple input is for easier management.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=seed) #state = seed of 42:  70/30 split
    print("Complete Data Splitting")
    return x_train, x_test, y_train, y_test


###########################
##### Data plotting##########
def dataplot(x,y):
    if len(x)>100: 
        sample_size = 100

    x_sample = x.sample(n=sample_size, random_state=1)
    y_sample = y.loc[x_sample.index]

    # Verify and print column names and Series name
    print("Columns in x:", x.columns)
    print("Name of y (Series):", y.name)

    # Plot Independent Variables
    plt.figure(figsize=(18, 12))
    if 'Basel Temperature [2 m elevation corrected]'in x_sample:
        plt.subplot(2, 3, 1)
        plt.scatter(x_sample.index, x_sample['Basel Temperature [2 m elevation corrected]'], c='b', edgecolor='k', s=50)
        plt.title('Temperature Over Time')
        plt.xlabel('Index')
        plt.ylabel('Temperature')
    if 'Basel Relative Humidity [2 m]' in x_sample:
        plt.subplot(2, 3, 2)
        plt.scatter(x_sample.index, x_sample['Basel Relative Humidity [2 m]'], c='g', edgecolor='k', s=50)
        plt.title('Humidity Over Time')
        plt.xlabel('Index')
        plt.ylabel('Humidity')
    if 'Basel Wind Speed [800 mb]' in x_sample:
        plt.subplot(2, 3, 3)
        plt.scatter(x_sample.index, x_sample['Basel Wind Speed [800 mb]'], c='r', edgecolor='k', s=50)
        plt.title('Wind Speed Over Time')
        plt.xlabel('Index')
        plt.ylabel('Wind Speed')
    if 'Basel Wind Direction [800 mb]' in x_sample:
        plt.subplot(2, 3, 4)
        plt.scatter(x_sample.index, x_sample['Basel Wind Direction [800 mb]'], c='c', edgecolor='k', s=50)
        plt.title('Wind Direction Over Time')
        plt.xlabel('Index')
        plt.ylabel('Wind Direction')
    if 'Basel Shortwave Radiation' in x_sample:
        plt.subplot(2, 3, 5)
        plt.scatter(x_sample.index, x_sample['Basel Shortwave Radiation'], c='m', edgecolor='k', s=50)
        plt.title('Shortwave Radiation Over Time')
        plt.xlabel('Index')
        plt.ylabel('Shortwave Radiation')

    # Plot Dependent Variable (Series)
    plt.subplot(2, 3, 6)
    plt.scatter(y_sample.index, y_sample.values, c='b', edgecolor='k', s=50)
    plt.title('Precipitation Over Time')
    plt.xlabel('Index')
    plt.ylabel('Precipitation')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

###################### Error Handling for input####################
                       #########################

#switched to handling in function, No recurring error requires issue function



###################################################################
#################DATABASE SECTION#############################


def SavetoDB():
    import sqlite3
    # Connect to the SQLite database (creates a new one if it doesn't exist)
    conn = sqlite3.connect('example.db')
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Create a table if it doesn't exist
    # Sample data
    # Insert data into the table
    #cursor.executemany('INSERT INTO users (name, email) VALUES (?, ?)', user_data)
    # Commit changes to the database
    conn.commit()
    # Close the cursor and connection
    cursor.close()
    conn.close()
    return "success"

def readfromDB():
    # Connect to the SQLite database
    conn = sqlite3.connect('Save.db')
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # SELECT query to retrieve data from the users table
    cursor.execute('SELECT * FROM users')
    # Fetch all rows from the result of the SELECT query
    rows = cursor.fetchall()
    # Print the retrieved data
    for row in rows:
        print(row)
    # Close the cursor and connection
    cursor.close()
    conn.close()

    return "success"



###############################################################
#########RUNNING##############################################

def main():
    x,y= Read_CSV()
    x,y=data_cleaning(x,y)
    x=filter_dataframe(x)
    dataplot(x,y)
    x_train, x_test, y_train, y_test= datasplt(0.7,42,x,y)
    x_test_norm,x_train_norm = data_norm(x_train,x_test)
    print("Ready to train.")
    return x_test_norm,x_train_norm,y_train,y_test


main()