#Import Modules where they are coded. Reason For Split to manage time and tasks
import sys
from app import API
from Model import Input
from Model import ML
from Model import Accuracy_Test as AT
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def modeltrain():
    select =[]
    try:
                    file_path = 'Dataset/filtered_data_x.csv'
                    file_path2 = 'Dataset/filtered_data_y.csv'
                    # Read the CSV file, 
                    x = pd.read_csv(file_path)
                    y= pd.read_csv(file_path2)
    except Exception as e:
                        # Catch issue with file naming.
                        print(f"Issue when  attempting to read files {e}")
                        sys.exit(1) #gracefull exit
                # Drop the target column and convert the remaining columns to float
    y = y['Basel Precipitation Total'].astype(float) #convert y to float and place in its own List
                    # Columns to convert to float
    for columns in x.columns:
                x[columns] = x[columns].astype(float)
                    # Convert the X columns to float
    x,y=Input.data_cleaning(x,y,100)
    x_train, x_test, y_train, y_test= Input.datasplt(x,y,70)
                #print(x_train)
    x_test_norm,x_train_norm = Input.data_norm(x_train,x_test)
    #mlr_pred = ML.MLR(x_test_norm,y_test,x_train_norm,y_train)
    knn = ML.KNN(x_test_norm,y_test,x_train_norm,y_train)
    print("DONE")
                #AT.calculateval(mlr_pred,y_test)
    while True:
            break
            print(" Program Initialisation")
            print("##Program Initialisation##")
            print("1. Train Model MLR")
            print("2. Train Model KNN")
            print("3. Launch Server")

            choice = input("\nEnter your choice (1-3): ")
            choice = choice.strip()
            select.append(choice)
            #x_test,x_train,y_train,y_test= Input.main()
            if choice == '1':       
                try:
                    file_path = 'Dataset/filtered_data_x.csv'
                    file_path2 = 'Dataset/filtered_data_y.csv'
                    # Read the CSV file, 
                    x = pd.read_csv(file_path)
                    y= pd.read_csv(file_path2)
                except Exception as e:
                        # Catch issue with file naming.
                        print(f"Issue when  attempting to read files {e}")
                        sys.exit(1) #gracefull exit
                # Drop the target column and convert the remaining columns to float
                y = y['Basel Precipitation Total'].astype(float) #convert y to float and place in its own List
                    # Columns to convert to float
                for columns in x.columns:
                    x[columns] = x[columns].astype(float)
                    # Convert the X columns to float
                print("ReadCSV complete")
                x_train, x_test, y_train, y_test= Input.datasplt(x,y,70)
                #print(x_train)
                x_test_norm,x_train_norm = Input.data_norm(x_train,x_test)
                
                print("\nData Preperation complete")
                mlr_pred = ML.MLR(x_train_norm,y_train,x_test_norm)
                print("MLR values")
                #AT.calculateval(mlr_pred,y_test)

            elif choice == '2':
                x_test,x_train,y_train,y_test= Input.main()
                knn_pred = ML.KNN(x_train,y_train,x_test)
                print("KNN values")
                AT.calculateval(knn_pred,y_test)
            elif choice =='3':
                API.start_server()  
                print("Select model to load: KNN or MLR")
                choice = input("\n ")
                select = choice.strip()
                API.start_server(select)  
                break         
            else:
                print("\nIncorrect input.")
                print("\nIncorrect input. Select the numbers of 1 - 3.")




# Initialize an empty list to hold the selected features
modeltrain()
