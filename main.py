#Import Modules where they are coded. Reason For Split to manage time and tasks
from app import app
from Model import Input
from Model import ML
from Model import Accuracy_Test as AT
import os
import joblib


select =[]
# the goal here in Main is to first take input from the user to check if the 
# model is created then move from this point to implement and start the App or go through the process of Identifying the app.
# This will ensure the correct flow.







def loadmodel(type):
    if type ==1:
        model_folder_path = ''  # Specify the folder path where you want to save the model
        os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        model_file_path = os.path.join(model_folder_path, 'KNN.pkl')  # Specify the file path and name
        KNN = joblib.load(model_file_path)
        return KNN
    elif type == 2:
        # Assuming 'knn' is your trained KNeighborsClassifier model
        model_folder_path = ''  # Specify the folder path where you want to save the model
        os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        model_file_path = os.path.join(model_folder_path, 'MLR.pkl')  # Specify the file path and name
        MLR = joblib.load(model_file_path)
        return MLR
    return "NULL"

def main():
    print(len(select))
    if len(select)==0:
        while True:
            print(" Program Initialisation")
            print("1. Train Model")
            print("2. Launch Server")

            choice = input("\nEnter your choice (1-3): ")
            choice = choice.strip()
            select.append(choice)
            if choice == '1': 
                x_test,x_train,y_train,y_test= Input.main()      
                print("\nData Preperation complete")
                mlr_pred = ML.MLR(x_train,y_train,x_test)
                knn_pred = ML.KNN(x_train,y_train,x_test)
                AT.calculateval(mlr_pred,y_test)
            elif choice == '2':
                break           
            else:
                if '1' not in select:
                    print("\nPlease train the model first before launching server.")
                else:
                    print("\nIncorrect input.")
    app.run(debug=True)

if __name__=='__main__':
    main()