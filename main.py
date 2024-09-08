#Import Modules where they are coded. Reason For Split to manage time and tasks
from app import API
from Model import Input
from Model import ML
from Model import Accuracy_Test as AT

select =[]
# the goal here in Main is to first take input from the user to check if the 
# model is created then move from this point to implement and start the App or go through the process of Identifying the app.
# This will ensure the correct flow.





def modeltrain():
      while True:
            print(" Program Initialisation")
            print("1. Train Model MLR")
            print("2. Train Model KNN")
            print("2. Launch Server")

            choice = input("\nEnter your choice (1-3): ")
            choice = choice.strip()
            select.append(choice)
            x_test,x_train,y_train,y_test= Input.main()
            if choice == '1':       
                print("\nData Preperation complete")
                mlr_pred = ML.MLR(x_train,y_train,x_test)
                
                print("MLR values")
                AT.calculateval(mlr_pred,y_test)
                
            elif choice == '2':
                knn_pred = ML.KNN(x_train,y_train,x_test)
                print("KNN values")
                AT.calculateval(knn_pred,y_test)
            elif choice =='3':
                API.start_server()  
                break         
            else:
                print("\nIncorrect input.")

modeltrain()