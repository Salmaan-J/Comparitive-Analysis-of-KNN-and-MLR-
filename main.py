#Import Modules where they are coded. Reason For Split to manage time and tasks
from app import API
from Model import Input
from Model import ML
from Model import Accuracy_Test as AT
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def modeltrain():
      select =[]
      x,y= Input.Read_CSV()
      while True:
            print("##Program Initialisation##")
            print("1. Train Model MLR")
            print("2. Train Model KNN")
            print("3. Launch Server")

            choice = input("\nEnter your choice (1-3): ")
            choice = choice.strip()
            select.append(choice)
            if choice =='0':
                correlation_with_y = {}
                corr = {}
                
                for column in x.columns:
                    correlation_with_y[column] = np.corrcoef(x[column], y)[0, 1]
                    #corr,_=pearsonr(x[column],y)   
                
                for x in correlation_with_y:
                     print(x)
                









            if choice == '1': 
                x_temp,y_temp=Input.data_cleaning(x,y,50)
                list1 = x['Basel Wind Direction [800 mb]']
                cor =list.corr(y)   
                print('Pearsons correlation: %.3f' % cor)
                x_train, x_test, y_train, y_test= Input.datasplt(x_temp,y_temp)
                x_test_norm,x_train_norm = Input.data_norm(x_train,x_test)
                print("\nData Preperation complete")
                mlr_pred = ML.MLR(x_test_norm,y_train,x_train_norm)
                print("MLR values")
                AT.calculateval(mlr_pred,y_test)
                
            elif choice == '2':
                x_test,x_train,y_train,y_test= Input.main()
                knn_pred = ML.KNN(x_train,y_train,x_test)
                print("KNN values")
                AT.calculateval(knn_pred,y_test)
            elif choice =='3':
                print("Select model to load: KNN or MLR")
                choice = input("\n ")
                select = choice.strip()
                API.start_server(select)  
                break         
            else:
                print("\nIncorrect input. Select the numbers of 1 - 3.")

modeltrain()