import csv
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as k
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from Model.Accuracy_Test import calculateval



def MLR(x_train,y_train,x_test,y_test):

    selected_features = []
    #print(f"x_test_norm shape: {x_test.shape}")   #print(x_test)
    #print(f"x_train_norm shape: {x_train.shape}")
    #print(f"Selected features: {selected_features}")
    #Initialize the master list of selected features
    master_feature_list = list(selected_features)
    baseline_performance = None  
    csv_file_path = 'model_performance.csv'
    # Create the CSV file and write headers if the file doesn't already exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Selected Features", "Adjusted R²", "RMSE", "R²","MSE","Accuracy","TS","SAR","MAR","Precision"])
    # Loop through all features and add one at a time
    for feature in x_train.columns:          
        # Check if feature is already in the selected list (avoid duplicates)
        if feature not in master_feature_list:
            # Create a temporary list with current features + the new one
            current_features = master_feature_list + [feature]
            
            # Train the model with the current features
            mlr_model = LinearRegression()
            mlr_model.fit(x_train[current_features], y_train)
            #Evaluate on validation data
            #Use cross-validation to evaluate the model  
            y_pred = mlr_model.predict(x_test[current_features])
            current_performance = mean_squared_error(y_test, y_pred)              
            r2 = r2_score(y_test, y_pred)
            rmse =root_mean_squared_error(y_test, y_pred)
            adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(current_features) - 1)
                # Print performance metrics
            print(f"Evaluating feature: {feature}")
            print(f'Adjusted R-Squared: {adjusted_r2}')
            print(f'Root Mean Squared Error (RMSE): {rmse}')
            print(f'Mean Squared Error (MSE): {current_performance}')
            print(f'R-squared (R^2): {r2}')
            Acc,ThreatS,Summary_AR,Missing_ar,prec=calculateval(y_pred, y_test)
            with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([current_features, adjusted_r2, rmse, r2,current_performance,Acc,ThreatS,Summary_AR,Missing_ar,prec])
                    # Compare performance to baseline and update the feature list
            if baseline_performance is None:
                print(f"Setting baseline performance to {current_performance} as it's the first comparison")
                baseline_performance = current_performance
            if current_performance < baseline_performance:
                print(f"Adding feature {feature} improves performance: {current_performance} < {baseline_performance}")
                master_feature_list.append(feature)
                baseline_performance = current_performance
            elif current_performance > baseline_performance:
                print(f"Feature {feature} does not improve performance: {current_performance} >= {baseline_performance}")
    # Output the final selected feature list

    print(f"Selected features so far: {master_feature_list}")
    mlr_model = LinearRegression()
    mlr_model.fit(x_train[master_feature_list], y_train)
    #Evaluate on validation data
    #could use cross-validation to evaluate the model  
    y_pred = mlr_model.predict(x_test[master_feature_list])
    current_performance = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(master_feature_list) - 1)
                # Print performance metrics
    print(f"Final feature: {master_feature_list}")
    print(f'Adjusted R-Squared: {adjusted_r2}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Squared Error (MSE): {current_performance}')
    print(f'R-squared (R^2): {r2}')
    #final_y_pred.to_csv("Final_y_pred.csv", index=False)
    print(y_pred)
    y_array = y_test.values
    print(y_array)
    calculateval(y_pred, y_test)
     

def KNN(x_train, y_train, x_test, y_test,input):
    selected_features = []
    n_neighbors = input
    master_feature_list = list(selected_features)
    baseline_performance = None
    csv_file_path = 'model_performance.csv'
    
    # Create the CSV file and write headers if the file doesn't already exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["n_neighbors","Selected Features", "Adjusted R²", "RMSE", "R²","MSE","Accuracy","TS","SAR","MAR","Precision"])
    for feature in x_train.columns:
                # Check if the feature is already selected (avoid duplicates)
                if feature not in master_feature_list:
                    # Add current feature to the selected list
                    current_features = master_feature_list + [feature]   
                
                # Train the KNN model
                knn_model = KNeighborsRegressor(n_neighbors)
                knn_model.fit(x_train[current_features], y_train)
                y_pred = knn_model.predict(x_test[current_features])
                # Evaluate model performance
                current_performance = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = root_mean_squared_error(y_test,y_pred)
                print(current_performance)
                r3=rmse *rmse
                adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(current_features) - 1)
                # Print performance metrics
                print(f"Evaluating feature: {feature}")
                print(f"Testing with n_neighbors = {n_neighbors}")
                print(f'Adjusted R-Squared: {adjusted_r2}')
                print(f'Root Mean Squared Error (RMSE): {rmse}')
                print(f'Mean Squared Error (MSE): {current_performance}')
                print(f'R-squared (R^2): {r2}')
                Acc,ThreatS,Summary_AR,Missing_ar,prec=calculateval(y_pred, y_test)
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([n_neighbors,current_features, adjusted_r2, rmse, r2,current_performance,Acc,ThreatS,Summary_AR,Missing_ar,prec])
                    # Compare performance to baseline and update the feature list
                if baseline_performance is None:
                    print(f"Setting baseline performance to {current_performance} as it's the first comparison")
                    baseline_performance = current_performance
                
                if current_performance < baseline_performance:
                    print(f"Adding feature {feature} and n_neighbors={n_neighbors} improves performance: {current_performance} < {baseline_performance}")
                    master_feature_list.append(feature)  # Keep the feature
                    baseline_performance = current_performance
                else:
                    print(f"Feature {feature} and n_neighbors={n_neighbors} do not improve performance: {current_performance} >= {baseline_performance}")
        
        # Append the selected features and neighbors to the best list
        
    print(f"Selected features so far: {master_feature_list}")
    final_knn_model = KNeighborsRegressor(n_neighbors)
    final_knn_model.fit(x_train[master_feature_list], y_train)
    with open('Second Step/knn_model.pkl', 'wb') as f:
        pickle.dump(final_knn_model, f)
    final_y_pred = final_knn_model.predict(x_test[master_feature_list])
    current_performance = mean_squared_error(y_test, final_y_pred)
    r2 = r2_score(y_test, final_y_pred)
    rmse = np.sqrt(current_performance)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(master_feature_list) - 1)
                # Print performance metrics
    print(f"Final feature: {master_feature_list}")
    print(f"Testing with n_neighbors = {n_neighbors}")
    print(f'Adjusted R-Squared: {adjusted_r2}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Squared Error (MSE): {current_performance}')
    print(f'R-squared (R^2): {r2}')

    # Convert to DataFrame    
    df = pd.DataFrame(final_y_pred)
    # Write to CSV
    df.to_csv('predicted_rainfall.csv', index=False)
   
    calculateval(final_y_pred, y_test)
 
