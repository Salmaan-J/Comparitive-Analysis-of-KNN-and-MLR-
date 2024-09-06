import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as k


def MLR(x_train,y_train,x_test):
    mlr_model = LinearRegression()
    mlr_model.fit(x_train, y_train)
    y_pred = mlr_model.predict(x_test)
    #print(k.root_mean_squared_error(y_test, y_pred))
    model_folder_path = 'Model/'  # Specify the folder path where you want to save the model
    os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_file_path = os.path.join(model_folder_path, 'MLR.pkl')  # Specify the file path and name
    joblib.dump(mlr_model, model_file_path)
    return y_pred

#############Implement Accuracy Models##########
def KNN(x_train,y_train,x_test):
    while True:
            try:
                n_neighbors = input("\nInsert the number of neighbors")
                n_neighbors = int(n_neighbors)
                break
            except ValueError:
                print("\nInvalid input. Please enter an integer.")
    knn_model = KNeighborsRegressor(n_neighbors)
    knn_model.fit(x_train,y_train)
    y_pred=knn_model.predict(x_test)
    model_folder_path = 'Model/'  # Specify the folder path where you want to save the model
    os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_file_path = os.path.join(model_folder_path, 'KNN.pkl')  # Specify the file path and name
    joblib.dump(knn_model, model_file_path)
    return y_pred




