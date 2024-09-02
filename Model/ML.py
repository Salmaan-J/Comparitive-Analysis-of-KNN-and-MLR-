import os
import Input
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as k


def MLR(x_train,x_test,y_train,y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #print(k.root_mean_squared_error(y_test, y_pred))
    model_folder_path = 'Model/'  # Specify the folder path where you want to save the model
    os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_file_path = os.path.join(model_folder_path, 'MLR.pkl')  # Specify the file path and name
    joblib.dump(model, model_file_path)
    return y_pred

#############Implement Accuracy Models##########
def KNN(x_train,x_test,y_train,y_test):
    knn_model = KNeighborsRegressor(n_neighbors=2)
    knn_model.fit(x_train,y_train)
    y_pred=knn_model.predict(x_test)
    model_folder_path = 'Modal/'  # Specify the folder path where you want to save the model
    os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_file_path = os.path.join(model_folder_path, 'KNN.pkl')  # Specify the file path and name
    joblib.dump(model, model_file_path)
    #print(k.root_mean_squared_error(y_test,y_pred))
    #print(k.mean_squared_error(y_test,y_pred))
    return y_pred


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




