import os
from flask import jsonify,Flask
import joblib


app = Flask(__name__)
with open('app\\config.txt','r') as file:
                        app.config['SECRET_KEY']=file.read()#no hard coded code

@app.route('/api/update', methods=['GET'])
def update(): #test for now sending to homepage HTML.... TO swap with Get or another method
                return jsonify("")

def loadmodel(type):
    if type ==1:
        model_folder_path = ''  # Specify the folder path where you want to save the model
        model_file_path = os.path.join(model_folder_path, 'KNN.pkl')  # Specify the file path and name
        KNN = joblib.load(model_file_path)
        return KNN
    elif type == 2:
        # Assuming 'knn' is your trained KNeighborsClassifier model
        model_folder_path = ''  # Specify the folder path where you want to save the model
        model_file_path = os.path.join(model_folder_path, 'MLR.pkl')  # Specify the file path and name
        MLR = joblib.load(model_file_path)
        return MLR


def start_server(model_select):
        model = loadmodel(model_select)
        app.run(debug=False)#Set to false when Server is started in Demo