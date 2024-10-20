import os
from flask import jsonify,Flask,request
import joblib


app = Flask(__name__)
with open('Second Step/config.txt','r') as file:
                        app.config['SECRET_KEY']=file.read()#no hard coded code

@app.route('/api/update', methods=['GET'])
def update(): #test for now sending to homepage HTML.... TO swap with Get or another method
    #get data from SQL DB
    #data = request.get_json()
    data = {
        'message': 'Here is your data',
        'items': [1, 2, 3, 4, 5]}
    #response = { 'input': data, 'prediction': 'Rainy' if data['humidity'] > 70 else 'Clear'
    #}
    return jsonify(data)

def loadmodel():
        # Assuming 'knn' is your trained KNeighborsClassifier model
        model_folder_path = ''  # Specify the folder path where you want to save the model
        model_file_path = os.path.join(model_folder_path, 'knn_model.pkl')  # Specify the file path and name
        KNN = joblib.load(model_file_path)
        return KNN


def start_server(model_select):
        model = loadmodel()
        app.run(debug=True)#Set to false when Server is started in Demo

if __name__ == '__main__':
    app.run(debug=True)
