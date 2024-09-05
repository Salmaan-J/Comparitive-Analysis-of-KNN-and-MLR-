from flask import jsonify,Flask


app = Flask(__name__)
with open('app\config.txt','r') as file:
                        app.config['SECRET_KEY']=file.read()#no hard coded code

@app.route('/api/update', methods=['GET'])
def update(): #test for now sending to homepage HTML.... TO swap with Get or another method
                return jsonify("data")


def start_server():
        app.run(debug=True)