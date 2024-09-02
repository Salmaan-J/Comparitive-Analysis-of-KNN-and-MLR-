from app import app
from flask import jsonify



with open('app\config.txt','r') as file:
        app.config['SECRET_KEY']=file.read()#no hard coded code

@app.route('/api/update')

def update(): #test for now sending to homepage HTML.... TO swap with Get or another method
    
    return jsonify("data")



