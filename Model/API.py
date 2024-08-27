from flask import Flask
from flask_sqlalchemy import SQLAlchemy

server = Flask(__name__) #Creating the server setting it to name
server.config['SQLALCHEMY_DATABASE_URI']='sqlite:///databse.sb'
db=SQLAlchemy(server)
@server.route("/") #root dorectory locations


def home(): #test for now sending to homepage HTML.... TO swap with Get or another method
    return"<h1>hello</h1>"






