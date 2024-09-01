from flask import Flask

server = Flask(__name__) #Creating the server setting it to name
@server.route('/')
def home(): #test for now sending to homepage HTML.... TO swap with Get or another method
    return"<h1>hello World</h1>"





