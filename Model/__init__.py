from flask import Flask
def create_app():
    server=Flask(__name__)
    server.config['SECRET_KEY']='DJFHKSJD KDSJFAK'
    #Importing files to render on website
    #registering the  pages to display
    return server