#Import Modules where they are coded. Reason For Split to manage time and tasks
from Model import API
from Model import create_app

server= create_app()

if __name__=='__main__':
    server.run(debug=True)




