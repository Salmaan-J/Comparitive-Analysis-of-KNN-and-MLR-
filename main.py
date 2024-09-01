#Import Modules where they are coded. Reason For Split to manage time and tasks
from Model import API

server  = API.server

if __name__=='__main__':
    server.run(debug=True)




