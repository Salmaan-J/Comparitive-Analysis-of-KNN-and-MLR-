
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/weather', methods=['GET'])
def get_weather_data():
    # Example data structure
    data = {
        "rainfall": 5.2,           # mm
        #"cloud_cover": 70,         # percentage
        #"longwave_radiation": 300, # W/m²
        #"wind_speed": 12.5,        # m/s
        #"wind_direction": 180       # degrees
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
