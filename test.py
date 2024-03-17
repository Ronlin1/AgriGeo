#Include the requests library.
import requests

# Replace with the actual URL where your Flask app is running
url = 'http://127.0.0.1:5000/predict'

# Sample JSON data
data = {
    'year': 2022,
    'month': 3,
    'ndvi_mean': 0.75,
    'rain_mean': 100.0,
    'et_mean': 5.0,
    'acled_count': 20,
    'p_staple_food': 90.0,
    'area': 5000.0,
    'cropland_pct': 30.0,
    'pop': 100000,
    'ruggedness_mean': 10.0,
    'pasture_pct': 10.0
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
