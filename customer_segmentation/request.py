import requests

url = "http://127.0.0.1:8000/predict"

data = [
    [9, 110, "USA/Canada/As", "Google"],
    [29.0, 55, "West_EU", "Friend"]
]

response = requests.post(url, json=data)
print(response.json())