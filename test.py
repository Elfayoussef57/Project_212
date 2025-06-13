import requests

url = "http://localhost:5000/api/analyze-scan"
file_path = '../Downloads/WhatsApp Image 2025-06-03 à 11.02.08_0ebe368b.jpg' 

with open(file_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")

try:
    json_data = response.json()
    print(json_data)
except Exception as e:
    print("❌ Erreur lors du décodage JSON:", e)
