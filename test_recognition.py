import os
import requests
from model import train_model_background

print("Training model locally...")
def cb(p, m):
    print(f"{p}% - {m}")

try:
    train_model_background("dataset", cb)
    print("Training finished.")
except Exception as e:
    print("Training error:", e)

url = 'http://127.0.0.1:5000/recognize_face'
image_path = 'dataset/1/1773015483.318820_0.jpg'
if os.path.exists(image_path):
    print(f"Testing recognition with {image_path}")
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(url, files=files)
            print("Response:", response.json())
        except Exception as e:
            print("API error:", e)
else:
    print("Image not found:", image_path)
