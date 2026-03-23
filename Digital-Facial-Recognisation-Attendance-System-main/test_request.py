import requests
import os

folder = "dataset/13"
if not os.path.exists(folder):
    print("No folder 13")
    exit()

files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
if not files:
    print("No images in 13")
    exit()

img_path = os.path.join(folder, files[0])

with open(img_path, 'rb') as f:
    r = requests.post("http://127.0.0.1:5000/recognize_face", files={"image": f})
    
print(r.status_code)
print(r.json())
