import os
import cv2
from model import extract_embedding_for_image, load_model_if_exists

def test_confidence():
    clf = load_model_if_exists()
    if not clf:
        print("No model found!")
        return

    dataset_dir = "dataset"
    for sid in os.listdir(dataset_dir):
        student_dir = os.path.join(dataset_dir, sid)
        if os.path.isdir(student_dir):
            files = [f for f in os.listdir(student_dir) if f.endswith('.jpg')]
            if files:
                img_path = os.path.join(student_dir, files[0])
                with open(img_path, 'rb') as f:
                    face_img = extract_embedding_for_image(f)
                    if face_img is not None:
                        label, dist = clf.predict(face_img)
                        print(f"File: {img_path} | True ID: {sid} | Predicted: {label} | Raw Distance: {dist}")

if __name__ == '__main__':
    test_confidence()
