import os
import cv2
import numpy as np

# Save the model as '.yml' instead of '.pkl' for OpenCV recognizer
MODEL_PATH = "model.yml"

# Cache detectors and models in memory to prevent slow disk reads per-frame
_face_cascade = None
_lbph_model = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade

def crop_face_and_embed(bgr_image, bbox):
    x, y, w, h = bbox
    # Add padding around face for better context
    padding = int(0.2 * min(w, h))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding
    
    # Ensure bounds are within image
    img_h, img_w = bgr_image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    face = bgr_image[y:y+h, x:x+w]
    if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
        return None
        
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Apply denoising to handle blur better
    face = cv2.fastNlMeansDenoising(face, None, 10, 7, 21)
    # Increase resolution for better features
    face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)
    # Apply histogram equalization for better lighting normalization
    face = cv2.equalizeHist(face)
    
    return face

def get_face_bbox(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = get_face_cascade()
    # Relaxed detection parameters to work with blur images
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) > 0:
        # Return the largest face detected
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return largest_face  # x, y, w, h
    return None

def extract_embedding_for_image(stream_or_bytes):
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    bbox = get_face_bbox(img)
    if bbox is None:
        return None
    return crop_face_and_embed(img, bbox)

def load_model_if_exists(force_reload=False):
    global _lbph_model
    if not os.path.exists(MODEL_PATH):
        return None
    if _lbph_model is None or force_reload:
        try:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.read(MODEL_PATH)
            _lbph_model = clf
        except Exception as e:
            print("Error loading LBPH model:", e)
            return None
    return _lbph_model

def predict_with_model(clf, face_img):
    label, confidence_distance = clf.predict(face_img)
    # LBPH returns distance (0 is perfect, > 80 is bad)
    # Convert to 0.0 - 1.0 confidence format expected by app.py
    conf = max(0.0, 1.0 - (confidence_distance / 200.0))
    return str(label), conf

def train_model_background(dataset_dir, progress_callback=None):
    faces = []
    labels = []
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_students = max(1, len(student_dirs))
    processed = 0

    for sid in student_dirs:
        try:
            label_int = int(sid)
        except ValueError:
            continue
            
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None: continue
            
            bbox = get_face_bbox(img)
            if bbox is None: continue
            
            face = crop_face_and_embed(img, bbox)
            if face is None: continue
            
            faces.append(face)
            labels.append(label_int)
            
        processed += 1
        if progress_callback:
            pct = int((processed/total_students)*80)
            progress_callback(pct, f"Processed {processed}/{total_students} students")

    if len(faces) == 0:
        if progress_callback: progress_callback(0, "No training data found")
        return

    labels = np.array(labels)

    if progress_callback: progress_callback(85, "Training LBPH Face Recognizer...")
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, labels)

    clf.write(MODEL_PATH)
    
    # Update global cache immediately so it doesn't use the old model
    global _lbph_model
    _lbph_model = clf

    if progress_callback: progress_callback(100, "Training complete")
