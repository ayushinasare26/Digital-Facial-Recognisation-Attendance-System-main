import os
import io
import json
import base64
import numpy as np
from PIL import Image
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    FACE_ENGINE_ERROR = None
except Exception as _fe_err:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False
    FACE_ENGINE_ERROR = str(_fe_err)

from config import Config
from core.db import get_db_connection, get_setting

# In-memory cache for fast matching: list of dicts: {'student_id': int, 'name': str, 'vector': np.ndarray}
_EMBEDDINGS_CACHE = None

def decode_image_bytes(image_data):
    """
    Decodes an image from bytes, BytesIO, or base64 string into an RGB numpy array.
    """
    if isinstance(image_data, str):
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        data = base64.b64decode(image_data)
    elif hasattr(image_data, "read"):
        data = image_data.read()
    elif isinstance(image_data, bytes):
        data = image_data
    else:
        raise ValueError("Unsupported image data type")
        
    pil_image = Image.open(io.BytesIO(data))
    # Convert RGBA or grayscale to RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return np.array(pil_image)

def detect_face_and_embedding(rgb_image, min_face_size=100):
    """
    Detects face in the image and extracts 128-dimensional deep ResNet embedding.
    Validates face resolution and framing.
    Returns: (face_location, embedding, error_message)
    """
    if rgb_image is None or rgb_image.size == 0:
        return None, None, "Invalid image data provided"
        
    if not FACE_RECOGNITION_AVAILABLE or face_recognition is None:
        return None, None, f"Biometric engine unavailable on this server: {FACE_ENGINE_ERROR}"
        
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    if len(face_locations) == 0:
        return None, None, "No face detected in the image"
    if len(face_locations) > 1:
        return None, None, f"Multiple faces ({len(face_locations)}) detected. Only one person must be in frame."
        
    location = face_locations[0]
    top, right, bottom, left = location
    face_h = bottom - top
    face_w = right - left
    
    if face_h < min_face_size or face_w < min_face_size:
        return None, None, f"Face is too far from camera ({face_w}x{face_h} px, required >= {min_face_size}px). Please move closer so your face fills the guide oval."
        
    encodings = face_recognition.face_encodings(rgb_image, known_face_locations=[location])
    
    if len(encodings) == 0:
        return None, None, "Failed to extract face embedding from detected face"
        
    embedding = encodings[0]
    return location, embedding, None

def load_embeddings_cache(force_reload=False):
    """
    Loads all enrolled embeddings from the database into the fast in-memory cache.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None and not force_reload:
        return _EMBEDDINGS_CACHE
        
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT e.student_id, s.name, e.embedding
        FROM embeddings e
        JOIN students s ON e.student_id = s.id
    """)
    rows = c.fetchall()
    conn.close()
    
    cache = []
    for r in rows:
        try:
            vec_list = json.loads(r["embedding"])
            vec_np = np.array(vec_list, dtype=np.float64)
            cache.append({
                "student_id": r["student_id"],
                "name": r["name"],
                "vector": vec_np
            })
        except Exception:
            continue
            
    _EMBEDDINGS_CACHE = cache
    return _EMBEDDINGS_CACHE

def match_face_embedding(target_embedding, match_threshold=None, review_threshold=None):
    """
    Matches target embedding against all enrolled student embeddings using
    multi-template consensus and margin verification.
    """
    if not FACE_RECOGNITION_AVAILABLE or face_recognition is None:
        return {
            "matched": False,
            "student_id": None,
            "name": None,
            "confidence": 0.0,
            "distance": 1.0,
            "status": "failed",
            "reason": f"Biometric engine unavailable: {FACE_ENGINE_ERROR}"
        }
        
    cache = load_embeddings_cache()
    if not cache:
        return {
            "matched": False,
            "student_id": None,
            "name": None,
            "confidence": 0.0,
            "distance": 1.0,
            "status": "failed",
            "reason": "No enrolled face templates found in the database. Please enroll students first."
        }
        
    if match_threshold is None:
        try:
            match_threshold = float(get_setting("match_threshold", Config.MATCH_THRESHOLD))
        except:
            match_threshold = Config.MATCH_THRESHOLD
            
    if review_threshold is None:
        try:
            review_threshold = float(get_setting("review_threshold", Config.REVIEW_THRESHOLD))
        except:
            review_threshold = Config.REVIEW_THRESHOLD

    # Group distance calculations by student
    student_scores = {}
    for item in cache:
        sid = item["student_id"]
        if sid not in student_scores:
            student_scores[sid] = {"student_id": sid, "name": item["name"], "distances": []}

    known_vectors = [item["vector"] for item in cache]
    distances = face_recognition.face_distance(known_vectors, target_embedding)
    
    for item, dist in zip(cache, distances):
        student_scores[item["student_id"]]["distances"].append(float(dist))

    # Rank students by aggregate consensus score (combining minimum distance and top-2 mean)
    ranked_students = []
    for sid, sdata in student_scores.items():
        dists = sorted(sdata["distances"])
        min_d = dists[0]
        # Top-2 consensus distance
        top2_avg = (dists[0] + dists[1]) / 2.0 if len(dists) >= 2 else min_d
        agg_dist = (min_d * 0.7) + (top2_avg * 0.3)
        ranked_students.append({
            "student_id": sid,
            "name": sdata["name"],
            "best_distance": min_d,
            "aggregate_distance": agg_dist
        })

    ranked_students.sort(key=lambda x: x["aggregate_distance"])
    best = ranked_students[0]
    second_best = ranked_students[1] if len(ranked_students) > 1 else None

    best_distance = best["best_distance"]
    agg_distance = best["aggregate_distance"]
    margin = (second_best["aggregate_distance"] - best["aggregate_distance"]) if second_best else 1.0

    # Calibrated confidence score (0.0 to 1.0)
    # dist 0.25 -> 88%
    # dist 0.30 -> 84%
    # dist 0.40 -> 75%
    # dist 0.48 -> 67%
    confidence = round(max(0.50, min(0.99, 1.0 - float(best_distance ** 1.5))), 4)

    # Reject if best distance exceeds strict threshold (0.48)
    if best_distance > match_threshold:
        return {
            "matched": False,
            "student_id": None,
            "name": None,
            "confidence": confidence,
            "distance": round(best_distance, 4),
            "status": "failed",
            "reason": f"Face does not match any enrolled student (closest distance: {best_distance:.2f}, required: <={match_threshold:.2f}). Please face camera directly in good lighting."
        }

    # Ambiguity check: only apply if match is not already very close (< 0.38)
    if second_best and margin < 0.04 and best_distance > 0.40:
        return {
            "matched": False,
            "student_id": None,
            "name": None,
            "confidence": confidence,
            "distance": round(best_distance, 4),
            "status": "failed",
            "reason": f"Ambiguous face match between {best['name']} and {second_best['name']} (margin too close: {margin:.3f}). Please move closer and align your face."
        }

    # If distance is <= 0.38, it is a very strong verified match
    if best_distance <= 0.38:
        status = "success"
        reason = None
    elif best_distance > review_threshold or margin < 0.05:
        status = "flagged"
        reason = f"Borderline match confidence ({round(confidence*100, 1)}%) — flagged for admin review"
    else:
        status = "success"
        reason = None

    return {
        "matched": True,
        "student_id": best["student_id"],
        "name": best["name"],
        "confidence": confidence,
        "distance": round(best_distance, 4),
        "status": status,
        "reason": reason
    }

def enroll_student_photo(student_id, image_bytes_or_np):
    """
    Extracts embedding from photo and stores in embeddings table.
    Returns: (success: bool, error_message: str)
    """
    if isinstance(image_bytes_or_np, (bytes, str)) or hasattr(image_bytes_or_np, "read"):
        rgb = decode_image_bytes(image_bytes_or_np)
    else:
        rgb = image_bytes_or_np
        
    location, embedding, err = detect_face_and_embedding(rgb)
    if err:
        return False, err
        
    vec_json = json.dumps(embedding.tolist())
    conn = get_db_connection()
    c = conn.cursor()
    import datetime
    now = datetime.datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO embeddings (student_id, embedding, created_at)
        VALUES (?, ?, ?)
    """, (student_id, vec_json, now))
    conn.commit()
    conn.close()
    
    # Invalidate cache
    load_embeddings_cache(force_reload=True)
    return True, None

def enroll_existing_dataset(dataset_dir=None):
    """
    Helper to index all photos in dataset/<student_id> into the embeddings table.
    """
    dataset_dir = dataset_dir or Config.DATASET_DIR
    if not os.path.exists(dataset_dir):
        return 0
        
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM students")
    valid_student_ids = {r["id"] for r in c.fetchall()}
    conn.close()
    
    enrolled_count = 0
    for sid_str in os.listdir(dataset_dir):
        try:
            sid = int(sid_str)
        except ValueError:
            continue
        if sid not in valid_student_ids:
            continue
            
        student_folder = os.path.join(dataset_dir, sid_str)
        if not os.path.isdir(student_folder):
            continue
            
        # Check if already has embeddings
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as count FROM embeddings WHERE student_id=?", (sid,))
        existing = c.fetchone()["count"]
        conn.close()
        
        if existing > 0:
            continue
            
        # Extract from first few clear photos
        image_files = [f for f in os.listdir(student_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        success_for_student = 0
        for img_name in image_files[:5]: # up to 5 reference photos per student
            img_path = os.path.join(student_folder, img_name)
            try:
                img_rgb = face_recognition.load_image_file(img_path)
                loc, emb, err = detect_face_and_embedding(img_rgb)
                if emb is not None:
                    vec_json = json.dumps(emb.tolist())
                    conn = get_db_connection()
                    c = conn.cursor()
                    import datetime
                    c.execute("""
                        INSERT INTO embeddings (student_id, embedding, created_at)
                        VALUES (?, ?, ?)
                    """, (sid, vec_json, datetime.datetime.utcnow().isoformat()))
                    conn.commit()
                    conn.close()
                    success_for_student += 1
            except Exception as e:
                continue
                
        if success_for_student > 0:
            enrolled_count += 1
            
    load_embeddings_cache(force_reload=True)
    return enrolled_count
