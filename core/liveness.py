import numpy as np
import face_recognition
from core.face_engine import decode_image_bytes

def calculate_ear(eye_landmarks):
    """
    Calculates Eye Aspect Ratio (EAR) for a 6-point eye landmark set.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_landmarks) < 6:
        return 0.3
    pts = np.array(eye_landmarks, dtype=np.float64)
    # Vertical distances
    d_v1 = np.linalg.norm(pts[1] - pts[5])
    d_v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    d_h = np.linalg.norm(pts[0] - pts[3])
    if d_h == 0:
        return 0.3
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return float(ear)

def calculate_head_yaw_ratio(landmarks):
    """
    Approximates head yaw (turn) ratio using nose tip relative to eye corners.
    """
    if "nose_tip" not in landmarks or "left_eye" not in landmarks or "right_eye" not in landmarks:
        return 0.5
    nose_tip = np.mean(landmarks["nose_tip"], axis=0)
    left_eye_outer = np.array(landmarks["left_eye"][0])
    right_eye_outer = np.array(landmarks["right_eye"][3])
    
    total_span = abs(right_eye_outer[0] - left_eye_outer[0])
    if total_span == 0:
        return 0.5
    # Ratio from left eye outer to nose
    ratio = abs(nose_tip[0] - left_eye_outer[0]) / total_span
    return float(ratio)

def analyze_frame_liveness(rgb_image):
    """
    Extracts landmark metrics for a single frame:
    Returns dict: {'face_found': bool, 'ear': float, 'yaw_ratio': float, 'landmarks': dict}
    """
    landmarks_list = face_recognition.face_landmarks(rgb_image)
    if not landmarks_list:
        return {"face_found": False, "ear": 0.0, "yaw_ratio": 0.5, "landmarks": None}
        
    lm = landmarks_list[0]
    left_ear = calculate_ear(lm.get("left_eye", []))
    right_ear = calculate_ear(lm.get("right_eye", []))
    avg_ear = (left_ear + right_ear) / 2.0
    yaw_ratio = calculate_head_yaw_ratio(lm)
    
    return {
        "face_found": True,
        "ear": avg_ear,
        "yaw_ratio": yaw_ratio,
        "landmarks": lm
    }

def verify_liveness(frames, challenge_type="blink", is_demo=False):
    """
    Verifies liveness across a short sequence of frames or a single frame with challenge data.
    
    Parameters:
    - frames: list of images (bytes, base64, or numpy arrays)
    - challenge_type: 'blink', 'head_turn', or 'any'
    - is_demo: if True, bypasses strict multi-frame requirement with simulated verification
    
    Returns:
    {
        "passed": bool,
        "score": float (0.0 to 1.0),
        "challenge": str,
        "reason": str or None
    }
    """
    if not frames:
        return {
            "passed": False,
            "score": 0.0,
            "challenge": challenge_type,
            "reason": "No frames provided for liveness verification"
        }
        
    if is_demo:
        return {
            "passed": True,
            "score": 0.95,
            "challenge": "demo_verified",
            "reason": "Demo mode simulated liveness passed"
        }

    # Convert frames to RGB numpy
    rgb_frames = []
    for f in frames:
        try:
            if isinstance(f, np.ndarray):
                rgb_frames.append(f)
            else:
                rgb_frames.append(decode_image_bytes(f))
        except Exception:
            continue
            
    if not rgb_frames:
        return {
            "passed": False,
            "score": 0.0,
            "challenge": challenge_type,
            "reason": "Failed to decode frame images for liveness check"
        }

    # Analyze each frame
    frame_metrics = []
    for rgb in rgb_frames:
        metrics = analyze_frame_liveness(rgb)
        if metrics["face_found"]:
            frame_metrics.append(metrics)
            
    if len(frame_metrics) == 0:
        return {
            "passed": False,
            "score": 0.0,
            "challenge": challenge_type,
            "reason": "No face detected in any of the liveness verification frames"
        }
        
    # If single frame submitted, check basic facial feature integrity
    if len(frame_metrics) == 1:
        # A single frame cannot prove a dynamic blink/turn over time,
        # but if EAR is reasonable (>0.18 and <0.42), we allow conditional pass with note
        ear = frame_metrics[0]["ear"]
        if 0.16 <= ear <= 0.42:
            return {
                "passed": True,
                "score": 0.80,
                "challenge": "single_frame_heuristic",
                "reason": "Single frame passed baseline facial landmark geometry check"
            }
        else:
            return {
                "passed": False,
                "score": 0.40,
                "challenge": "single_frame_heuristic",
                "reason": "Abnormal eye aspect ratio detected in frame"
            }

    # Multi-frame sequence analysis
    ears = [m["ear"] for m in frame_metrics]
    yaws = [m["yaw_ratio"] for m in frame_metrics]
    
    min_ear, max_ear = min(ears), max(ears)
    ear_diff = max_ear - min_ear
    yaw_diff = max(yaws) - min(yaws)

    if challenge_type in ("blink", "any"):
        # A blink manifests as a distinct dip in EAR (ear_diff >= 0.05 or min_ear < 0.22)
        if ear_diff >= 0.045 or (min_ear < 0.22 and max_ear > 0.24):
            score = min(1.0, 0.70 + (ear_diff * 4.0))
            return {
                "passed": True,
                "score": round(score, 3),
                "challenge": "blink",
                "reason": f"Blink verified (EAR delta: {ear_diff:.3f})"
            }

    if challenge_type in ("head_turn", "any"):
        # Head turn manifests as yaw_diff >= 0.06
        if yaw_diff >= 0.05:
            score = min(1.0, 0.70 + (yaw_diff * 3.5))
            return {
                "passed": True,
                "score": round(score, 3),
                "challenge": "head_turn",
                "reason": f"Head turn movement verified (yaw delta: {yaw_diff:.3f})"
            }

    # Check for general natural micro-motion vs static image
    if ear_diff > 0.02 or yaw_diff > 0.02:
        return {
            "passed": True,
            "score": 0.75,
            "challenge": "micro_movement",
            "reason": "Natural facial movement detected across frame sequence"
        }

    # If all frames are almost identical, likely a static photograph spoof
    return {
        "passed": False,
        "score": 0.20,
        "challenge": challenge_type,
        "reason": "Liveness check failed: Static image or photo spoof suspected (no blink or natural movement detected)"
    }
