import os
import io
import json
import random
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Config
from core.db import get_db_connection, init_db
from core.face_engine import enroll_existing_dataset, load_embeddings_cache
from core.geotag import stamp_geotag

# Fictional enrolled demo profiles
DEMO_STUDENTS = [
    {"name": "Aarav Sharma (Demo)", "roll": "DEMO-101", "class": "B.Tech CSE", "section": "A", "reg_no": "REG202601"},
    {"name": "Zara Patel (Demo)", "roll": "DEMO-102", "class": "B.Tech IT", "section": "B", "reg_no": "REG202602"},
    {"name": "Rohan Deshmukh (Demo)", "roll": "DEMO-103", "class": "B.Tech AI-DS", "section": "A", "reg_no": "REG202603"},
    {"name": "Ananya Roy (Demo)", "roll": "DEMO-104", "class": "B.Tech CSE", "section": "C", "reg_no": "REG202604"},
    {"name": "Kabir Mehta (Demo)", "roll": "DEMO-105", "class": "B.Tech ECE", "section": "A", "reg_no": "REG202605"},
    {"name": "Sneha Kulkarni (Demo)", "roll": "DEMO-106", "class": "B.Tech IT", "section": "A", "reg_no": "REG202606"},
    {"name": "Aditya Verma (Demo)", "roll": "DEMO-107", "class": "B.Tech CSE", "section": "B", "reg_no": "REG202607"},
    {"name": "Pooja Hegde (Demo)", "roll": "DEMO-108", "class": "B.Tech AI-DS", "section": "B", "reg_no": "REG202608"},
    {"name": "Vikram Malhotra (Demo)", "roll": "DEMO-109", "class": "B.Tech CSE", "section": "A", "reg_no": "REG202609"},
    {"name": "Meera Iyer (Demo)", "roll": "DEMO-110", "class": "B.Tech IT", "section": "C", "reg_no": "REG202610"},
]

DEMO_LOCATIONS = [
    (19.0760, 72.8777, "Nariman Point, Marine Drive, Mumbai, Maharashtra, India"),
    (19.1136, 72.8697, "Andheri East, MIDC Tech Park, Mumbai, Maharashtra, India"),
    (28.6139, 77.2090, "Connaught Place, Central Delhi, New Delhi, India"),
    (28.5355, 77.3910, "Sector 62, Electronic City, Noida, Uttar Pradesh, India"),
    (12.9716, 77.5946, "MG Road, Indiranagar, Bengaluru, Karnataka, India"),
    (12.9279, 77.6271, "Koramangala 4th Block, Bengaluru, Karnataka, India"),
    (18.5204, 73.8567, "Shivaji Nagar, FC Road, Pune, Maharashtra, India"),
    (18.5590, 73.7868, "Baner Cyber City, Pune, Maharashtra, India"),
    (17.3850, 78.4867, "HITEC City, Madhapur, Hyderabad, Telangana, India"),
    (51.5074, -0.1278, "Trafalgar Square, Westminster, London, United Kingdom"),
    (40.7128, -74.0060, "Broadway, Lower Manhattan, New York, NY, USA"),
]

def generate_avatar_image(name, width=480, height=480):
    """Generates an aesthetic synthetic avatar photo for demo records."""
    img = Image.new("RGB", (width, height), color=(240, 244, 250))
    draw = ImageDraw.Draw(img)
    
    # Gradient or geometric aesthetic background
    palette = [
        ((37, 99, 235), (99, 102, 241)),
        ((13, 148, 136), (16, 185, 129)),
        ((124, 58, 237), (217, 70, 239)),
        ((225, 29, 72), (244, 63, 94))
    ]
    color_pair = random.choice(palette)
    
    # Draw decorative circles
    draw.ellipse([(-50, -50), (width + 50, height + 50)], fill=(245, 248, 255))
    draw.ellipse([(width//2 - 120, height//2 - 140), (width//2 + 120, height//2 + 100)], fill=color_pair[0])
    
    # Shoulders
    draw.ellipse([(width//2 - 160, height//2 + 40), (width//2 + 160, height + 100)], fill=color_pair[1])
    
    # Face outline
    draw.ellipse([(width//2 - 80, height//2 - 110), (width//2 + 80, height//2 + 50)], fill=(254, 215, 170))
    
    # Hair
    draw.arc([(width//2 - 85, height//2 - 120), (width//2 + 85, height//2 - 30)], 180, 360, fill=(30, 41, 59), width=24)
    
    # Eyes
    draw.ellipse([(width//2 - 40, height//2 - 40), (width//2 - 25, height//2 - 25)], fill=(30, 41, 59))
    draw.ellipse([(width//2 + 25, height//2 - 40), (width//2 + 40, height//2 - 25)], fill=(30, 41, 59))
    
    # Smile
    draw.arc([(width//2 - 30, height//2 - 10), (width//2 + 30, height//2 + 25)], 0, 180, fill=(185, 28, 28), width=4)
    
    # Stamp "SYNTHETIC DEMO AVATAR" watermark tag
    try:
        font = ImageFont.truetype("arialbd.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((20, 20), f"DEMO PROFILE: {name}", fill=(100, 116, 139), font=font)
    
    return img

def seed_demo_data():
    """Seeds enrolled demo students, embeddings, and 30+ attendance records."""
    init_db()
    conn = get_db_connection()
    c = conn.cursor()
    
    print("1. Indexing real dataset photos if not already indexed...")
    enroll_existing_dataset()

    print("2. Seeding fictional demo students...")
    now = datetime.datetime.utcnow().isoformat()
    created_student_ids = []
    
    for s in DEMO_STUDENTS:
        c.execute("SELECT id FROM students WHERE roll=?", (s["roll"],))
        existing = c.fetchone()
        if existing:
            sid = existing["id"]
        else:
            c.execute("""
                INSERT INTO students (name, roll, class, section, reg_no, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (s["name"], s["roll"], s["class"], s["section"], s["reg_no"], now))
            sid = c.lastrowid
            
            # Create synthetic 128-d normalized embedding vector
            # (Orthogonal unit vector so they don't collide)
            vec = np.random.randn(128)
            vec /= np.linalg.norm(vec)
            vec_json = json.dumps(vec.tolist())
            c.execute("""
                INSERT INTO embeddings (student_id, embedding, created_at)
                VALUES (?, ?, ?)
            """, (sid, vec_json, now))
            
            # Save a sample image in dataset/<sid>/
            student_folder = os.path.join(Config.DATASET_DIR, str(sid))
            os.makedirs(student_folder, exist_ok=True)
            avatar = generate_avatar_image(s["name"])
            avatar.save(os.path.join(student_folder, "demo_ref_1.jpg"), "JPEG")
            
        created_student_ids.append((sid, s["name"]))
    
    conn.commit()

    # Also grab real enrolled students
    c.execute("SELECT id, name FROM students")
    all_students = [(r["id"], r["name"]) for r in c.fetchall()]

    print("3. Seeding 35+ realistic attendance records with geotags & pipeline logs...")
    c.execute("SELECT COUNT(*) as cnt FROM attendance")
    existing_attendance_cnt = c.fetchone()["cnt"]
    
    if existing_attendance_cnt < 30:
        base_time = datetime.datetime.utcnow()
        for i in range(35):
            student_id, student_name = random.choice(all_students)
            # Spread across last 14 days
            days_ago = random.randint(0, 14)
            hours_ago = random.randint(1, 10)
            mins_ago = random.randint(0, 59)
            rec_dt = base_time - datetime.timedelta(days=days_ago, hours=hours_ago, minutes=mins_ago)
            rec_ts = rec_dt.isoformat()
            ts_display = rec_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

            # 80% success, 15% flagged for review (low confidence or location issue), 5% location denied
            roll = random.random()
            if roll < 0.78:
                status = "success"
                confidence = round(random.uniform(0.85, 0.98), 4)
                lat, lon, address = random.choice(DEMO_LOCATIONS)
                liveness_passed = 1
            elif roll < 0.90:
                status = "flagged"
                confidence = round(random.uniform(0.56, 0.63), 4) # Borderline confidence
                lat, lon, address = random.choice(DEMO_LOCATIONS)
                liveness_passed = 1
            else:
                status = "flagged"
                confidence = round(random.uniform(0.88, 0.96), 4)
                lat, lon = None, None
                address = "Location Permission Denied by User"
                liveness_passed = 1

            # Generate synthetic geotagged photo
            avatar = generate_avatar_image(student_name)
            photo_path = stamp_geotag(
                image_input=avatar,
                student_name=student_name,
                student_id=student_id,
                timestamp_str=ts_display,
                latitude=lat,
                longitude=lon,
                address=address,
                confidence=confidence,
                status=status
            )

            c.execute("""
                INSERT INTO attendance (
                    student_id, name, timestamp, latitude, longitude,
                    address, confidence, liveness_passed, geotagged_photo_path, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (student_id, student_name, rec_ts, lat, lon, address, confidence, liveness_passed, photo_path, status))
            att_id = c.lastrowid

            # Seed 9 pipeline stages for this attendance record
            run_id = f"demo_pipe_{att_id}"
            stages = [
                ("Photo Received", "Completed", "Selfie payload validated (640x480 RGB)"),
                ("Liveness Check", "Completed", "Eye blink & head micro-movement verified (Score: 0.94)"),
                ("Face Detection", "Completed", "Single face located bounding box (top: 110, right: 380, bottom: 380, left: 110)"),
                ("Embedding Extraction", "Completed", "128-d deep ResNet embedding vector generated"),
                ("Embedding Match", "Completed", f"Matched against enrolled gallery with {round(confidence*100, 1)}% confidence"),
                ("Geolocation Capture", "Completed" if lat else "Failed", f"Coordinates acquired: {lat}, {lon}" if lat else "Client denied location permission"),
                ("Reverse Geocoding", "Completed" if lat else "Skipped", f"Resolved via OSM Nominatim: {address}" if lat else "Skipped due to missing GPS"),
                ("Geotag Stamping", "Completed", f"Geotag stamped into {photo_path}"),
                ("Record Saved", "Completed", f"Saved to database (ID #{att_id})")
            ]
            for stage_name, stage_status, msg in stages:
                c.execute("""
                    INSERT INTO pipeline_logs (attendance_id, run_id, stage, status, message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (att_id, run_id, stage_name, stage_status, msg, rec_ts))

        conn.commit()
        print(f"Seeded 35 attendance records successfully.")
    else:
        print(f"Database already has {existing_attendance_cnt} records.")

    conn.close()
    load_embeddings_cache(force_reload=True)
    print("Demo data seeding complete!")

if __name__ == "__main__":
    seed_demo_data()
