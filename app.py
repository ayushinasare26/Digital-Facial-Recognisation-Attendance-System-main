"""
app.py - Geo-Verified Facial Recognition Attendance System for Enterprise Workforce Management
Three-Portal Architecture:
  - Employee Portal (/employee/*) for mobile-first Check-In / Check-Out and personal attendance history
  - Manager Portal (/manager/*) for team-scoped supervision and correction request approvals
  - Admin Portal (/admin/*) for multi-site governance, shift rules, compliance, and payroll exports
"""

import os
from flask import (
    Flask, redirect, url_for, session, send_file, abort
)
from config import Config, format_local_timestamp
from core.db import init_db
from core.face_engine import load_embeddings_cache
from blueprints.employee import employee_bp
from blueprints.manager import manager_bp
from blueprints.admin import admin_bp

# Initialize Flask Application
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = Config.SECRET_KEY

@app.template_filter("local_time")
def local_time_filter(val):
    return format_local_timestamp(val)

# Ensure database schema is migrated with WAL mode, roles, sites, shifts, and indices
STARTUP_WARNING = None
try:
    init_db()
except Exception as _e:
    import traceback
    STARTUP_WARNING = f"init_db warning: {traceback.format_exc()}"

# Warm up biometric facial embeddings cache into memory
try:
    load_embeddings_cache()
except Exception as _e:
    import traceback
    if not STARTUP_WARNING:
        STARTUP_WARNING = f"embeddings warning: {traceback.format_exc()}"

# Register Blueprints
app.register_blueprint(employee_bp, url_prefix="/employee")
app.register_blueprint(manager_bp, url_prefix="/manager")
app.register_blueprint(admin_bp, url_prefix="/admin")

# ==========================================
# Root & Entry Redirection
# ==========================================
@app.route("/")
def index():
    """
    Intelligent root router:
    - Admin session -> Admin Dashboard
    - Manager session -> Manager Team Dashboard
    - Employee session -> Employee Check-In/Check-Out Camera
    - Unauthenticated -> Employee Portal Login
    """
    role = session.get("role")
    if role == "admin" and session.get("admin_id"):
        return redirect(url_for("admin.dashboard"))
    elif role == "manager" and (session.get("manager_id") or session.get("employee_id")):
        return redirect(url_for("manager.dashboard"))
    elif role in ("employee", "user") and (session.get("employee_id") or session.get("student_id")):
        return redirect(url_for("employee.check_in_out"))
    return redirect(url_for("employee.login"))

@app.route("/login", methods=["GET", "POST"])
def root_login():
    """Default entry point routes to Employee Portal login."""
    from blueprints.employee.routes import login as employee_login
    return employee_login()

@app.route("/api/health")
def api_health():
    """Healthcheck endpoint for deployment verification and runtime diagnostics."""
    import sys
    from core.face_engine import FACE_RECOGNITION_AVAILABLE
    return {
        "status": "healthy" if not STARTUP_WARNING else "degraded",
        "python_version": sys.version,
        "is_vercel": Config.IS_VERCEL,
        "db_path": Config.DB_PATH,
        "db_exists": os.path.exists(Config.DB_PATH),
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "startup_warning": STARTUP_WARNING,
    }

# ==========================================
# Protected Media Serving Routes (Compliance & Scoping)
# ==========================================
def get_latest_employee_image_path(eid: int):
    """Returns the most recent image path for a given employee ID."""
    folder = os.path.join(Config.DATASET_DIR, str(eid))
    if not os.path.isdir(folder):
        return None
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, files[0])

@app.route("/attendance_photos/<path:filename>")
def serve_attendance_photo(filename):
    """
    Serves geotagged watermarked proof-of-attendance photos.
    Compliance: Raw attendance photos are restricted to Admins and the specific Employee themselves.
    Managers view generalized attendance logs without raw biometric photos to protect privacy.
    """
    role = session.get("role")
    session_emp_id = session.get("employee_id") or session.get("student_id")
    
    # If filename starts with <employee_id>_
    parts = filename.split("_", 1)
    try:
        photo_emp_id = int(parts[0])
    except ValueError:
        photo_emp_id = None

    if role != "admin" and (not session_emp_id or session_emp_id != photo_emp_id):
        return abort(403)

    photo_path = os.path.join(Config.ATTENDANCE_PHOTOS_DIR, filename)
    if not os.path.isfile(photo_path) and photo_emp_id is not None:
        import difflib
        candidates = [f for f in os.listdir(Config.ATTENDANCE_PHOTOS_DIR) if f.startswith(f"{photo_emp_id}_")]
        if candidates:
            closest = difflib.get_close_matches(filename, candidates, n=1, cutoff=0.6)
            if closest:
                photo_path = os.path.join(Config.ATTENDANCE_PHOTOS_DIR, closest[0])
            else:
                candidates.sort(key=lambda f: os.path.getmtime(os.path.join(Config.ATTENDANCE_PHOTOS_DIR, f)), reverse=True)
                photo_path = os.path.join(Config.ATTENDANCE_PHOTOS_DIR, candidates[0])

    if os.path.isfile(photo_path):
        return send_file(photo_path)
    return abort(404)

@app.route("/student_image/<int:sid>")
@app.route("/employee_image/<int:sid>")
def employee_image(sid):
    """
    Serves enrolled reference portrait.
    Authorization: Accessible only to authenticated admins or the employee themselves.
    """
    role = session.get("role")
    session_sid = session.get("employee_id") or session.get("student_id")
    if role != "admin" and session_sid != sid:
        return abort(403)

    img_path = get_latest_employee_image_path(sid)
    if img_path and os.path.isfile(img_path):
        return send_file(img_path)
    return abort(404)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)