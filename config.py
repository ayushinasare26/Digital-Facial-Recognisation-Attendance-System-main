import os
import datetime
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def get_utc_now():
    """Returns current timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

def get_utc_iso():
    """Returns current UTC ISO-formatted string."""
    return get_utc_now().isoformat()

def get_local_now():
    """Returns current local datetime on the host system."""
    return datetime.datetime.now().astimezone()

def format_local_timestamp(iso_val, include_year=True):
    """
    Converts any stored UTC ISO string or timestamp into the actual real
    local date and time of the user/system (e.g. 'Sep 27, 2026, 11:10 PM').
    """
    if not iso_val:
        return "—"
    try:
        if isinstance(iso_val, str):
            clean_str = iso_val.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(clean_str)
        elif isinstance(iso_val, datetime.datetime):
            dt = iso_val
        else:
            return str(iso_val)

        if dt.tzinfo is None:
            # Assume stored naive timestamp was UTC
            dt = dt.replace(tzinfo=datetime.timezone.utc)

        local_dt = dt.astimezone()
        if include_year:
            return local_dt.strftime("%b %d, %Y - %I:%M %p")
        return local_dt.strftime("%b %d - %I:%M %p")
    except Exception:
        return str(iso_val)

class Config:
    BASE_DIR = BASE_DIR
    IS_VERCEL = bool(os.environ.get("VERCEL"))
    
    if IS_VERCEL:
        tmp_db = "/tmp/attendance.db"
        if not os.path.exists(tmp_db) and (BASE_DIR / "attendance.db").exists():
            import shutil
            try:
                shutil.copyfile(str(BASE_DIR / "attendance.db"), tmp_db)
            except Exception:
                pass
        DB_PATH = os.environ.get("DATABASE_PATH") or tmp_db
        DATASET_DIR = "/tmp/dataset"
        EMBEDDINGS_DIR = "/tmp/embeddings"
        ATTENDANCE_PHOTOS_DIR = "/tmp/attendance_photos"
    else:
        DB_PATH = os.environ.get("DATABASE_PATH") or str(BASE_DIR / "attendance.db")
        DATASET_DIR = str(BASE_DIR / "dataset")
        EMBEDDINGS_DIR = str(BASE_DIR / "embeddings")
        ATTENDANCE_PHOTOS_DIR = str(BASE_DIR / "attendance_photos")
    
    # Thresholds (dlib ResNet Euclidean distance)
    MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.48))
    REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", 0.44))
    DUPLICATE_COOLDOWN_SECONDS = int(os.environ.get("DUPLICATE_COOLDOWN_SECONDS", 300))
    
    # Geocoding
    GEOCODING_PROVIDER = os.environ.get("GEOCODING_PROVIDER", "nominatim")
    GEOCODING_API_KEY = os.environ.get("GEOCODING_API_KEY", "")
    NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", "GeoFaceAttendanceSystem/2.0")
    
    # Flask Session / Auth
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "geo-face-attendance-secret-2026")
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

for folder in [Config.DATASET_DIR, Config.EMBEDDINGS_DIR, Config.ATTENDANCE_PHOTOS_DIR]:
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        pass
