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

try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

def get_app_timezone():
    """
    Returns the target application timezone.
    Configurable via APP_TIMEZONE or TIMEZONE environment variables.
    Defaults to 'Asia/Kolkata' (IST, UTC+5:30) for Mumbai Enterprise site.
    """
    tz_str = os.environ.get("APP_TIMEZONE") or os.environ.get("TIMEZONE") or "Asia/Kolkata"
    try:
        return zoneinfo.ZoneInfo(tz_str)
    except Exception:
        try:
            return datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        except Exception:
            return datetime.timezone.utc

def get_utc_now():
    """Returns current timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

def get_utc_iso():
    """Returns current UTC ISO-formatted string."""
    return get_utc_now().isoformat()

def get_local_now():
    """Returns current real local datetime in the configured application timezone."""
    return datetime.datetime.now(get_app_timezone())

def format_local_timestamp(iso_val, include_year=True):
    """
    Converts any stored UTC ISO string or timestamp into the actual real
    local date and time in the configured application timezone (e.g. 'Sep 30, 2026 - 11:16 AM').
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

        app_tz = get_app_timezone()
        local_dt = dt.astimezone(app_tz)
        if include_year:
            return local_dt.strftime("%b %d, %Y - %I:%M %p")
        return local_dt.strftime("%b %d - %I:%M %p")
    except Exception:
        return str(iso_val)

def format_local_date(iso_val):
    """
    Converts any stored UTC ISO string or timestamp into YYYY-MM-DD date
    in the configured application timezone.
    """
    if not iso_val:
        return ""
    try:
        if isinstance(iso_val, str):
            clean_str = iso_val.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(clean_str)
        elif isinstance(iso_val, datetime.datetime):
            dt = iso_val
        else:
            return str(iso_val)[:10]

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)

        app_tz = get_app_timezone()
        return dt.astimezone(app_tz).strftime("%Y-%m-%d")
    except Exception:
        return str(iso_val)[:10]

def is_dir_writable(path):
    try:
        testfile = os.path.join(path, ".writetest")
        with open(testfile, "w") as f:
            f.write("1")
        os.remove(testfile)
        return True
    except Exception:
        return False

class Config:
    BASE_DIR = BASE_DIR
    IS_VERCEL = bool(
        os.environ.get("VERCEL")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or not is_dir_writable(str(BASE_DIR))
    )
    
    if IS_VERCEL:
        tmp_db = "/tmp/attendance.db"
        if not os.path.exists(tmp_db):
            for candidate in [BASE_DIR / "attendance.db", Path("/var/task/attendance.db")]:
                if candidate.exists():
                    import shutil
                    try:
                        shutil.copyfile(str(candidate), tmp_db)
                        break
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
