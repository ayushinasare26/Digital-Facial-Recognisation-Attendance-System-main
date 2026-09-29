import os
import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config, get_utc_iso

def get_db_connection(db_path=None):
    """Returns a SQLite connection with Row factory and WAL mode enabled."""
    path = db_path or Config.DB_PATH
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn

def init_db(db_path=None):
    """
    Initializes database schema and executes migration-safe ALTER TABLE
    statements for the enterprise workforce domain model.
    """
    conn = get_db_connection(db_path)
    c = conn.cursor()
    now = get_utc_iso()
    
    # 1. Base legacy students table (maintained for backward compatibility)
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll TEXT,
            class TEXT,
            section TEXT,
            reg_no TEXT,
            email TEXT,
            password_hash TEXT,
            role TEXT DEFAULT 'employee',
            department_id INTEGER,
            site_id INTEGER,
            shift_id INTEGER,
            manager_id INTEGER,
            active INTEGER DEFAULT 1,
            biometric_consent INTEGER DEFAULT 0,
            biometric_consent_timestamp TEXT,
            created_at TEXT NOT NULL
        )
    """)
    
    # Migration safety for students table
    c.execute("PRAGMA table_info(students)")
    existing_student_cols = [row["name"] for row in c.fetchall()]
    student_col_defs = [
        ("email", "TEXT"),
        ("password_hash", "TEXT"),
        ("role", "TEXT DEFAULT 'employee'"),
        ("department_id", "INTEGER"),
        ("site_id", "INTEGER"),
        ("shift_id", "INTEGER"),
        ("manager_id", "INTEGER"),
        ("active", "INTEGER DEFAULT 1"),
        ("biometric_consent", "INTEGER DEFAULT 0"),
        ("biometric_consent_timestamp", "TEXT")
    ]
    for col_name, col_type in student_col_defs:
        if col_name not in existing_student_cols:
            try:
                c.execute(f"ALTER TABLE students ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

    # 2. Dedicated admins table
    c.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # 3. Base attendance table (extended with enterprise fields)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            timestamp TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            address TEXT,
            confidence REAL,
            liveness_passed INTEGER DEFAULT 1,
            geotagged_photo_path TEXT,
            status TEXT DEFAULT 'success',
            reviewed_by_admin_id INTEGER,
            review_note TEXT,
            event_type TEXT DEFAULT 'check_in',
            site_id INTEGER,
            distance_from_site_meters REAL,
            within_geofence INTEGER DEFAULT 1,
            flagged INTEGER DEFAULT 0,
            flag_reason TEXT
        )
    """)
    
    c.execute("PRAGMA table_info(attendance)")
    existing_att_cols = [row["name"] for row in c.fetchall()]
    att_col_defs = [
        ("latitude", "REAL"),
        ("longitude", "REAL"),
        ("address", "TEXT"),
        ("confidence", "REAL"),
        ("liveness_passed", "INTEGER DEFAULT 1"),
        ("geotagged_photo_path", "TEXT"),
        ("status", "TEXT DEFAULT 'success'"),
        ("reviewed_by_admin_id", "INTEGER"),
        ("review_note", "TEXT"),
        ("event_type", "TEXT DEFAULT 'check_in'"),
        ("site_id", "INTEGER"),
        ("distance_from_site_meters", "REAL"),
        ("within_geofence", "INTEGER DEFAULT 1"),
        ("flagged", "INTEGER DEFAULT 0"),
        ("flag_reason", "TEXT")
    ]
    for col_name, col_type in att_col_defs:
        if col_name not in existing_att_cols:
            try:
                c.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
            
    # 4. Embeddings table (128-d deep ResNet vectors)
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
    """)
    
    # 5. Pipeline logs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attendance_id INTEGER,
            run_id TEXT,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (attendance_id) REFERENCES attendance(id) ON DELETE CASCADE
        )
    """)
    c.execute("PRAGMA table_info(pipeline_logs)")
    pcols = [row["name"] for row in c.fetchall()]
    if "run_id" not in pcols:
        c.execute("ALTER TABLE pipeline_logs ADD COLUMN run_id TEXT")
    
    # 6. Sites Table (Office locations, branches, and remote policies)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            address TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            geofence_radius_meters REAL DEFAULT 200.0,
            geofencing_enabled INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)

    # 7. Shifts Table (Industrial shifts)
    c.execute("""
        CREATE TABLE IF NOT EXISTS shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            grace_period_minutes INTEGER DEFAULT 15,
            break_duration_minutes INTEGER DEFAULT 60,
            created_at TEXT NOT NULL
        )
    """)

    # 8. Departments Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            manager_id INTEGER,
            created_at TEXT NOT NULL
        )
    """)

    # 9. Employees Table (Full enterprise domain model)
    c.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            employee_code TEXT UNIQUE NOT NULL,
            department_id INTEGER,
            site_id INTEGER,
            shift_id INTEGER,
            manager_id INTEGER,
            role TEXT DEFAULT 'employee',
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            biometric_consent INTEGER DEFAULT 0,
            biometric_consent_timestamp TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id),
            FOREIGN KEY (site_id) REFERENCES sites(id),
            FOREIGN KEY (shift_id) REFERENCES shifts(id),
            FOREIGN KEY (manager_id) REFERENCES employees(id)
        )
    """)

    # 10. Attendance Events Table (Paired check-in and check-out)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            address TEXT,
            site_id INTEGER,
            distance_from_site_meters REAL,
            within_geofence INTEGER DEFAULT 1,
            confidence REAL,
            liveness_passed INTEGER DEFAULT 1,
            geotagged_photo_path TEXT,
            status TEXT DEFAULT 'on_time',
            flagged INTEGER DEFAULT 0,
            flag_reason TEXT,
            reviewed_by INTEGER,
            review_note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (site_id) REFERENCES sites(id)
        )
    """)

    # 11. Correction Requests Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS correction_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            attendance_date TEXT NOT NULL,
            reason TEXT NOT NULL,
            requested_change TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            reviewed_by INTEGER,
            review_note TEXT,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(id),
            FOREIGN KEY (reviewed_by) REFERENCES employees(id)
        )
    """)

    # 12. Audit Logs Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_id INTEGER,
            actor_role TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            details TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 13. Settings table
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    default_settings = {
        "match_threshold": str(Config.MATCH_THRESHOLD),
        "review_threshold": str(Config.REVIEW_THRESHOLD),
        "duplicate_cooldown_seconds": str(Config.DUPLICATE_COOLDOWN_SECONDS),
        "geocoding_provider": Config.GEOCODING_PROVIDER
    }
    for k, v in default_settings.items():
        c.execute("""
            INSERT OR IGNORE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (k, v, now))
        
    # 14. Seed initial admin account if none exists
    c.execute("SELECT COUNT(*) FROM admins")
    if c.fetchone()[0] == 0:
        admin_hash = generate_password_hash(Config.ADMIN_PASSWORD)
        c.execute("""
            INSERT INTO admins (name, username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("System Administrator", Config.ADMIN_USERNAME, "admin@geoface.edu", admin_hash, now))

    # 15. Seed default Site if none exists
    c.execute("SELECT COUNT(*) FROM sites")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO sites (name, address, latitude, longitude, geofence_radius_meters, geofencing_enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("Tech Park Headquarters (Mumbai)", "Bandra Kurla Complex, Bandra East, Mumbai, Maharashtra 400051", 19.0657, 72.8687, 200.0, 1, now))
        c.execute("""
            INSERT INTO sites (name, address, latitude, longitude, geofence_radius_meters, geofencing_enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("Bengaluru Tech Hub", "Electronic City Phase 1, Bengaluru, Karnataka 560100", 12.8452, 77.6602, 250.0, 1, now))
        c.execute("""
            INSERT INTO sites (name, address, latitude, longitude, geofence_radius_meters, geofencing_enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("Remote / Field Operations (No Geofence)", "Pan-India Field Staff / Sales Operations", 28.6139, 77.2090, 0.0, 0, now))

    # 16. Seed default Shifts if none exists
    c.execute("SELECT COUNT(*) FROM shifts")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO shifts (name, start_time, end_time, grace_period_minutes, break_duration_minutes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("General Shift (09:00 - 18:00)", "09:00", "18:00", 15, 60, now))
        c.execute("""
            INSERT INTO shifts (name, start_time, end_time, grace_period_minutes, break_duration_minutes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Early Morning Shift (07:00 - 15:30)", "07:00", "15:30", 10, 45, now))
        c.execute("""
            INSERT INTO shifts (name, start_time, end_time, grace_period_minutes, break_duration_minutes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Evening Shift (13:00 - 22:00)", "13:00", "22:00", 15, 60, now))
        c.execute("""
            INSERT INTO shifts (name, start_time, end_time, grace_period_minutes, break_duration_minutes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Flexible Shift (09:00 - 18:00, 30m grace)", "09:00", "18:00", 30, 60, now))

    # 17. Seed default Departments if none exists
    c.execute("SELECT COUNT(*) FROM departments")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO departments (name, manager_id, created_at)
            VALUES (?, ?, ?)
        """, ("Engineering & Architecture", None, now))
        c.execute("""
            INSERT INTO departments (name, manager_id, created_at)
            VALUES (?, ?, ?)
        """, ("Operations & Logistics", None, now))
        c.execute("""
            INSERT INTO departments (name, manager_id, created_at)
            VALUES (?, ?, ?)
        """, ("Field Sales & Services", None, now))

    # 18. Synchronize existing students into employees table if employees table is empty
    c.execute("SELECT COUNT(*) FROM employees")
    if c.fetchone()[0] == 0:
        c.execute("SELECT * FROM students")
        existing_students = c.fetchall()
        for s in existing_students:
            s_dict = dict(s)
            emp_code = s_dict.get("roll") or f"EMP-{s_dict['id']:04d}"
            pw = s_dict.get("password_hash") or generate_password_hash("employee123")
            role_val = s_dict.get("role") or "employee"
            if role_val == "user": role_val = "employee"
            try:
                c.execute("""
                    INSERT OR IGNORE INTO employees (
                        id, name, employee_code, department_id, site_id, shift_id,
                        manager_id, role, email, password_hash, active,
                        biometric_consent, biometric_consent_timestamp, created_at
                    ) VALUES (?, ?, ?, 1, 1, 1, NULL, ?, ?, ?, 1, 1, ?, ?)
                """, (
                    s_dict["id"],
                    s_dict["name"],
                    emp_code,
                    role_val,
                    s_dict.get("email") or f"emp{s_dict['id']}@enterprise.corp",
                    pw,
                    now,
                    s_dict.get("created_at") or now
                ))
            except Exception:
                pass

    # Ensure all employees & students have a default password and consent
    default_emp_hash = generate_password_hash("employee123")
    c.execute("""
        UPDATE students
        SET password_hash = COALESCE(password_hash, ?),
            role = CASE WHEN role = 'user' THEN 'employee' ELSE COALESCE(role, 'employee') END,
            active = COALESCE(active, 1),
            biometric_consent = COALESCE(biometric_consent, 1),
            biometric_consent_timestamp = COALESCE(biometric_consent_timestamp, ?)
        WHERE password_hash IS NULL OR password_hash = '' OR biometric_consent IS NULL
    """, (default_emp_hash, now))

    c.execute("""
        UPDATE employees
        SET biometric_consent = 1,
            biometric_consent_timestamp = COALESCE(biometric_consent_timestamp, ?)
        WHERE biometric_consent IS NULL OR biometric_consent = 0
    """, (now,))

    # Indices
    c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_time ON attendance(student_id, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_status ON attendance(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_student_id ON embeddings(student_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_logs_attendance_id ON pipeline_logs(attendance_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_employees_site ON employees(site_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_employees_manager ON employees(manager_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_att_events_emp_time ON attendance_events(employee_id, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_att_events_flagged ON attendance_events(flagged)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_corr_reqs_emp ON correction_requests(employee_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_corr_reqs_status ON correction_requests(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_id)")
    
    conn.commit()
    conn.close()

def log_audit(actor_id, actor_role, action, resource_type, resource_id=None, details=None, db_conn=None):
    """
    Inserts a record into the audit_logs table for enterprise audit trail.
    """
    close_at_end = False
    if db_conn is None:
        db_conn = get_db_connection()
        close_at_end = True
    try:
        now = get_utc_iso()
        db_conn.execute("""
            INSERT INTO audit_logs (actor_id, actor_role, action, resource_type, resource_id, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (actor_id, actor_role, action, resource_type, str(resource_id) if resource_id else None, str(details) if details else None, now))
        db_conn.commit()
    except Exception:
        pass
    finally:
        if close_at_end:
            db_conn.close()

def get_setting(key, default=None):
    """Retrieve runtime setting value from database or fallback."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    if row:
        return row["value"]
    return default

def set_setting(key, value):
    """Update runtime setting value."""
    conn = get_db_connection()
    c = conn.cursor()
    now = get_utc_iso()
    c.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    """, (key, str(value), now))
    conn.commit()
    conn.close()
