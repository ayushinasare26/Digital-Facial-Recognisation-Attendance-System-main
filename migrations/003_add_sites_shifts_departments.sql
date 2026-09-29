-- migrations/003_add_sites_shifts_departments.sql
-- Enterprise Workforce Schema Migration: Sites, Shifts, Departments, Employees, Attendance Events, Correction Requests, and Audit Logs

-- 1. Sites Table (Office locations, branches, and remote/field policies)
CREATE TABLE IF NOT EXISTS sites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    address TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    geofence_radius_meters REAL DEFAULT 200.0,
    geofencing_enabled INTEGER DEFAULT 1, -- 1: Geofence enforced, 0: Field/Remote (no geofence)
    created_at TEXT NOT NULL
);

-- 2. Shifts Table (Industrial shift schedule rules)
CREATE TABLE IF NOT EXISTS shifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    start_time TEXT NOT NULL,          -- Format 'HH:MM' (24-hour, e.g. '09:00')
    end_time TEXT NOT NULL,            -- Format 'HH:MM' (24-hour, e.g. '18:00')
    grace_period_minutes INTEGER DEFAULT 15,
    break_duration_minutes INTEGER DEFAULT 60,
    created_at TEXT NOT NULL
);

-- 3. Departments Table
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    manager_id INTEGER,
    created_at TEXT NOT NULL
);

-- 4. Employees Table (Industrial workforce domain model)
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    employee_code TEXT UNIQUE NOT NULL,
    department_id INTEGER,
    site_id INTEGER,
    shift_id INTEGER,
    manager_id INTEGER,
    role TEXT DEFAULT 'employee',      -- 'employee', 'manager', 'admin'
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    active INTEGER DEFAULT 1,          -- 1: Active, 0: Offboarded / Deactivated
    biometric_consent INTEGER DEFAULT 0, -- 1: Explicit consent granted
    biometric_consent_timestamp TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id),
    FOREIGN KEY (site_id) REFERENCES sites(id),
    FOREIGN KEY (shift_id) REFERENCES shifts(id),
    FOREIGN KEY (manager_id) REFERENCES employees(id)
);

-- 5. Attendance Events Table (Paired check-in and check-out events with geofencing audit)
CREATE TABLE IF NOT EXISTS attendance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,          -- 'check_in' or 'check_out'
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
    status TEXT DEFAULT 'on_time',     -- 'on_time', 'late', 'early_leave', 'overtime', 'flagged'
    flagged INTEGER DEFAULT 0,         -- 1 if flagged for administrative audit
    flag_reason TEXT,
    reviewed_by INTEGER,
    review_note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id),
    FOREIGN KEY (site_id) REFERENCES sites(id)
);

-- 6. Correction Requests Table (Workflow for missed check-in/out or location disputes)
CREATE TABLE IF NOT EXISTS correction_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    attendance_date TEXT NOT NULL,     -- Date 'YYYY-MM-DD'
    reason TEXT NOT NULL,              -- e.g. 'GPS failed', 'Forgot to check out'
    requested_change TEXT NOT NULL,    -- Details of requested event correction
    status TEXT DEFAULT 'pending',     -- 'pending', 'approved', 'rejected'
    reviewed_by INTEGER,               -- Manager or Admin who acted
    review_note TEXT,
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id),
    FOREIGN KEY (reviewed_by) REFERENCES employees(id)
);

-- 7. Audit Logs Table (Biometric and administrative governance access trail)
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id INTEGER,
    actor_role TEXT,
    action TEXT NOT NULL,              -- e.g. 'EXPORT_PAYROLL', 'VIEW_BIOMETRICS', 'APPROVE_CORRECTION'
    resource_type TEXT NOT NULL,       -- e.g. 'employee', 'attendance', 'payroll', 'biometric'
    resource_id TEXT,
    details TEXT,
    created_at TEXT NOT NULL
);

-- Indices for performance and multi-site querying
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department_id);
CREATE INDEX IF NOT EXISTS idx_employees_site ON employees(site_id);
CREATE INDEX IF NOT EXISTS idx_employees_manager ON employees(manager_id);
CREATE INDEX IF NOT EXISTS idx_employees_active ON employees(active);
CREATE INDEX IF NOT EXISTS idx_att_events_emp_time ON attendance_events(employee_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_att_events_date ON attendance_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_att_events_flagged ON attendance_events(flagged);
CREATE INDEX IF NOT EXISTS idx_corr_reqs_emp ON correction_requests(employee_id);
CREATE INDEX IF NOT EXISTS idx_corr_reqs_status ON correction_requests(status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_id);
