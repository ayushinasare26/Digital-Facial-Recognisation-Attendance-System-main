-- Migration 001: Add geotag, liveness, and pipeline tracking columns and tables
-- Target DB: SQLite (Safe ALTER TABLE statements)

-- 1. Create embeddings table to store 128-d deep face vectors (replaces legacy model.yml)
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    embedding TEXT NOT NULL, -- JSON serialized list of 128 float values or binary blob
    created_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

-- 2. Create pipeline_logs table to track stage-by-stage pipeline status
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_id INTEGER,
    stage TEXT NOT NULL,       -- e.g., 'Capture Intake', 'Liveness Check', 'Face Detection', 'Embedding Extraction', 'Embedding Match', 'Geolocation Capture', 'Reverse Geocoding', 'Geotag Stamping', 'Record Saved'
    status TEXT NOT NULL,      -- 'Completed', 'Failed', 'Skipped', 'Processing'
    message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (attendance_id) REFERENCES attendance(id) ON DELETE CASCADE
);

-- 3. Create settings table for dynamic runtime thresholds and configurations
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 4. Create indices for performance
CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp);
CREATE INDEX IF NOT EXISTS idx_embeddings_student_id ON embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_attendance_id ON pipeline_logs(attendance_id);
