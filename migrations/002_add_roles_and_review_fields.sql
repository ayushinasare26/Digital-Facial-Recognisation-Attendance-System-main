-- Migration 002: Add admins table, student auth fields, and attendance review columns

-- 1. Create admins table for dedicated administrative authentication
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 2. Extend students table with auth fields if they don't already exist
-- (Handled safely in core/db.py migration helper)
-- ALTER TABLE students ADD COLUMN email TEXT;
-- ALTER TABLE students ADD COLUMN password_hash TEXT;
-- ALTER TABLE students ADD COLUMN role TEXT DEFAULT 'user';

-- 3. Extend attendance table with admin review fields if they don't already exist
-- (Handled safely in core/db.py migration helper)
-- ALTER TABLE attendance ADD COLUMN reviewed_by_admin_id INTEGER;
-- ALTER TABLE attendance ADD COLUMN review_note TEXT;

-- 4. Create performance indexes for scoped user & admin queries
CREATE INDEX IF NOT EXISTS idx_attendance_student_time ON attendance(student_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_attendance_status ON attendance(status);
