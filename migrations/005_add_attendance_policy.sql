-- migrations/005_add_attendance_policy.sql
-- Add attendance_policy column to sites table and default global policy setting
-- Supported values for policy: 'face_only', 'manual_only', 'both', or NULL (inherit)

ALTER TABLE sites ADD COLUMN attendance_policy TEXT DEFAULT NULL;

INSERT OR IGNORE INTO settings (key, value, updated_at)
VALUES ('global_attendance_policy', 'both', datetime('now'));
