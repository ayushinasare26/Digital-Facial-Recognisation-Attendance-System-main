-- migrations/004_add_attendance_method.sql
-- Add attendance_method column to attendance_events and attendance tables
-- Supported values: 'face' (default), 'manual'

ALTER TABLE attendance_events ADD COLUMN attendance_method TEXT DEFAULT 'face';
ALTER TABLE attendance ADD COLUMN attendance_method TEXT DEFAULT 'face';
