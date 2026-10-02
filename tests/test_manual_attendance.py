"""
tests/test_manual_attendance.py - Comprehensive Unit & Integration Tests for Optional Facial Recognition

Validates:
Test 1: Employee Face Recognition Check-In (existing behavior preserved)
Test 2: Employee Manual Check-In (no camera/biometric scan required, attendance_method='manual')
Test 3: Employee Manual Check-Out (event_type='check_out', attendance_method='manual')
Test 4: Duplicate Manual Check-In rejected ("You are already checked in.")
Test 5: Manual Check-Out without Check-In rejected ("You cannot check out because you have not checked in today.")
Test 6: Hybrid Flow: Manual Check-In -> Face Check-Out (both recorded with proper methods)
Test 7: Hybrid Flow: Face Check-In -> Manual Check-Out (both recorded with proper methods)
Test 8: Security Integrity: Tampered client employee_id ignored in favor of server session employee_id
Test 9: Geofence Validation: Coordinates outside geofence flagged properly in manual attendance
Test 10: Unauthenticated access blocked
"""

import os
import json
import unittest
from datetime import datetime, timezone
from app import app
from core.db import get_db_connection

class TestManualAttendance(unittest.TestCase):
    test_emp_id = 29  # Aayushi in seeded DB (matches demo biometric embedding)
    created_event_ids = []

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        self._clear_test_events()

    def tearDown(self):
        self._clear_test_events()

    def _clear_test_events(self):
        """Clean up attendance events created during testing for test_emp_id."""
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM attendance_events WHERE employee_id = ?", (self.test_emp_id,))
        c.execute("DELETE FROM attendance WHERE student_id = ?", (self.test_emp_id,))
        conn.commit()
        conn.close()

    def _login_employee(self, emp_id=29, emp_name="Aayushi"):
        with self.client.session_transaction() as sess:
            sess["role"] = "employee"
            sess["employee_id"] = emp_id
            sess["student_id"] = emp_id
            sess["employee_name"] = emp_name
            sess["student_name"] = emp_name
            sess["employee_code"] = "789"
            sess["roll"] = "789"

    def test_01_face_check_in_works(self):
        """Test 1: Existing Face Recognition attendance works exactly as before with method='face'."""
        self._login_employee()
        resp = self.client.post("/employee/check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "event_type": "check_in",
            "is_demo": True,
            "bypass_cooldown": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("attendance_method"), "face")

        # Verify DB entry
        conn = get_db_connection()
        row = conn.execute(
            "SELECT event_type, attendance_method FROM attendance_events WHERE employee_id = ? ORDER BY id DESC LIMIT 1",
            (self.test_emp_id,)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "check_in")
        self.assertEqual(row["attendance_method"], "face")

    def test_02_manual_check_in_works(self):
        """Test 2: Manual Check-In creates attendance without camera/face recognition."""
        self._login_employee()
        resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("attendance_method"), "manual")
        self.assertEqual(data.get("event_type"), "check_in")

        # Verify DB entry: no biometric artifacts
        conn = get_db_connection()
        row = conn.execute(
            "SELECT event_type, attendance_method, confidence, liveness_passed, geotagged_photo_path FROM attendance_events WHERE employee_id = ? ORDER BY id DESC LIMIT 1",
            (self.test_emp_id,)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "check_in")
        self.assertEqual(row["attendance_method"], "manual")
        self.assertIsNone(row["confidence"])
        self.assertIsNone(row["liveness_passed"])
        self.assertIsNone(row["geotagged_photo_path"])

    def test_03_manual_check_out_works(self):
        """Test 3: Manual Check-Out records successfully after check-in."""
        self._login_employee()
        # First check in manually
        in_resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(in_resp.status_code, 200)

        # Now check out manually
        out_resp = self.client.post("/employee/manual-check-out", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(out_resp.status_code, 200)
        data = out_resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("attendance_method"), "manual")
        self.assertEqual(data.get("event_type"), "check_out")

        # Verify DB contains check_out record
        conn = get_db_connection()
        events = conn.execute(
            "SELECT event_type, attendance_method FROM attendance_events WHERE employee_id = ? ORDER BY id ASC",
            (self.test_emp_id,)
        ).fetchall()
        conn.close()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "check_in")
        self.assertEqual(events[0]["attendance_method"], "manual")
        self.assertEqual(events[1]["event_type"], "check_out")
        self.assertEqual(events[1]["attendance_method"], "manual")

    def test_04_duplicate_manual_check_in_rejected(self):
        """Test 4: Employee attempting manual check-in twice without checkout is rejected."""
        self._login_employee()
        # First check-in
        res1 = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(res1.status_code, 200)

        # Second check-in
        res2 = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(res2.status_code, 400)
        data = res2.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("already checked in", data.get("message", "").lower())

    def test_05_manual_check_out_without_check_in_rejected(self):
        """Test 5: Employee attempting manual check-out without checking in today is rejected."""
        self._login_employee()
        resp = self.client.post("/employee/manual-check-out", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("have not checked in", data.get("message", "").lower())

    def test_06_manual_check_in_then_face_check_out(self):
        """Test 6: Hybrid flow: Employee manually checks in and then uses Face Recognition to check out."""
        self._login_employee()
        # 1. Manual Check-In
        in_resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(in_resp.status_code, 200)

        # 2. Face Recognition Check-Out
        out_resp = self.client.post("/employee/check-out", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "event_type": "check_out",
            "is_demo": True,
            "bypass_cooldown": True
        })
        self.assertEqual(out_resp.status_code, 200)
        out_data = out_resp.get_json()
        self.assertTrue(out_data.get("success"))

        # Verify DB entries: In=manual, Out=face
        conn = get_db_connection()
        events = conn.execute(
            "SELECT event_type, attendance_method FROM attendance_events WHERE employee_id = ? ORDER BY id ASC",
            (self.test_emp_id,)
        ).fetchall()
        conn.close()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "check_in")
        self.assertEqual(events[0]["attendance_method"], "manual")
        self.assertEqual(events[1]["event_type"], "check_out")
        self.assertEqual(events[1]["attendance_method"], "face")

    def test_07_face_check_in_then_manual_check_out(self):
        """Test 7: Hybrid flow: Employee uses Face Recognition to check in and manual attendance to check out."""
        self._login_employee()
        # 1. Face Recognition Check-In
        in_resp = self.client.post("/employee/check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "event_type": "check_in",
            "is_demo": True,
            "bypass_cooldown": True
        })
        self.assertEqual(in_resp.status_code, 200)

        # 2. Manual Check-Out
        out_resp = self.client.post("/employee/manual-check-out", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(out_resp.status_code, 200)
        out_data = out_resp.get_json()
        self.assertTrue(out_data.get("success"))
        self.assertEqual(out_data.get("attendance_method"), "manual")

        # Verify DB entries: In=face, Out=manual
        conn = get_db_connection()
        events = conn.execute(
            "SELECT event_type, attendance_method FROM attendance_events WHERE employee_id = ? ORDER BY id ASC",
            (self.test_emp_id,)
        ).fetchall()
        conn.close()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "check_in")
        self.assertEqual(events[0]["attendance_method"], "face")
        self.assertEqual(events[1]["event_type"], "check_out")
        self.assertEqual(events[1]["attendance_method"], "manual")

    def test_08_tampered_employee_id_in_request_is_ignored(self):
        """Test 8: Changing employee_id in client payload is strictly ignored in favor of session employee ID."""
        self._login_employee(emp_id=self.test_emp_id)

        conn = get_db_connection()
        count_before_22 = conn.execute("SELECT COUNT(*) FROM attendance_events WHERE employee_id = 22").fetchone()[0]
        conn.close()

        # Forged client payload tries to submit for ID 22
        resp = self.client.post("/employee/manual-check-in", json={
            "employee_id": 22,
            "student_id": 22,
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(resp.status_code, 200)

        conn = get_db_connection()
        count_after_22 = conn.execute("SELECT COUNT(*) FROM attendance_events WHERE employee_id = 22").fetchone()[0]
        latest_event = conn.execute(
            "SELECT employee_id FROM attendance_events WHERE employee_id = ? ORDER BY id DESC LIMIT 1",
            (self.test_emp_id,)
        ).fetchone()
        conn.close()

        # No event was recorded for employee 22
        self.assertEqual(count_after_22, count_before_22)
        # Event was recorded for the authenticated session employee
        self.assertIsNotNone(latest_event)
        self.assertEqual(latest_event["employee_id"], self.test_emp_id)

    def test_09_geofence_validation_manual_attendance(self):
        """Test 9: Employee outside the configured site geofence has attendance recorded with flagged status."""
        self._login_employee()
        # Coordinates in Delhi (~1100km away from Mumbai site)
        resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 28.6139,
            "longitude": 77.2090,
            "bypass_cooldown": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("flagged"))
        self.assertFalse(data.get("within_geofence"))
        self.assertIn("geofence", data.get("flag_reason", "").lower())

        conn = get_db_connection()
        row = conn.execute(
            "SELECT within_geofence, flagged, flag_reason, attendance_method FROM attendance_events WHERE employee_id = ? ORDER BY id DESC LIMIT 1",
            (self.test_emp_id,)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["attendance_method"], "manual")
        self.assertEqual(row["within_geofence"], 0)
        self.assertEqual(row["flagged"], 1)

    def test_10_unauthenticated_manual_attendance_rejected(self):
        """Test 10: Manual attendance request without session authentication is rejected with 401 or redirect."""
        resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687
        })
        # login_required_employee decorator returns 401 for JSON requests or 302 redirect
        self.assertIn(resp.status_code, [401, 302])

if __name__ == "__main__":
    unittest.main()
