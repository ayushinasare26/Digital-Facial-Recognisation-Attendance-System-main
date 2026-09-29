import os
import json
import unittest
import numpy as np

import app as flask_app

class TestE2EPipeline(unittest.TestCase):
    created_attendance_ids = []

    def setUp(self):
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        if cls.created_attendance_ids:
            import sqlite3
            from core.db import get_db_connection
            conn = get_db_connection()
            for att_id in cls.created_attendance_ids:
                row = conn.execute("SELECT geotagged_photo_path FROM attendance WHERE id = ?", (att_id,)).fetchone()
                if row and row["geotagged_photo_path"] and os.path.exists(row["geotagged_photo_path"]):
                    try:
                        os.remove(row["geotagged_photo_path"])
                    except Exception:
                        pass
                conn.execute("DELETE FROM attendance WHERE id = ?", (att_id,))
                conn.execute("DELETE FROM pipeline_logs WHERE attendance_id = ?", (att_id,))
            conn.commit()
            conn.close()

    def test_01_dashboard_summary_api(self):
        with self.client.session_transaction() as sess:
            sess["role"] = "admin"
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"

        resp = self.client.get("/admin/dashboard/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("total_enrolled", data)
        self.assertIn("confidence_distribution", data)
        self.assertIn("trend", data)

    def test_02_csv_download(self):
        with self.client.session_transaction() as sess:
            sess["role"] = "admin"
            sess["admin_id"] = 1
            sess["admin_username"] = "admin"

        resp = self.client.get("/admin/reports/csv")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "text/csv")
        content = resp.data.decode("utf-8")
        self.assertIn("ID,Student ID,Name", content)

    def test_03_mark_attendance_demo_mode(self):
        # Demo mode test with mock GPS & demo selfie (student 29)
        with self.client.session_transaction() as sess:
            sess["role"] = "user"
            sess["student_id"] = 29
            sess["student_name"] = "Aayushi"
            sess["roll"] = "24070521203"

        resp = self.client.post("/portal/mark-attendance", data={
            "latitude": "19.0760",
            "longitude": "72.8777",
            "challenge_type": "blink",
            "is_demo": "true",
            "bypass_cooldown": "true"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("attendance_id", data)
        self.__class__.created_attendance_ids.append(data.get("attendance_id"))
        self.assertIn("stages", data)
        self.assertEqual(len(data["stages"]), 9)
        # Check all 9 stages executed
        stage_names = [s["stage"] for s in data["stages"]]
        self.assertIn("Photo Received", stage_names)
        self.assertIn("Liveness Check", stage_names)
        self.assertIn("Face Detection", stage_names)
        self.assertIn("Embedding Extraction", stage_names)
        self.assertIn("Embedding Match", stage_names)
        self.assertIn("Geolocation Capture", stage_names)
        self.assertIn("Reverse Geocoding", stage_names)
        self.assertIn("Geotag Stamping", stage_names)
        self.assertIn("Record Saved", stage_names)

    def test_04_mark_attendance_duplicate_prevention(self):
        with self.client.session_transaction() as sess:
            sess["role"] = "user"
            sess["student_id"] = 29
            sess["student_name"] = "Aayushi"
            sess["roll"] = "24070521203"

        # Immediate subsequent submission without bypass_cooldown should be blocked by cooldown
        resp = self.client.post("/portal/mark-attendance", data={
            "latitude": "19.0760",
            "longitude": "72.8777",
            "challenge_type": "blink",
            "is_demo": "true"
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data.get("error_stage"), "Duplicate Check")
        self.assertIn("Duplicate attendance prevented", data.get("message"))

    def test_05_mark_attendance_location_denied_flags_record(self):
        with self.client.session_transaction() as sess:
            sess["role"] = "user"
            sess["student_id"] = 29
            sess["student_name"] = "Aayushi"
            sess["roll"] = "24070521203"

        # Omitting coordinates flags the record as 'flagged' for review
        resp = self.client.post("/portal/mark-attendance", data={
            "challenge_type": "blink",
            "is_demo": "true",
            "bypass_cooldown": "true"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        if data.get("attendance_id"):
            self.__class__.created_attendance_ids.append(data.get("attendance_id"))
        self.assertEqual(data.get("status"), "flagged")
        self.assertIn("Denied", data.get("address"))

if __name__ == "__main__":
    unittest.main()
