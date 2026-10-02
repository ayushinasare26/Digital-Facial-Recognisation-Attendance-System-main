"""
tests/test_attendance_policy.py - Tests for Admin-Controlled Attendance Method Policy
(Face Recognition vs Manual Attendance with Global & Per-Site Inheritance)
"""

import json
import unittest
from app import app
from core.db import (
    get_db_connection,
    get_setting,
    set_setting,
    get_effective_attendance_policy,
    get_site_attendance_policy_info,
    get_all_sites_attendance_policies,
    update_attendance_policies
)

class TestAttendancePolicy(unittest.TestCase):
    test_emp_id = 29  # Aayushi (has site_id=1, general shift)

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        self._clear_test_events()
        self._save_original_policy_state()

    def tearDown(self):
        self._clear_test_events()
        self._restore_original_policy_state()

    def _save_original_policy_state(self):
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key='global_attendance_policy'")
        row = c.fetchone()
        self.orig_global = row["value"] if row else "both"

        c.execute("SELECT id, attendance_policy FROM sites")
        self.orig_site_overrides = {r["id"]: r["attendance_policy"] for r in c.fetchall()}
        conn.close()

    def _restore_original_policy_state(self):
        set_setting("global_attendance_policy", self.orig_global)
        conn = get_db_connection()
        c = conn.cursor()
        for site_id, override in self.orig_site_overrides.items():
            c.execute("UPDATE sites SET attendance_policy = ? WHERE id = ?", (override, site_id))
        conn.commit()
        conn.close()

    def _clear_test_events(self):
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

    def _login_admin(self, admin_id=1):
        with self.client.session_transaction() as sess:
            sess["role"] = "admin"
            sess["admin_id"] = admin_id
            sess["admin_user"] = "admin"

    # =========================================================================
    # 1. HELPER RESOLUTION & INHERITANCE LOGIC TESTS
    # =========================================================================

    def test_01_global_default_resolution_when_no_override(self):
        """A site with NULL override inherits the global default policy."""
        set_setting("global_attendance_policy", "both")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        policy = get_effective_attendance_policy(site_id=1)
        self.assertEqual(policy, "both")

        # Change global default to face_only
        set_setting("global_attendance_policy", "face_only")
        policy = get_effective_attendance_policy(site_id=1)
        self.assertEqual(policy, "face_only")

        # Change global default to manual_only
        set_setting("global_attendance_policy", "manual_only")
        policy = get_effective_attendance_policy(site_id=1)
        self.assertEqual(policy, "manual_only")

    def test_02_site_override_takes_precedence_over_global(self):
        """An explicit per-site override takes precedence over the global default."""
        set_setting("global_attendance_policy", "face_only")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = 'manual_only' WHERE id = 1")
        conn.commit()
        conn.close()

        # Site 1 has manual_only override, should ignore face_only global default
        policy = get_effective_attendance_policy(site_id=1)
        self.assertEqual(policy, "manual_only")

        # Non-overridden site inherits global default
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 2")
        conn.commit()
        conn.close()
        policy_site2 = get_effective_attendance_policy(site_id=2)
        self.assertEqual(policy_site2, "face_only")

    def test_03_reverting_site_to_inherit_restores_global_default(self):
        """Setting site override to None/'inherit' cleanly restores global inheritance."""
        update_attendance_policies(
            global_policy="both",
            site_overrides={"1": "face_only"},
            actor_id=1
        )
        self.assertEqual(get_effective_attendance_policy(1), "face_only")

        # Reset to inherit
        update_attendance_policies(
            global_policy="both",
            site_overrides={"1": "inherit"},
            actor_id=1
        )
        self.assertEqual(get_effective_attendance_policy(1), "both")

    # =========================================================================
    # 2. ADMIN API & AUDIT TRAIL LOGGING
    # =========================================================================

    def test_04_admin_api_updates_policies_and_logs_audit(self):
        """Admin API successfully saves global & site overrides and creates audit logs."""
        self._login_admin()

        resp = self.client.post("/admin/api/admin/settings/attendance-policy", json={
            "global_attendance_policy": "face_only",
            "site_overrides": {
                "1": "both",
                "2": "manual_only"
            }
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["global_policy"], "face_only")

        # Check DB directly
        self.assertEqual(get_setting("global_attendance_policy"), "face_only")
        self.assertEqual(get_effective_attendance_policy(1), "both")
        self.assertEqual(get_effective_attendance_policy(2), "manual_only")

        # Verify Audit Log entry
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT action, resource_type, details FROM audit_logs WHERE action LIKE 'UPDATE_%_ATTENDANCE_POLICY' ORDER BY id DESC LIMIT 5")
        logs = c.fetchall()
        conn.close()

        actions = [l["action"] for l in logs]
        self.assertIn("UPDATE_GLOBAL_ATTENDANCE_POLICY", actions)

    def test_05_unauthorized_user_cannot_access_policy_admin_api(self):
        """Non-admin users cannot update attendance policies."""
        # Not logged in
        resp = self.client.post("/admin/api/admin/settings/attendance-policy", json={
            "global_attendance_policy": "face_only"
        })
        self.assertIn(resp.status_code, (302, 401, 403))

        # Logged in as employee
        self._login_employee()
        resp = self.client.post("/admin/api/admin/settings/attendance-policy", json={
            "global_attendance_policy": "face_only"
        })
        self.assertIn(resp.status_code, (302, 401, 403))

    # =========================================================================
    # 3. SERVER-SIDE ENFORCEMENT ON CHECK-IN / CHECK-OUT
    # =========================================================================

    def test_06_manual_attendance_blocked_when_policy_is_face_only(self):
        """When policy is face_only, manual check-in and check-out are rejected with 403."""
        set_setting("global_attendance_policy", "face_only")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        self._login_employee()

        # Try manual check-in via /employee/manual-check-in
        resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687
        })
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertTrue(data.get("policy_mismatch"))
        self.assertIn("Face Recognition", data["message"])

        # Try manual check-in via /employee/check-in with method="manual"
        resp2 = self.client.post("/employee/check-in", json={
            "attendance_method": "manual",
            "latitude": 19.0657,
            "longitude": 72.8687
        })
        self.assertEqual(resp2.status_code, 403)
        data2 = resp2.get_json()
        self.assertTrue(data2.get("policy_mismatch"))

    def test_07_face_attendance_blocked_when_policy_is_manual_only(self):
        """When policy is manual_only, face recognition check-in is rejected with 403."""
        set_setting("global_attendance_policy", "manual_only")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        self._login_employee()

        # Try face check-in via /employee/check-in
        resp = self.client.post("/employee/check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "is_demo": True
        })
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertTrue(data.get("policy_mismatch"))
        self.assertIn("Manual Attendance", data["message"])

    def test_08_both_policy_permits_manual_check_in_and_out(self):
        """When policy is both, employee can record manual attendance successfully."""
        set_setting("global_attendance_policy", "both")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        self._login_employee()

        # Manual Check-In
        in_resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687
        })
        self.assertEqual(in_resp.status_code, 200)
        in_data = in_resp.get_json()
        self.assertTrue(in_data["success"])
        self.assertEqual(in_data["attendance_method"], "manual")

        # Manual Check-Out
        out_resp = self.client.post("/employee/manual-check-out", json={
            "latitude": 19.0657,
            "longitude": 72.8687,
            "bypass_cooldown": True
        })
        self.assertEqual(out_resp.status_code, 200)
        out_data = out_resp.get_json()
        self.assertTrue(out_data["success"])
        self.assertEqual(out_data["attendance_method"], "manual")

    # =========================================================================
    # 4. PER-SITE OVERRIDE SERVER-SIDE ENFORCEMENT
    # =========================================================================

    def test_09_site_override_overrules_global_for_employee(self):
        """Global policy is face_only, but employee site has manual_only override -> manual allowed."""
        set_setting("global_attendance_policy", "face_only")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = 'manual_only' WHERE id = 1")
        conn.commit()
        conn.close()

        self._login_employee()

        # Manual check-in should SUCCEED because site 1 overrides to manual_only
        resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])

    # =========================================================================
    # 5. DYNAMIC MID-SESSION POLICY CHANGE
    # =========================================================================

    def test_10_mid_session_policy_change_rejected_on_submission(self):
        """If admin changes policy while employee is on page, next check-in is rejected."""
        # 1. Start with policy allowing manual
        set_setting("global_attendance_policy", "both")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        self._login_employee()

        # Employee checks their policy via endpoint (returns both)
        p_resp = self.client.get("/employee/api/attendance-policy")
        self.assertEqual(p_resp.status_code, 200)
        self.assertEqual(p_resp.get_json()["effective_policy"], "both")

        # 2. Admin switches policy to face_only in the background
        set_setting("global_attendance_policy", "face_only")

        # 3. Employee attempts manual check-in -> immediate 403 rejection
        resp = self.client.post("/employee/manual-check-in", json={
            "latitude": 19.0657,
            "longitude": 72.8687
        })
        self.assertEqual(resp.status_code, 403)
        data = resp.get_json()
        self.assertTrue(data["policy_mismatch"])
        self.assertIn("refresh", data["message"])

    # =========================================================================
    # 6. FACE ENROLLMENT REQUIRED EDGE CASE
    # =========================================================================

    def test_11_face_only_policy_blocks_unenrolled_employee(self):
        """When face_only is active, employee with 0 enrolled face embeddings is blocked with guidance."""
        set_setting("global_attendance_policy", "face_only")
        conn = get_db_connection()
        conn.execute("UPDATE sites SET attendance_policy = NULL WHERE id = 1")

        # Create temporary unenrolled employee
        c = conn.cursor()
        c.execute("""
            INSERT INTO employees (name, employee_code, email, password_hash, site_id, active, created_at)
            VALUES ('Unenrolled Staff', 'UNENROLLED_999', 'unenrolled@test.com', 'dummy_hash', 1, 1, '2026-01-01T00:00:00Z')
        """)
        new_emp_id = c.lastrowid
        conn.commit()
        conn.close()

        try:
            self._login_employee(emp_id=new_emp_id, emp_name="Unenrolled Staff")

            # Policy check endpoint reports has_face_enrollment: False
            p_resp = self.client.get("/employee/api/attendance-policy")
            self.assertEqual(p_resp.status_code, 200)
            self.assertFalse(p_resp.get_json()["has_face_enrollment"])

            # Face check-in attempt is blocked
            resp = self.client.post("/employee/check-in", json={
                "latitude": 19.0657,
                "longitude": 72.8687,
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
            })
            self.assertEqual(resp.status_code, 403)
            data = resp.get_json()
            self.assertTrue(data.get("enrollment_required"))
            self.assertIn("biometric enrollment", data["message"].lower())
        finally:
            conn = get_db_connection()
            conn.execute("DELETE FROM employees WHERE id = ?", (new_emp_id,))
            conn.commit()
            conn.close()

if __name__ == "__main__":
    unittest.main()
