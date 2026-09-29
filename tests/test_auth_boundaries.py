"""
tests/test_auth_boundaries.py - Enterprise Role Boundaries & Data Scoping Tests
Verifies strict three-portal role separation and data scoping:
1. Anonymous user blocked from protected employee, manager, and admin routes
2. Regular employee (role='employee'/'user') cannot access any /admin/* or /manager/* endpoints
3. Regular employee cannot view another employee's attendance records by guessing IDs
4. Regular employee's attendance queries are strictly scoped to their own session employee_id
5. Manager (role='manager') can view their own team's data
6. Manager CANNOT view another manager's team data (server-side query rejection with 403)
7. Manager CANNOT approve correction requests for employees outside their team (403 Forbidden)
8. Admin (role='admin') has company-wide access across all sites and departments
9. Biometric identity mismatch (proxy attempt) is rejected with 403
"""

import os
import unittest
import sqlite3
from app import app
from core.db import get_db_connection

class TestAuthBoundaries(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_unauthenticated_access_redirects(self):
        """Anonymous visitors must be redirected when accessing protected routes."""
        # Employee protected routes
        res = self.client.get("/employee/check-in-out", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/employee/login", res.headers["Location"])

        res = self.client.get("/employee/my-attendance", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/employee/login", res.headers["Location"])

        # Manager protected routes
        res = self.client.get("/manager/dashboard", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/employee/login", res.headers["Location"])

        # Admin protected routes
        res = self.client.get("/admin/dashboard", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/admin/login", res.headers["Location"])

        res = self.client.get("/admin/attendance", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/admin/login", res.headers["Location"])

    def test_employee_cannot_access_admin_dashboard(self):
        """A logged-in employee cannot access /admin/dashboard."""
        with self.client.session_transaction() as sess:
            sess["role"] = "employee"
            sess["employee_id"] = 29
            sess["student_id"] = 29
            sess["employee_name"] = "Aayushi"

        res = self.client.get("/admin/dashboard", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/employee/check-in-out", res.headers["Location"])

    def test_employee_cannot_access_manager_portal(self):
        """A regular employee cannot access manager team endpoints."""
        with self.client.session_transaction() as sess:
            sess["role"] = "employee"
            sess["employee_id"] = 29
            sess["student_id"] = 29
            sess["employee_name"] = "Aayushi"

        res = self.client.get("/manager/dashboard", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/employee/check-in-out", res.headers["Location"])

    def test_employee_cannot_view_other_employees_record(self):
        """
        Data Scoping Check:
        An employee cannot view attendance detail of a record belonging to another employee.
        """
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO attendance_events (employee_id, event_type, timestamp, confidence, status, address, created_at)
            VALUES (22, 'check_in', '2026-09-27T10:00:00Z', 0.95, 'on_time', 'Zone B', '2026-09-27T10:00:00Z')
        """)
        other_record_id = c.lastrowid

        c.execute("""
            INSERT INTO attendance_events (employee_id, event_type, timestamp, confidence, status, address, created_at)
            VALUES (29, 'check_in', '2026-09-27T10:05:00Z', 0.94, 'on_time', 'Zone A', '2026-09-27T10:05:00Z')
        """)
        own_record_id = c.lastrowid
        conn.commit()
        conn.close()

        with self.client.session_transaction() as sess:
            sess["role"] = "employee"
            sess["employee_id"] = 29
            sess["student_id"] = 29
            sess["employee_name"] = "Aayushi"

        try:
            # Own record -> 200 OK
            res_own = self.client.get(f"/employee/my-attendance/{own_record_id}")
            self.assertEqual(res_own.status_code, 200)
            data_own = res_own.get_json()
            self.assertEqual(data_own["employee_id"], 29)

            # Other employee's record -> Denied (404/403)
            res_other = self.client.get(f"/employee/my-attendance/{other_record_id}")
            self.assertEqual(res_other.status_code, 404)
        finally:
            conn = get_db_connection()
            conn.execute("DELETE FROM attendance_events WHERE id IN (?, ?)", (other_record_id, own_record_id))
            conn.commit()
            conn.close()

    def test_manager_cannot_view_other_teams_data(self):
        """
        Manager Team Scoping Check:
        Manager A (Dept 1) must be forbidden (403) when attempting to view records
        of an employee in Dept 2 by tampering with employee_id query parameter.
        """
        conn = get_db_connection()
        c = conn.cursor()
        
        # Create test manager in dept 1
        c.execute("""
            INSERT INTO employees (name, employee_code, department_id, site_id, shift_id, role, password_hash, created_at)
            VALUES ('Manager A', 'MGR-A', 1, 1, 1, 'manager', 'hash', '2026-09-29T10:00:00')
        """)
        mgr_a_id = c.lastrowid
        c.execute("UPDATE departments SET manager_id = ? WHERE id = 1", (mgr_a_id,))

        # Create employee in dept 2 (other team)
        c.execute("""
            INSERT INTO employees (name, employee_code, department_id, site_id, shift_id, role, password_hash, created_at)
            VALUES ('Employee in Dept 2', 'EMP-D2', 2, 1, 1, 'employee', 'hash', '2026-09-29T10:00:00')
        """)
        other_emp_id = c.lastrowid
        conn.commit()
        conn.close()

        with self.client.session_transaction() as sess:
            sess["role"] = "manager"
            sess["manager_id"] = mgr_a_id
            sess["employee_id"] = mgr_a_id
            sess["employee_name"] = "Manager A"

        try:
            # Manager A queries attendance specifically for other_emp_id -> must be rejected with 403 Forbidden!
            res = self.client.get(f"/manager/attendance?employee_id={other_emp_id}")
            self.assertEqual(res.status_code, 403)
        finally:
            conn = get_db_connection()
            conn.execute("DELETE FROM employees WHERE id IN (?, ?)", (mgr_a_id, other_emp_id))
            conn.commit()
            conn.close()

    def test_manager_cannot_approve_other_teams_correction_request(self):
        """
        Manager Team Scoping Check on Correction Requests:
        Manager A cannot approve a correction request submitted by an employee in Dept 2.
        """
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO employees (name, employee_code, department_id, site_id, shift_id, role, password_hash, created_at)
            VALUES ('Manager X', 'MGR-X', 1, 1, 1, 'manager', 'hash', '2026-09-29T10:00:00')
        """)
        mgr_x_id = c.lastrowid
        c.execute("UPDATE departments SET manager_id = ? WHERE id = 1", (mgr_x_id,))

        c.execute("""
            INSERT INTO employees (name, employee_code, department_id, site_id, shift_id, role, password_hash, created_at)
            VALUES ('Employee Y (Dept 2)', 'EMP-Y', 2, 1, 1, 'employee', 'hash', '2026-09-29T10:00:00')
        """)
        emp_y_id = c.lastrowid

        c.execute("""
            INSERT INTO correction_requests (employee_id, attendance_date, reason, requested_change, status, created_at)
            VALUES (?, '2026-09-28', 'GPS failed', 'Mark check-in at 09:00', 'pending', '2026-09-28T10:00:00')
        """, (emp_y_id,))
        corr_req_id = c.lastrowid
        conn.commit()
        conn.close()

        with self.client.session_transaction() as sess:
            sess["role"] = "manager"
            sess["manager_id"] = mgr_x_id
            sess["employee_id"] = mgr_x_id
            sess["employee_name"] = "Manager X"

        try:
            # Manager X attempts to approve correction request for Employee Y in Dept 2 -> 403 Forbidden!
            res = self.client.post(f"/manager/correction-requests/{corr_req_id}/review", data={
                "action": "approve",
                "review_note": "Illegal cross-team approval attempt"
            })
            self.assertEqual(res.status_code, 403)
        finally:
            conn = get_db_connection()
            conn.execute("DELETE FROM correction_requests WHERE id = ?", (corr_req_id,))
            conn.execute("DELETE FROM employees WHERE id IN (?, ?)", (mgr_x_id, emp_y_id))
            conn.commit()
            conn.close()

    def test_admin_can_access_admin_portal(self):
        """An authenticated administrator (role='admin') has access to administrative pages."""
        with self.client.session_transaction() as sess:
            sess["role"] = "admin"
            sess["admin_id"] = 1
            sess["admin_name"] = "Administrator"
            sess["admin_username"] = "admin"

        res_dash = self.client.get("/admin/dashboard")
        self.assertEqual(res_dash.status_code, 200)

        res_sites = self.client.get("/admin/sites")
        self.assertEqual(res_sites.status_code, 200)

        res_shifts = self.client.get("/admin/shifts")
        self.assertEqual(res_shifts.status_code, 200)

        res_att = self.client.get("/admin/attendance")
        self.assertEqual(res_att.status_code, 200)

        res_flag = self.client.get("/admin/flagged")
        self.assertEqual(res_flag.status_code, 200)

        res_payroll = self.client.get("/admin/payroll-export")
        self.assertEqual(res_payroll.status_code, 200)

    def test_proxy_attendance_is_rejected(self):
        """
        Biometric Integrity Check:
        If a logged-in employee submits attendance where the face recognized
        belongs to a different person, return 403 Forbidden and flag proxy attempt.
        """
        with self.client.session_transaction() as sess:
            sess["role"] = "employee"
            sess["employee_id"] = 22  # Logged in as ID 22
            sess["student_id"] = 22
            sess["employee_name"] = "Student TwentyTwo"
            sess["student_name"] = "Student TwentyTwo"

        # Post demo attendance (demo mode returns student_id 29)
        res = self.client.post("/employee/check-in", data={
            "is_demo": "true",
            "bypass_cooldown": "true",
            "latitude": "19.0657",
            "longitude": "72.8687",
            "challenge_type": "blink"
        })
        
        # Session employee_id is 22 and demo face is 29 -> proxy attempt!
        self.assertEqual(res.status_code, 403)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("Biometric mismatch", json_data["message"])

    @classmethod
    def tearDownClass(cls):
        """Clean up proxy test artifacts."""
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM attendance_events WHERE flag_reason LIKE '%Session was Student TwentyTwo%'")
        c.execute("DELETE FROM attendance WHERE review_note LIKE '%Session was Student TwentyTwo%'")
        conn.commit()
        conn.close()

if __name__ == "__main__":
    unittest.main()
