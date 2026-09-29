import os
import sys
import datetime
from werkzeug.security import generate_password_hash

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db import get_db_connection, init_db, log_audit
from core.geofence import validate_geofence
from core.shift_engine import evaluate_check_in, evaluate_check_out, calculate_shift_hours

def seed_enterprise_data():
    """
    Seeds comprehensive enterprise data for testing and demonstration:
    - Sites: Mumbai HQ, Bengaluru Hub, Remote Field Operations
    - Shifts: General (09:00-18:00), Early, Night, Flexible
    - Departments: Engineering, Operations, Field Sales
    - Roles: Admin, Managers (Vikram, Priya), Employees (Aarav, Rohan, Sneha, Aditya)
    - Biometric consent records with timestamps
    - Paired attendance events (on-time, overtime, geofence violations, remote passes)
    - Correction requests
    """
    print("[*] Initializing database schema and default tables...")
    init_db()

    conn = get_db_connection()
    c = conn.cursor()
    now_iso = datetime.datetime.now().isoformat()
    today_str = datetime.date.today().isoformat()
    yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    print("[*] 1. Seeding / Updating Work Sites...")
    sites_data = [
        ("Mumbai Corporate HQ", "Bandra Kurla Complex, Mumbai, MH", 19.0657, 72.8687, 200, 1),
        ("Bengaluru Tech Hub", "Electronic City Phase 1, Bengaluru, KA", 12.8399, 77.6770, 250, 1),
        ("Remote & Field Operations", "Pan-India Field Locations", 20.5937, 78.9629, 500000, 0),
    ]
    site_ids = {}
    for name, addr, lat, lon, radius, enabled in sites_data:
        c.execute("SELECT id FROM sites WHERE name = ?", (name,))
        row = c.fetchone()
        if not row:
            c.execute("""
                INSERT INTO sites (name, address, latitude, longitude, geofence_radius_meters, geofencing_enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, addr, lat, lon, radius, enabled, now_iso))
            site_ids[name] = c.lastrowid
        else:
            site_ids[name] = row["id"]

    print("[*] 2. Seeding / Updating Shifts...")
    shifts_data = [
        ("Standard General Shift", "09:00", "18:00", 15, 60),
        ("Early Production Shift", "07:00", "16:00", 10, 60),
        ("Night Support Shift", "22:00", "06:00", 15, 45),
        ("Flexible Field Shift", "09:30", "18:30", 30, 60),
    ]
    shift_ids = {}
    for name, st, et, grace, brk in shifts_data:
        c.execute("SELECT id FROM shifts WHERE name = ?", (name,))
        row = c.fetchone()
        if not row:
            c.execute("""
                INSERT INTO shifts (name, start_time, end_time, grace_period_minutes, break_duration_minutes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, st, et, grace, brk, now_iso))
            shift_ids[name] = c.lastrowid
        else:
            shift_ids[name] = row["id"]

    print("[*] 3. Seeding Admin & Managers...")
    pwd_hash = generate_password_hash("password123")
    admin_pwd_hash = generate_password_hash("admin123")

    # Admin User
    c.execute("SELECT id FROM employees WHERE employee_code = 'ADM-001'")
    if not c.fetchone():
        c.execute("""
            INSERT INTO employees (name, employee_code, email, role, password_hash, biometric_consent, biometric_consent_timestamp, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("System Administrator", "ADM-001", "admin@enterprise.internal", "admin", admin_pwd_hash, 1, now_iso, 1, now_iso))

    # Manager 1: Vikram Sharma (Engineering)
    c.execute("SELECT id FROM employees WHERE employee_code = 'MGR-001'")
    mgr1_row = c.fetchone()
    if not mgr1_row:
        c.execute("""
            INSERT INTO employees (name, employee_code, email, role, password_hash, biometric_consent, biometric_consent_timestamp, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("Vikram Sharma", "MGR-001", "vikram.sharma@enterprise.internal", "manager", pwd_hash, 1, now_iso, 1, now_iso))
        mgr1_id = c.lastrowid
    else:
        mgr1_id = mgr1_row["id"]

    # Manager 2: Priya Patel (Operations & Logistics)
    c.execute("SELECT id FROM employees WHERE employee_code = 'MGR-002'")
    mgr2_row = c.fetchone()
    if not mgr2_row:
        c.execute("""
            INSERT INTO employees (name, employee_code, email, role, password_hash, biometric_consent, biometric_consent_timestamp, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("Priya Patel", "MGR-002", "priya.patel@enterprise.internal", "manager", pwd_hash, 1, now_iso, 1, now_iso))
        mgr2_id = c.lastrowid
    else:
        mgr2_id = mgr2_row["id"]

    print("[*] 4. Seeding / Updating Departments...")
    depts = [
        ("Engineering & Architecture", mgr1_id),
        ("Operations & Logistics", mgr2_id),
        ("Field Sales & Services", mgr2_id),
    ]
    dept_ids = {}
    for name, mgr_id in depts:
        c.execute("SELECT id FROM departments WHERE name = ?", (name,))
        d_row = c.fetchone()
        if not d_row:
            c.execute("INSERT INTO departments (name, manager_id, created_at) VALUES (?, ?, ?)", (name, mgr_id, now_iso))
            dept_ids[name] = c.lastrowid
        else:
            dept_ids[name] = d_row["id"]
            c.execute("UPDATE departments SET manager_id = ? WHERE id = ?", (mgr_id, d_row["id"]))

    print("[*] 5. Seeding Employees across Sites & Shifts...")
    employees_data = [
        ("Aarav Sharma", "EMP-001", "aarav@enterprise.internal", dept_ids["Engineering & Architecture"], site_ids["Mumbai Corporate HQ"], shift_ids["Standard General Shift"], mgr1_id),
        ("Aditya Verma", "EMP-002", "aditya@enterprise.internal", dept_ids["Engineering & Architecture"], site_ids["Mumbai Corporate HQ"], shift_ids["Standard General Shift"], mgr1_id),
        ("Rohan Deshmukh", "EMP-003", "rohan@enterprise.internal", dept_ids["Operations & Logistics"], site_ids["Bengaluru Tech Hub"], shift_ids["Standard General Shift"], mgr2_id),
        ("Sneha Kulkarni", "EMP-004", "sneha@enterprise.internal", dept_ids["Field Sales & Services"], site_ids["Remote & Field Operations"], shift_ids["Flexible Field Shift"], mgr2_id),
    ]
    emp_ids = {}
    for name, code, email, did, sid, shid, mid in employees_data:
        c.execute("SELECT id FROM employees WHERE employee_code = ?", (code,))
        erow = c.fetchone()
        if not erow:
            c.execute("""
                INSERT INTO employees (name, employee_code, email, department_id, site_id, shift_id, manager_id, role, password_hash, biometric_consent, biometric_consent_timestamp, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'employee', ?, 1, ?, 1, ?)
            """, (name, code, email, did, sid, shid, mid, pwd_hash, now_iso, now_iso))
            emp_ids[code] = c.lastrowid
        else:
            emp_ids[code] = erow["id"]
            c.execute("""
                UPDATE employees SET department_id=?, site_id=?, shift_id=?, manager_id=?, biometric_consent=1, biometric_consent_timestamp=?
                WHERE id=?
            """, (did, sid, shid, mid, now_iso, erow["id"]))

    conn.commit()

    print("[*] 6. Seeding Attendance Events & End-to-End Daily Cycles...")
    # Cycle 1: Aarav Sharma (EMP-001) - Full Day Cycle (Check-in 08:58 on-time, Check-out 19:15 with overtime)
    c.execute("SELECT id FROM attendance_events WHERE employee_id = ? AND date(timestamp) = ?", (emp_ids["EMP-001"], today_str))
    if not c.fetchone():
        t_in = f"{today_str}T08:58:12"
        t_out = f"{today_str}T19:15:45"
        # Check-in
        c.execute("""
            INSERT INTO attendance_events 
            (employee_id, event_type, timestamp, latitude, longitude, address, distance_from_site_meters, within_geofence, confidence, liveness_passed, status, flagged, flag_reason, created_at)
            VALUES (?, 'check_in', ?, 19.0659, 72.8688, 'BKC Gate 2, Mumbai HQ', 28.4, 1, 0.94, 1, 'on_time', 0, '', ?)
        """, (emp_ids["EMP-001"], t_in, t_in))
        # Check-out
        c.execute("""
            INSERT INTO attendance_events 
            (employee_id, event_type, timestamp, latitude, longitude, address, distance_from_site_meters, within_geofence, confidence, liveness_passed, status, flagged, flag_reason, created_at)
            VALUES (?, 'check_out', ?, 19.0658, 72.8686, 'BKC Exit 1, Mumbai HQ', 18.2, 1, 0.93, 1, 'overtime', 0, '', ?)
        """, (emp_ids["EMP-001"], t_out, t_out))

    # Cycle 2: Aditya Verma (EMP-002) - Geofence Violation (Attempted check-in 120km away from Mumbai HQ)
    c.execute("SELECT id FROM attendance_events WHERE employee_id = ? AND date(timestamp) = ?", (emp_ids["EMP-002"], today_str))
    if not c.fetchone():
        t_aditya = f"{today_str}T09:18:30"
        c.execute("""
            INSERT INTO attendance_events 
            (employee_id, event_type, timestamp, latitude, longitude, address, distance_from_site_meters, within_geofence, confidence, liveness_passed, status, flagged, flag_reason, created_at)
            VALUES (?, 'check_in', ?, 18.5204, 73.8567, 'FC Road, Pune, MH (Unauthorized location)', 124800.0, 0, 0.91, 1, 'flagged', 1, 'Geofence violation: 124800m exceeds radius 200m', ?)
        """, (emp_ids["EMP-002"], t_aditya, t_aditya))

    # Cycle 3: Sneha Kulkarni (EMP-004) - Remote Worker (Geofencing disabled, accepted with coordinates logged)
    c.execute("SELECT id FROM attendance_events WHERE employee_id = ? AND date(timestamp) = ?", (emp_ids["EMP-004"], today_str))
    if not c.fetchone():
        t_sneha_in = f"{today_str}T09:20:00"
        t_sneha_out = f"{today_str}T18:35:10"
        c.execute("""
            INSERT INTO attendance_events 
            (employee_id, event_type, timestamp, latitude, longitude, address, distance_from_site_meters, within_geofence, confidence, liveness_passed, status, flagged, flag_reason, created_at)
            VALUES (?, 'check_in', ?, 28.6139, 77.2090, 'Connaught Place Client Office, Delhi', 0.0, 1, 0.96, 1, 'on_time', 0, '', ?)
        """, (emp_ids["EMP-004"], t_sneha_in, t_sneha_in))
        c.execute("""
            INSERT INTO attendance_events 
            (employee_id, event_type, timestamp, latitude, longitude, address, distance_from_site_meters, within_geofence, confidence, liveness_passed, status, flagged, flag_reason, created_at)
            VALUES (?, 'check_out', ?, 28.6139, 77.2090, 'Connaught Place Client Office, Delhi', 0.0, 1, 0.95, 1, 'on_time', 0, '', ?)
        """, (emp_ids["EMP-004"], t_sneha_out, t_sneha_out))

    # Cycle 4: Rohan Deshmukh (EMP-003) - Pending Correction Request
    c.execute("SELECT id FROM correction_requests WHERE employee_id = ?", (emp_ids["EMP-003"],))
    if not c.fetchone():
        c.execute("""
            INSERT INTO correction_requests (employee_id, attendance_date, reason, requested_change, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
        """, (emp_ids["EMP-003"], yesterday_str, "Forgot to check out due to client emergency outage at data center.", "Set Check-Out time to 18:30 PM", now_iso))

    conn.commit()

    # Log initial compliance and audit events
    log_audit(c, 1, "admin", "SEED_ENTERPRISE_SYSTEM", "system", "initial_setup", "Enterprise data successfully seeded with sites, shifts, employees, and full cycles.")
    conn.commit()
    conn.close()

    print("[SUCCESS] Enterprise database seeded successfully!")
    print("----------------------------------------------------------------------")
    print("User Credentials for Testing:")
    print("1. Admin:    Username: admin          | Password: admin123")
    print("2. Manager:  Email: vikram.sharma@... | Password: password123 (Engineering)")
    print("3. Manager:  Email: priya.patel@...   | Password: password123 (Operations/Sales)")
    print("4. Employee: Email: aarav@...         | Password: password123 (Mumbai HQ)")
    print("5. Employee: Email: rohan@...         | Password: password123 (Bengaluru Hub)")
    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    seed_enterprise_data()
