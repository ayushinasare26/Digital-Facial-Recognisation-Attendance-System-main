"""
blueprints/employee/routes.py - Enterprise Employee Portal Routes
Features:
- Mobile-first Check-In / Check-Out with live camera, liveness, and GPS
- Dynamic status display ('Not checked in', 'Checked in at HH:MM', 'Checked out at HH:MM')
- Pre-submission geofence verification warning
- Punctuality & Overtime personal attendance history
- Submission of attendance correction requests
"""

import os
import datetime
from flask import (
    render_template, request, jsonify, redirect,
    url_for, session, flash, abort, current_app
)
from . import employee_bp
from core.db import get_db_connection, log_audit
from core.auth import (
    login_required_employee,
    authenticate_employee,
    rate_limit
)
from core.pipeline import run_attendance_pipeline
from core.shift_engine import calculate_shift_hours, evaluate_check_in, evaluate_check_out
from config import Config, get_utc_now, get_local_now, format_local_timestamp

@employee_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Employee Portal Login:
    Accepts Employee Code, ID, or Email with password.
    Includes quick demo employee shortcuts for testing.
    """
    if session.get("role") in ("employee", "user", "manager") and (session.get("employee_id") or session.get("student_id")):
        return redirect(url_for("employee.check_in_out"))

    error = None
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "").strip()
        demo_id = request.form.get("demo_employee_id") or request.form.get("demo_student_id")

        if demo_id:
            user = authenticate_employee(demo_id, is_demo=True)
        else:
            user = authenticate_employee(identifier, password)

        if user:
            session.clear()
            user_role = user.get("role") or "employee"
            session["role"] = user_role
            session["employee_id"] = user["id"]
            session["student_id"] = user["id"] # Backward compatibility
            session["employee_name"] = user["name"]
            session["student_name"] = user["name"]
            session["employee_code"] = user.get("employee_code") or f"EMP-{user['id']:04d}"
            session["roll"] = session["employee_code"]
            session["department_id"] = user.get("department_id")
            session["site_id"] = user.get("site_id")
            session["shift_id"] = user.get("shift_id")
            session["department_name"] = user.get("department_name") or "General"
            session["site_name"] = user.get("site_name") or "Headquarters"
            session["shift_name"] = user.get("shift_name") or "General Shift"

            if user_role == "manager":
                session["manager_id"] = user["id"]
                return redirect(url_for("manager.dashboard"))
            return redirect(url_for("employee.check_in_out"))
        else:
            error = "Invalid credentials. Please verify your Employee Code/Email and password."

    # Fetch 4 demo employees for 1-click evaluation
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT e.id, e.name, e.employee_code, d.name AS department_name, s.name AS site_name, sh.name AS shift_name
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN sites s ON e.site_id = s.id
        LEFT JOIN shifts sh ON e.shift_id = sh.id
        WHERE e.active = 1
        ORDER BY e.id LIMIT 4
    """)
    demo_employees = c.fetchall()
    conn.close()

    return render_template("employee/login.html", error=error, demo_employees=demo_employees)

@employee_bp.route("/logout")
def logout():
    emp_id = session.get("employee_id") or session.get("student_id")
    if emp_id:
        log_audit(emp_id, session.get("role", "employee"), "LOGOUT", "auth", emp_id, "Employee logged out")
    session.clear()
    flash("You have been signed out of the employee portal.", "info")
    return redirect(url_for("employee.login"))

@employee_bp.route("/check-in-out", methods=["GET"])
@login_required_employee
def check_in_out():
    """
    Mobile-first Check-In / Check-Out View:
    - Displays current status: 'Not checked in', 'Checked in at 9:04 AM', 'Checked out at 6:12 PM'
    - Single toggle action button for Check-In or Check-Out
    - Live webcam, liveness challenge, and GPS capture
    - Provides assigned site coordinates and geofence radius for pre-submission warning
    """
    emp_id = session.get("employee_id") or session.get("student_id")
    emp_name = session.get("employee_name") or session.get("student_name", "Employee")
    emp_code = session.get("employee_code") or session.get("roll", f"EMP-{emp_id}")

    conn = get_db_connection()
    c = conn.cursor()

    # 1. Fetch employee's assigned site and shift metadata
    c.execute("""
        SELECT e.id, e.name, e.site_id, e.shift_id,
               s.name AS site_name, s.address AS site_address, s.latitude AS site_lat, s.longitude AS site_lon,
               s.geofence_radius_meters, s.geofencing_enabled,
               sh.name AS shift_name, sh.start_time, sh.end_time, sh.grace_period_minutes, sh.break_duration_minutes
        FROM employees e
        LEFT JOIN sites s ON e.site_id = s.id
        LEFT JOIN shifts sh ON e.shift_id = sh.id
        WHERE e.id = ?
    """, (emp_id,))
    emp_data = c.fetchone()

    # 2. Fetch today's check-in / check-out events
    today_date = get_local_now().date().isoformat()
    c.execute("""
        SELECT id, event_type, timestamp, latitude, longitude, address,
               distance_from_site_meters, within_geofence, confidence, status, flagged, flag_reason, geotagged_photo_path
        FROM attendance_events
        WHERE employee_id = ? AND date(timestamp) = ?
        ORDER BY timestamp ASC
    """, (emp_id, today_date))
    events_today = [dict(r) for r in c.fetchall()]
    
    # Fallback to legacy attendance table if attendance_events is empty
    if not events_today:
        c.execute("""
            SELECT id, 'check_in' as event_type, timestamp, latitude, longitude, address,
                   confidence, status, geotagged_photo_path
            FROM attendance
            WHERE student_id = ? AND date(timestamp) = ?
            ORDER BY timestamp ASC
        """, (emp_id, today_date))
        events_today = [dict(r) for r in c.fetchall()]

    conn.close()

    # Determine current status and next recommended action
    has_checkin = False
    has_checkout = False
    checkin_time_str = None
    checkout_time_str = None
    last_status = "Not checked in"
    next_action = "check_in"

    for ev in events_today:
        ev_type = ev.get("event_type", "check_in")
        ev["time_display"] = format_local_timestamp(ev["timestamp"], include_year=False)
        if ev_type == "check_in":
            has_checkin = True
            checkin_time_str = ev["time_display"]
        elif ev_type == "check_out":
            has_checkout = True
            checkout_time_str = ev["time_display"]

    if has_checkin and not has_checkout:
        last_status = f"Checked in at {checkin_time_str}"
        next_action = "check_out"
    elif has_checkin and has_checkout:
        last_status = f"Completed workday (Checked out at {checkout_time_str})"
        next_action = "check_in" # Allows re-entry or additional shift
    else:
        last_status = "Not checked in today"
        next_action = "check_in"

    site_info = {
        "name": emp_data["site_name"] if emp_data and emp_data["site_name"] else "Headquarters",
        "address": emp_data["site_address"] if emp_data and emp_data["site_address"] else "Main Office",
        "latitude": emp_data["site_lat"] if emp_data and emp_data["site_lat"] is not None else 19.0657,
        "longitude": emp_data["site_lon"] if emp_data and emp_data["site_lon"] is not None else 72.8687,
        "geofence_radius_meters": emp_data["geofence_radius_meters"] if emp_data and emp_data["geofence_radius_meters"] is not None else 200.0,
        "geofencing_enabled": bool(emp_data["geofencing_enabled"]) if emp_data and emp_data["geofencing_enabled"] is not None else True
    }

    shift_info = {
        "name": emp_data["shift_name"] if emp_data and emp_data["shift_name"] else "General Shift",
        "start_time": emp_data["start_time"] if emp_data and emp_data["start_time"] else "09:00",
        "end_time": emp_data["end_time"] if emp_data and emp_data["end_time"] else "18:00",
        "grace_period_minutes": emp_data["grace_period_minutes"] if emp_data and emp_data["grace_period_minutes"] is not None else 15
    }

    return render_template(
        "employee/check_in_out.html",
        employee_id=emp_id,
        employee_name=emp_name,
        employee_code=emp_code,
        current_status=last_status,
        next_action=next_action,
        has_checkin=has_checkin,
        has_checkout=has_checkout,
        events_today=events_today,
        site=site_info,
        shift=shift_info
    )

@employee_bp.route("/", methods=["GET"])
@employee_bp.route("/mark-attendance", methods=["GET"])
@login_required_employee
def mark_attendance_alias():
    return check_in_out()

@employee_bp.route("/check-in", methods=["POST"])
@employee_bp.route("/api/check-in", methods=["POST"])
@employee_bp.route("/mark-attendance", methods=["POST"])
@login_required_employee
@rate_limit(max_requests=20, window_seconds=60, scope="emp_checkin")
def api_check_in():
    """Processes check-in event via live camera + liveness + GPS."""
    return _process_attendance_event(event_type="check_in")

@employee_bp.route("/check-out", methods=["POST"])
@employee_bp.route("/api/check-out", methods=["POST"])
@login_required_employee
@rate_limit(max_requests=20, window_seconds=60, scope="emp_checkout")
def api_check_out():
    """Processes check-out event via live camera + liveness + GPS."""
    return _process_attendance_event(event_type="check_out")

def _process_attendance_event(event_type="check_in"):
    session_emp_id = session.get("employee_id") or session.get("student_id")
    session_emp_name = session.get("employee_name") or session.get("student_name")

    primary_image = None
    lat = None
    lon = None
    challenge_type = "blink"
    is_demo = False
    liveness_frames = []
    bypass_cooldown = False

    # Extract multipart
    if request.files:
        if "image" in request.files:
            primary_image = request.files["image"].read()
        elif "photo" in request.files:
            primary_image = request.files["photo"].read()
        for lf in request.files.getlist("liveness_frames[]"):
            liveness_frames.append(lf.read())

    if request.form:
        lat = request.form.get("latitude")
        lon = request.form.get("longitude")
        challenge_type = request.form.get("challenge_type", "blink")
        is_demo = request.form.get("is_demo", "false").lower() in ("true", "1", "yes")
        bypass_cooldown = request.form.get("bypass_cooldown", "false").lower() in ("true", "1", "yes")
        form_event_type = request.form.get("event_type")
        if form_event_type:
            event_type = form_event_type

    # Extract JSON
    if not primary_image and request.is_json:
        data = request.get_json() or {}
        primary_image = data.get("image") or data.get("photo")
        lat = data.get("latitude")
        lon = data.get("longitude")
        challenge_type = data.get("challenge_type", "blink")
        is_demo = bool(data.get("is_demo", False))
        bypass_cooldown = bool(data.get("bypass_cooldown", False))
        liveness_frames = data.get("liveness_frames") or []
        if data.get("event_type"):
            event_type = data.get("event_type")

    if lat in ("", "null", "undefined"): lat = None
    if lon in ("", "null", "undefined"): lon = None

    # Execute Enterprise Pipeline
    result = run_attendance_pipeline(
        primary_image_data=primary_image,
        latitude=lat,
        longitude=lon,
        liveness_frames=liveness_frames,
        challenge_type=challenge_type,
        is_demo=is_demo,
        bypass_cooldown=bypass_cooldown,
        event_type=event_type,
        expected_employee_id=session_emp_id
    )

    if not result.get("success"):
        return jsonify(result), 400

    # Authorization Check: Biometric Proxy Defense
    matched_id = result.get("student_id") or result.get("employee_id")
    if session_emp_id and matched_id != session_emp_id:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE attendance
            SET status = 'flagged',
                review_note = ?
            WHERE id = ?
        """, (f"Biometric Proxy Attempt: Logged-in session {session_emp_name} (#{session_emp_id}) but face matched {result.get('student_name')} (#{matched_id}).", result.get("attendance_id")))
        c.execute("""
            UPDATE attendance_events
            SET status = 'flagged',
                flagged = 1,
                flag_reason = ?
            WHERE id = ?
        """, (f"Biometric Proxy Attempt: Session was {session_emp_name} (#{session_emp_id}) but face matched #{matched_id}", result.get("event_id")))
        conn.commit()
        conn.close()

        log_audit(session_emp_id, "employee", "PROXY_VIOLATION", "attendance_event", result.get("event_id"), f"Face matched ID #{matched_id}")

        return jsonify({
            "success": False,
            "error_stage": "Identity Verification",
            "message": f"Biometric mismatch: The face recognized belongs to {result.get('student_name')}, not your active account ({session_emp_name}). Proxy attendance is prohibited.",
            "stages": result.get("stages", [])
        }), 403

    return jsonify(result), 200

@employee_bp.route("/my-attendance", methods=["GET"])
@employee_bp.route("/api/employee/my-attendance", methods=["GET"])
@login_required_employee
def my_attendance():
    """
    Employee Portal: Scoped Personal Attendance History & Analytics.
    STRICT SECURITY: Every query is hard-scoped to session['employee_id'].
    Consolidates check-in & check-out pairs per day with regular hours, overtime, and punctuality.
    """
    emp_id = session.get("employee_id") or session.get("student_id")
    emp_name = session.get("employee_name") or session.get("student_name")
    from_date = request.args.get("from")
    to_date = request.args.get("to")
    status_filter = request.args.get("status", "all")

    conn = get_db_connection()
    c = conn.cursor()

    # 1. Fetch employee's shift
    c.execute("""
        SELECT sh.name, sh.start_time, sh.end_time, sh.grace_period_minutes, sh.break_duration_minutes
        FROM employees e
        LEFT JOIN shifts sh ON e.shift_id = sh.id
        WHERE e.id = ?
    """, (emp_id,))
    sh_row = c.fetchone()
    shift = {
        "name": sh_row["name"] if sh_row else "General Shift",
        "start_time": sh_row["start_time"] if sh_row else "09:00",
        "end_time": sh_row["end_time"] if sh_row else "18:00",
        "grace_period_minutes": sh_row["grace_period_minutes"] if sh_row else 15,
        "break_duration_minutes": sh_row["break_duration_minutes"] if sh_row else 60
    }

    # 2. Fetch all events for this employee
    query = """
        SELECT id, event_type, timestamp, latitude, longitude, address,
               distance_from_site_meters, within_geofence, confidence,
               status, flagged, flag_reason, geotagged_photo_path
        FROM attendance_events
        WHERE employee_id = ?
    """
    params = [emp_id]

    if from_date:
        query += " AND date(timestamp) >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date(timestamp) <= ?"
        params.append(to_date)
    if status_filter in ("on_time", "late", "early_leave", "overtime", "flagged"):
        query += " AND (status = ? OR (flagged = 1 AND ? = 'flagged'))"
        params.extend([status_filter, status_filter])

    query += " ORDER BY timestamp DESC LIMIT 300"
    c.execute(query, tuple(params))
    events = [dict(r) for r in c.fetchall()]

    # Fallback to legacy attendance table if no events in attendance_events
    if not events:
        c.execute("""
            SELECT id, 'check_in' as event_type, timestamp, latitude, longitude, address,
                   confidence, status, geotagged_photo_path,
                   distance_from_site_meters, within_geofence, flagged, flag_reason
            FROM attendance
            WHERE student_id = ?
            ORDER BY timestamp DESC LIMIT 300
        """, (emp_id,))
        events = [dict(r) for r in c.fetchall()]

    # 3. Group events into daily logs (Pair check-in & check-out)
    days_map = {}
    for ev in events:
        dt_val = ev["timestamp"]
        try:
            day_key = dt_val.split("T")[0]
        except Exception:
            day_key = dt_val[:10]

        if day_key not in days_map:
            days_map[day_key] = {"check_ins": [], "check_outs": []}

        if ev["event_type"] == "check_out":
            days_map[day_key]["check_outs"].append(ev)
        else:
            days_map[day_key]["check_ins"].append(ev)

    daily_records = []
    total_hours_sum = 0.0
    total_overtime_sum = 0.0
    total_late_days = 0
    total_flagged_days = 0
    total_present_days = 0
    is_currently_checked_in = False
    today_active_hours = 0.0

    local_now = get_local_now()
    local_today_str = local_now.date().isoformat()

    for day_str in sorted(days_map.keys(), reverse=True):
        day_data = days_map[day_str]
        check_ins = sorted(day_data["check_ins"], key=lambda x: x["timestamp"])
        check_outs = sorted(day_data["check_outs"], key=lambda x: x["timestamp"])

        first_in = check_ins[0] if check_ins else None
        last_out = check_outs[-1] if check_outs else None

        in_ts = first_in["timestamp"] if first_in else None
        out_ts = last_out["timestamp"] if last_out else None

        is_active_today = (day_str == local_today_str and first_in is not None and last_out is None)
        if is_active_today:
            is_currently_checked_in = True

        hours_calc = calculate_shift_hours(in_ts, out_ts, shift, allow_in_progress=is_active_today)
        
        reg_h = hours_calc.get("regular_hours", 0.0)
        ot_h = hours_calc.get("overtime_hours", 0.0)
        net_h = hours_calc.get("net_hours", 0.0)

        is_present = True if first_in else False
        if is_present:
            total_present_days += 1

        in_eval = evaluate_check_in(in_ts, shift) if in_ts else None
        if in_eval and in_eval["status"] == "late":
            punctuality = "late"
            total_late_days += 1
        elif last_out:
            out_eval = evaluate_check_out(out_ts, shift)
            if out_eval["status"] == "overtime":
                punctuality = "overtime"
            elif out_eval["status"] == "early_leave":
                punctuality = "early_leave"
            else:
                punctuality = "on_time"
        else:
            punctuality = in_eval["status"] if in_eval else "on_time"

        is_flagged = any(ev.get("flagged") for ev in (check_ins + check_outs))
        flag_reason = "; ".join([ev["flag_reason"] for ev in (check_ins + check_outs) if ev.get("flag_reason")])

        if is_flagged:
            total_flagged_days += 1

        if is_active_today:
            today_active_hours = net_h

        total_hours_sum += net_h
        total_overtime_sum += ot_h

        daily_records.append({
            "date": day_str,
            "check_in_time": format_local_timestamp(in_ts, include_year=False) if in_ts else "—",
            "check_out_time": format_local_timestamp(out_ts, include_year=False) if out_ts else "—",
            "hours_worked": net_h,
            "regular_hours": reg_h,
            "overtime_hours": ot_h,
            "status": punctuality,
            "punctuality": punctuality,
            "is_active_today": is_active_today,
            "is_present": is_present,
            "flagged": is_flagged,
            "flag_reason": flag_reason or "—",
            "check_in_id": first_in["id"] if first_in else None,
            "check_out_id": last_out["id"] if last_out else None,
            "photo_path": first_in["geotagged_photo_path"] if first_in else (last_out["geotagged_photo_path"] if last_out else None),
            "address": first_in["address"] if first_in else (last_out["address"] if last_out else "—")
        })

    # Fetch correction requests for this employee
    c.execute("""
        SELECT id, attendance_date, reason, requested_change, status, reviewed_by, review_note, created_at
        FROM correction_requests
        WHERE employee_id = ?
        ORDER BY created_at DESC LIMIT 20
    """, (emp_id,))
    correction_requests = [dict(r) for r in c.fetchall()]

    conn.close()

    if request.is_json or request.args.get("format") == "json":
        return jsonify({
            "daily_records": daily_records,
            "total_hours": round(total_hours_sum, 1),
            "total_overtime": round(total_overtime_sum, 1),
            "total_late_days": total_late_days,
            "total_days_worked": total_present_days,
            "is_currently_checked_in": is_currently_checked_in
        })

    return render_template(
        "employee/my_attendance.html",
        records=daily_records,
        total_hours=round(total_hours_sum, 1),
        total_overtime=round(total_overtime_sum, 1),
        total_late_days=total_late_days,
        total_flagged_days=total_flagged_days,
        total_days_worked=total_present_days,
        is_currently_checked_in=is_currently_checked_in,
        today_active_hours=round(today_active_hours, 1),
        from_date=from_date or "",
        to_date=to_date or "",
        status_filter=status_filter,
        correction_requests=correction_requests
    )

@employee_bp.route("/correction-request", methods=["POST"])
@employee_bp.route("/api/employee/correction-request", methods=["POST"])
@login_required_employee
def submit_correction_request():
    """
    Submits a correction request for a missed/failed check-in or out.
    Enforces server-side employee ownership.
    """
    emp_id = session.get("employee_id") or session.get("student_id")
    
    date_val = request.form.get("attendance_date") or (request.get_json() or {}).get("attendance_date")
    reason = request.form.get("reason") or (request.get_json() or {}).get("reason")
    change = request.form.get("requested_change") or (request.get_json() or {}).get("requested_change")

    if not date_val or not reason or not change:
        if request.is_json:
            return jsonify({"success": False, "error": "Missing attendance date, reason, or requested correction details."}), 400
        flash("Please provide the date, reason, and requested correction.", "danger")
        return redirect(url_for("employee.my_attendance"))

    conn = get_db_connection()
    c = conn.cursor()
    now = get_utc_now().isoformat()
    c.execute("""
        INSERT INTO correction_requests (employee_id, attendance_date, reason, requested_change, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
    """, (emp_id, date_val, reason.strip(), change.strip(), now))
    req_id = c.lastrowid
    conn.commit()
    conn.close()

    log_audit(emp_id, "employee", "SUBMIT_CORRECTION", "correction_request", req_id, f"Date: {date_val}")

    if request.is_json:
        return jsonify({"success": True, "request_id": req_id, "message": "Correction request submitted to manager."}), 201

    flash("Correction request submitted to your manager for approval.", "success")
    return redirect(url_for("employee.my_attendance"))

@employee_bp.route("/my-attendance/<int:record_id>", methods=["GET"])
@login_required_employee
def api_record_detail(record_id):
    """Single event detail with ownership check."""
    emp_id = session.get("employee_id") or session.get("student_id")
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM attendance_events WHERE id = ?", (record_id,))
    ev_row = c.fetchone()
    if ev_row:
        if ev_row["employee_id"] != emp_id:
            conn.close()
            return jsonify({"error": "Attendance record not found or access denied"}), 404
        row = ev_row
    else:
        c.execute("SELECT * FROM attendance WHERE id = ? AND student_id = ?", (record_id, emp_id))
        row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Attendance record not found or access denied"}), 404

    data = dict(row)
    data["time_display"] = format_local_timestamp(data["timestamp"], include_year=True)
    data["photo_url"] = f"/{data['geotagged_photo_path']}" if data.get("geotagged_photo_path") else None
    return jsonify(data)
