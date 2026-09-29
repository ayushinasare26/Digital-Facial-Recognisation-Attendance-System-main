"""
blueprints/manager/routes.py - Enterprise Manager Portal Routes
Scoped strictly to employees within the manager's department or supervisees.
Features:
- Team Dashboard: Real-time team status (Checked-in, Not Checked-in, Late, Absent)
- Team Attendance Records with date range & employee filtering
- Correction Requests Queue with server-side authorized Approve/Reject workflow
- Compliance: Restricts precise raw GPS coordinates and raw photos from manager view
"""

import datetime
from flask import (
    render_template, request, jsonify, redirect,
    url_for, session, flash, abort
)
from . import manager_bp
from core.db import get_db_connection, log_audit
from core.auth import (
    login_required_manager,
    get_manager_team_scope,
    is_employee_in_manager_scope
)
from core.shift_engine import calculate_shift_hours
from config import get_local_now, format_local_timestamp

@manager_bp.route("/login", methods=["GET", "POST"])
def login():
    """Manager login redirect to unified authentication portal."""
    if session.get("role") in ("manager", "admin") and (session.get("manager_id") or session.get("admin_id")):
        return redirect(url_for("manager.dashboard"))
    return redirect(url_for("employee.login", portal="manager"))

@manager_bp.route("/logout")
def logout():
    """Manager logout."""
    mgr_id = session.get("manager_id")
    if mgr_id:
        log_audit(mgr_id, "manager", "LOGOUT", "auth", mgr_id, "Manager logged out")
    session.clear()
    flash("You have been signed out of the Manager Portal.", "info")
    return redirect(url_for("employee.login", portal="manager"))

@manager_bp.route("/")
@manager_bp.route("/dashboard")
@login_required_manager
def dashboard():
    """
    Manager Team Dashboard:
    Real-time status for the manager's team today:
    - Checked in
    - Not yet checked in
    - Late
    - Absent
    """
    manager_id = session.get("manager_id") or session.get("employee_id") or session.get("admin_id")
    dept_filter = request.args.get("department_id")
    shift_filter = request.args.get("shift_id")

    scope = get_manager_team_scope(manager_id)
    emp_ids = scope["employee_ids"]

    if not emp_ids:
        return render_template(
            "manager/dashboard.html",
            team_members=[],
            total_team=0,
            checked_in_count=0,
            not_checked_in_count=0,
            late_count=0,
            departments=[],
            shifts=[]
        )

    conn = get_db_connection()
    c = conn.cursor()

    # Fetch departments and shifts managed by this manager
    placeholders = ",".join("?" for _ in emp_ids)
    c.execute(f"""
        SELECT DISTINCT d.id, d.name
        FROM departments d
        JOIN employees e ON e.department_id = d.id
        WHERE e.id IN ({placeholders})
    """, tuple(emp_ids))
    departments = c.fetchall()

    c.execute("SELECT id, name FROM shifts")
    shifts = c.fetchall()

    # Build query for team members
    emp_query = f"""
        SELECT e.id, e.name, e.employee_code, e.department_id, e.site_id, e.shift_id,
               d.name AS department_name, s.name AS site_name, sh.name AS shift_name,
               sh.start_time, sh.end_time, sh.grace_period_minutes, sh.break_duration_minutes
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN sites s ON e.site_id = s.id
        LEFT JOIN shifts sh ON e.shift_id = sh.id
        WHERE e.id IN ({placeholders}) AND e.active = 1
    """
    params = list(emp_ids)

    if dept_filter:
        emp_query += " AND e.department_id = ?"
        params.append(dept_filter)
    if shift_filter:
        emp_query += " AND e.shift_id = ?"
        params.append(shift_filter)

    emp_query += " ORDER BY e.name ASC"
    c.execute(emp_query, tuple(params))
    employees = [dict(r) for r in c.fetchall()]

    today_date = get_local_now().date().isoformat()
    team_status_list = []
    checked_in_count = 0
    not_checked_in_count = 0
    late_count = 0

    for emp in employees:
        eid = emp["id"]
        # Fetch today's check-in & check-out events
        c.execute("""
            SELECT id, event_type, timestamp, status, within_geofence, flagged
            FROM attendance_events
            WHERE employee_id = ? AND date(timestamp) = ?
            ORDER BY timestamp ASC
        """, (eid, today_date))
        today_evs = [dict(r) for r in c.fetchall()]

        # Fallback to legacy attendance table if needed
        if not today_evs:
            c.execute("""
                SELECT id, 'check_in' AS event_type, timestamp, status, within_geofence, flagged
                FROM attendance
                WHERE student_id = ? AND date(timestamp) = ?
                ORDER BY timestamp ASC
            """, (eid, today_date))
            today_evs = [dict(r) for r in c.fetchall()]

        first_in = next((ev for ev in today_evs if ev["event_type"] == "check_in"), None)
        last_out = next((ev for ev in reversed(today_evs) if ev["event_type"] == "check_out"), None)

        if first_in:
            checked_in_count += 1
            in_time_str = format_local_timestamp(first_in["timestamp"], include_year=False)
            status_val = first_in.get("status") or "on_time"
            if status_val == "late":
                late_count += 1
        else:
            not_checked_in_count += 1
            in_time_str = "—"
            status_val = "not_checked_in"

        out_time_str = format_local_timestamp(last_out["timestamp"], include_year=False) if last_out else "—"

        team_status_list.append({
            "id": eid,
            "name": emp["name"],
            "code": emp["employee_code"],
            "department": emp["department_name"] or "General",
            "shift": emp["shift_name"] or "General Shift",
            "site": emp["site_name"] or "Headquarters",
            "status": status_val,
            "check_in_time": in_time_str,
            "check_out_time": out_time_str,
            "within_geofence": first_in["within_geofence"] if first_in else True,
            "flagged": any(ev.get("flagged") for ev in today_evs)
        })

    # Fetch count of pending correction requests from team
    c.execute(f"""
        SELECT COUNT(*) FROM correction_requests
        WHERE employee_id IN ({placeholders}) AND status = 'pending'
    """, tuple(emp_ids))
    pending_corrections_count = c.fetchone()[0] or 0

    conn.close()

    return render_template(
        "manager/dashboard.html",
        team_members=team_status_list,
        total_team=len(employees),
        checked_in_count=checked_in_count,
        not_checked_in_count=not_checked_in_count,
        late_count=late_count,
        pending_corrections=pending_corrections_count,
        departments=departments,
        shifts=shifts,
        selected_dept=dept_filter or "",
        selected_shift=shift_filter or ""
    )

@manager_bp.route("/attendance", methods=["GET"])
@manager_bp.route("/attendance-records", methods=["GET"])
@manager_bp.route("/api/manager/team-attendance", methods=["GET"])
@login_required_manager
def team_attendance():
    """
    Team Attendance Records:
    Strictly scoped to manager's team.
    Filters: date range, specific employee, status.
    Compliance: Sanitizes raw GPS coordinates and hides raw photos.
    """
    manager_id = session.get("manager_id") or session.get("employee_id") or session.get("admin_id")
    scope = get_manager_team_scope(manager_id)
    emp_ids = scope["employee_ids"]

    from_date = request.args.get("from")
    to_date = request.args.get("to")
    filter_emp_id = request.args.get("employee_id")
    status_filter = request.args.get("status", "all")

    if not emp_ids:
        if request.is_json or request.args.get("format") == "json":
            return jsonify({"records": [], "team_members": []})
        return render_template("manager/attendance_records.html", records=[], team_members=[])

    # Server-Side Scope Verification: If filter_emp_id is provided, verify it belongs to manager
    if filter_emp_id:
        if not is_employee_in_manager_scope(manager_id, int(filter_emp_id)):
            abort(403, "Access denied: Target employee is not in your managed team.")
        query_ids = [int(filter_emp_id)]
    else:
        query_ids = emp_ids

    conn = get_db_connection()
    c = conn.cursor()

    # Fetch team members list for dropdown
    placeholders = ",".join("?" for _ in emp_ids)
    c.execute(f"SELECT id, name, employee_code FROM employees WHERE id IN ({placeholders}) ORDER BY name", tuple(emp_ids))
    team_members = [dict(r) for r in c.fetchall()]

    # Fetch attendance events for query_ids
    q_placeholders = ",".join("?" for _ in query_ids)
    query = f"""
        SELECT a.id, a.employee_id, a.event_type, a.timestamp, a.address,
               a.distance_from_site_meters, a.within_geofence, a.status, a.flagged, a.flag_reason,
               e.name, e.employee_code, d.name AS department_name, s.name AS site_name
        FROM attendance_events a
        JOIN employees e ON a.employee_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN sites s ON a.site_id = s.id
        WHERE a.employee_id IN ({q_placeholders})
    """
    params = list(query_ids)

    if from_date:
        query += " AND date(a.timestamp) >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date(a.timestamp) <= ?"
        params.append(to_date)
    if status_filter in ("on_time", "late", "early_leave", "overtime", "flagged"):
        query += " AND (a.status = ? OR (a.flagged = 1 AND ? = 'flagged'))"
        params.extend([status_filter, status_filter])

    query += " ORDER BY a.timestamp DESC LIMIT 500"
    c.execute(query, tuple(params))
    raw_events = [dict(r) for r in c.fetchall()]

    # Fallback to legacy attendance table if empty
    if not raw_events:
        query_legacy = f"""
            SELECT a.id, a.student_id AS employee_id, 'check_in' AS event_type, a.timestamp, a.address,
                   a.distance_from_site_meters, a.within_geofence, a.status, a.flagged, a.flag_reason,
                   e.name, e.employee_code, d.name AS department_name, 'Headquarters' AS site_name
            FROM attendance a
            JOIN employees e ON a.student_id = e.id
            LEFT JOIN departments d ON e.department_id = d.id
            WHERE a.student_id IN ({q_placeholders})
            ORDER BY a.timestamp DESC LIMIT 500
        """
        c.execute(query_legacy, tuple(query_ids))
        raw_events = [dict(r) for r in c.fetchall()]

    conn.close()

    formatted_events = []
    for ev in raw_events:
        formatted_events.append({
            "id": ev["id"],
            "employee_id": ev["employee_id"],
            "name": ev["name"],
            "code": ev["employee_code"],
            "department": ev["department_name"] or "General",
            "site": ev["site_name"] or "Headquarters",
            "event_type": ev.get("event_type", "check_in"),
            "time_display": format_local_timestamp(ev["timestamp"], include_year=True),
            "status": ev.get("status") or "on_time",
            "within_geofence": bool(ev.get("within_geofence", 1)),
            "distance_meters": ev.get("distance_from_site_meters"),
            "flagged": bool(ev.get("flagged", 0)),
            "flag_reason": ev.get("flag_reason") or "—",
            # Compliance: Location resolved to general area/address, exact lat/lon omitted from manager view
            "location_display": ev.get("address") or "Location Unavailable"
        })

    if request.is_json or request.args.get("format") == "json":
        return jsonify({
            "records": formatted_events,
            "team_members": team_members
        })

    return render_template(
        "manager/attendance_records.html",
        records=formatted_events,
        team_members=team_members,
        selected_emp=filter_emp_id or "",
        from_date=from_date or "",
        to_date=to_date or "",
        status_filter=status_filter
    )

@manager_bp.route("/correction-requests", methods=["GET"])
@manager_bp.route("/api/manager/correction-requests", methods=["GET"])
@login_required_manager
def correction_requests():
    """
    Correction Requests Queue:
    Displays pending and historical correction requests strictly from manager's team.
    """
    manager_id = session.get("manager_id") or session.get("employee_id") or session.get("admin_id")
    scope = get_manager_team_scope(manager_id)
    emp_ids = scope["employee_ids"]

    if not emp_ids:
        if request.is_json or request.args.get("format") == "json":
            return jsonify({"requests": []})
        return render_template("manager/correction_requests.html", requests=[])

    conn = get_db_connection()
    c = conn.cursor()
    placeholders = ",".join("?" for _ in emp_ids)
    c.execute(f"""
        SELECT cr.id, cr.employee_id, cr.attendance_date, cr.reason, cr.requested_change,
               cr.status, cr.reviewed_by, cr.review_note, cr.reviewed_at, cr.created_at,
               e.name, e.employee_code, d.name AS department_name
        FROM correction_requests cr
        JOIN employees e ON cr.employee_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE cr.employee_id IN ({placeholders})
        ORDER BY cr.created_at DESC LIMIT 200
    """, tuple(emp_ids))
    requests_list = [dict(r) for r in c.fetchall()]
    conn.close()

    for r in requests_list:
        r["created_display"] = format_local_timestamp(r["created_at"], include_year=False)
        r["reviewed_display"] = format_local_timestamp(r["reviewed_at"], include_year=False) if r.get("reviewed_at") else "—"

    if request.is_json or request.args.get("format") == "json":
        return jsonify({"requests": requests_list})

    return render_template("manager/correction_requests.html", requests=requests_list)

@manager_bp.route("/correction-requests/<int:request_id>/review", methods=["POST"])
@manager_bp.route("/api/manager/correction-requests/<int:request_id>", methods=["PATCH", "POST"])
@login_required_manager
def review_correction_request(request_id):
    """
    Approve or Reject a correction request from the manager's team.
    STRICT SECURITY: Checks that the target employee belongs to the manager's scope.
    """
    manager_id = session.get("manager_id") or session.get("employee_id") or session.get("admin_id")
    
    action = request.form.get("action") or (request.get_json() or {}).get("action") or "approve"
    review_note = request.form.get("review_note") or (request.get_json() or {}).get("review_note") or ""

    new_status = "approved" if action == "approve" else "rejected"

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, employee_id, attendance_date, status FROM correction_requests WHERE id = ?", (request_id,))
    cr = c.fetchone()

    if not cr:
        conn.close()
        return jsonify({"success": False, "error": "Correction request not found"}), 404

    target_emp_id = cr["employee_id"]

    # Server-Side Scope Authorization:
    # Manager can only approve/reject requests for their own team!
    if not is_employee_in_manager_scope(manager_id, target_emp_id):
        conn.close()
        abort(403, "Access denied: You cannot review correction requests outside your team.")

    now_iso = get_local_now().isoformat()
    note_text = review_note.strip() or f"Reviewed by Manager #{manager_id} ({new_status.title()})"
    c.execute("""
        UPDATE correction_requests
        SET status = ?,
            reviewed_by = ?,
            review_note = ?,
            reviewed_at = ?
        WHERE id = ?
    """, (new_status, manager_id, note_text, now_iso, request_id))
    conn.commit()
    conn.close()

    log_audit(manager_id, "manager", f"CORRECTION_{new_status.upper()}", "correction_request", request_id, note_text)

    if request.is_json:
        return jsonify({
            "success": True,
            "request_id": request_id,
            "status": new_status,
            "message": f"Correction request #{request_id} has been {new_status}."
        })

    flash(f"Correction request #{request_id} has been {new_status}.", "success")
    return redirect(url_for("manager.correction_requests"))
