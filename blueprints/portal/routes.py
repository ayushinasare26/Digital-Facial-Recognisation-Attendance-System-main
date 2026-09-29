import os
import datetime
from flask import (
    render_template, request, jsonify, redirect,
    url_for, session, flash, abort, current_app
)
from . import portal_bp
from core.db import get_db_connection
from core.auth import (
    login_required_user,
    authenticate_user,
    rate_limit
)
from core.pipeline import run_attendance_pipeline
from config import Config, get_utc_now, format_local_timestamp

@portal_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    User Portal Login:
    Accepts Student ID, Roll No, or Email with password.
    Includes quick demo student login shortcuts for project demonstration.
    """
    if session.get("role") == "user" and session.get("student_id"):
        return redirect(url_for("portal.mark_attendance"))

    error = None
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "").strip()
        demo_student_id = request.form.get("demo_student_id")

        if demo_student_id:
            user = authenticate_user(demo_student_id, is_demo=True)
        else:
            user = authenticate_user(identifier, password)

        if user:
            session.clear()
            session["role"] = "user"
            session["student_id"] = user["id"]
            session["student_name"] = user["name"]
            session["roll"] = user["roll"] or f"ID-{user['id']}"
            session["department"] = user["class"] or "Student"
            return redirect(url_for("portal.mark_attendance"))
        else:
            error = "Invalid credentials. Please verify your Student ID/Roll number and password."

    # Fetch 4 demo students for 1-click evaluation shortcuts
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class FROM students WHERE id IN (29, 22, 23, 25) ORDER BY id LIMIT 4")
    demo_students = c.fetchall()
    conn.close()

    return render_template("portal/login.html", error=error, demo_students=demo_students)

@portal_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out of your student portal.", "info")
    return redirect(url_for("portal.login"))

@portal_bp.route("/")
@portal_bp.route("/mark-attendance", methods=["GET"])
@login_required_user
def mark_attendance():
    """
    User Portal: Live Geotagged Attendance Marking View.
    Mobile-first interface with live webcam and liveness prompts.
    """
    student_id = session.get("student_id")
    student_name = session.get("student_name", "Student")
    roll = session.get("roll", "—")

    # Fetch today's check-in status for this student
    conn = get_db_connection()
    c = conn.cursor()
    today_str = get_utc_now().date().isoformat()
    c.execute("""
        SELECT id, timestamp, address, confidence, status, geotagged_photo_path
        FROM attendance
        WHERE student_id = ? AND date(timestamp) = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (student_id, today_str))
    today_record = c.fetchone()
    conn.close()

    today_rec_dict = None
    if today_record:
        today_rec_dict = dict(today_record)
        today_rec_dict["time_display"] = format_local_timestamp(today_record["timestamp"], include_year=True)

    return render_template(
        "portal/mark_attendance.html",
        student_id=student_id,
        student_name=student_name,
        roll=roll,
        today_record=today_rec_dict
    )

@portal_bp.route("/mark-attendance", methods=["POST"])
@login_required_user
@rate_limit(max_requests=20, window_seconds=60, scope="user_attendance")
def api_mark_attendance():
    """
    Processes live attendance capture for the authenticated user:
    1. Runs liveness check & face recognition
    2. Validates that the face matches the logged-in student (prevents proxy marking)
    3. Stamps proof-of-location watermark with GPS coordinates
    4. Logs attendance event
    """
    session_student_id = session.get("student_id")
    session_student_name = session.get("student_name")

    primary_image = None
    lat = None
    lon = None
    challenge_type = "blink"
    is_demo = False
    liveness_frames = []

    # Multipart Form Data
    if request.files:
        if "image" in request.files:
            primary_image = request.files["image"].read()
        elif "photo" in request.files:
            primary_image = request.files["photo"].read()
        for lf in request.files.getlist("liveness_frames[]"):
            liveness_frames.append(lf.read())

    bypass_cooldown = False
    if request.form:
        lat = request.form.get("latitude")
        lon = request.form.get("longitude")
        challenge_type = request.form.get("challenge_type", "blink")
        is_demo = request.form.get("is_demo", "false").lower() in ("true", "1", "yes")
        bypass_cooldown = request.form.get("bypass_cooldown", "false").lower() in ("true", "1", "yes")

    # JSON Payload support
    if not primary_image and request.is_json:
        data = request.get_json() or {}
        primary_image = data.get("image") or data.get("photo")
        lat = data.get("latitude")
        lon = data.get("longitude")
        challenge_type = data.get("challenge_type", "blink")
        is_demo = bool(data.get("is_demo", False))
        bypass_cooldown = bool(data.get("bypass_cooldown", False))
        liveness_frames = data.get("liveness_frames") or []

    # Clean coordinates
    if lat in ("", "null", "undefined"): lat = None
    if lon in ("", "null", "undefined"): lon = None

    # Execute 9-stage pipeline
    result = run_attendance_pipeline(
        primary_image_data=primary_image,
        latitude=lat,
        longitude=lon,
        liveness_frames=liveness_frames,
        challenge_type=challenge_type,
        is_demo=is_demo,
        bypass_cooldown=bypass_cooldown
    )

    if not result.get("success"):
        return jsonify(result), 400

    if not lat or not lon:
        result["status"] = "flagged"
        conn = get_db_connection()
        conn.execute("UPDATE attendance SET status = 'flagged', flagged = 1, flag_reason = 'Location Permission Denied' WHERE id = ?", (result.get("attendance_id"),))
        conn.commit()
        conn.close()

    # Authorization Check: Verify matched person corresponds to logged-in user
    matched_id = result.get("student_id")
    if session_student_id and matched_id != session_student_id:
        # Flag proxy attempt
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE attendance
            SET status = 'flagged',
                review_note = ?
            WHERE id = ?
        """, (f"Identity Mismatch / Proxy Warning: Session was {session_student_name} (#{session_student_id}) but face matched {result.get('student_name')} (#{matched_id}).", result.get("attendance_id")))
        conn.commit()
        conn.close()

        return jsonify({
            "success": False,
            "error_stage": "Identity Verification",
            "message": f"Biometric mismatch: The face in front of the camera matched {result.get('student_name')}, which does not match your active session ({session_student_name}). Proxy attendance is prohibited.",
            "stages": result.get("stages", [])
        }), 403

    return jsonify(result), 200

@portal_bp.route("/my-attendance", methods=["GET"])
@login_required_user
def my_attendance():
    """
    User Portal: Scoped Attendance History.
    STRICT SECURITY: Every query is hard-scoped to session['student_id'].
    Users can never view any other student's records or coordinates.
    """
    student_id = session.get("student_id")
    from_date = request.args.get("from")
    to_date = request.args.get("to")
    status_filter = request.args.get("status", "all")

    conn = get_db_connection()
    c = conn.cursor()

    query = """
        SELECT id, timestamp, latitude, longitude, address, confidence,
               liveness_passed, geotagged_photo_path, status
        FROM attendance
        WHERE student_id = ?
    """
    params = [student_id]

    if from_date:
        query += " AND date(timestamp) >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date(timestamp) <= ?"
        params.append(to_date)
    if status_filter in ("success", "flagged"):
        query += " AND status = ?"
        params.append(status_filter)

    query += " ORDER BY timestamp DESC LIMIT 200"
    c.execute(query, tuple(params))
    records = c.fetchall()

    # Calculate student summary metrics
    c.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ?", (student_id,))
    total_attended = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND status = 'success'", (student_id,))
    verified_count = c.fetchone()[0] or 0

    conn.close()

    formatted = []
    for r in records:
        conf_pct = round((r["confidence"] or 0.0) * 100, 1)
        time_display = format_local_timestamp(r["timestamp"], include_year=True)

        formatted.append({
            "id": r["id"],
            "time_display": time_display,
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "address": r["address"] or "Location Unavailable",
            "confidence": conf_pct,
            "status": r["status"] or "success",
            "photo_path": r["geotagged_photo_path"]
        })

    return render_template(
        "portal/my_attendance.html",
        records=formatted,
        total_attended=total_attended,
        verified_count=verified_count,
        from_date=from_date or "",
        to_date=to_date or "",
        status_filter=status_filter
    )

@portal_bp.route("/my-attendance/<int:record_id>", methods=["GET"])
@login_required_user
def api_my_attendance_detail(record_id):
    """
    User Portal: Single Record Detail.
    STRICT SECURITY: Enforces ownership check. If record_id does not belong
    to session['student_id'], returns 404/403 to prevent record snooping.
    """
    session_student_id = session.get("student_id")

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, student_id, name, timestamp, latitude, longitude,
               address, confidence, liveness_passed, geotagged_photo_path, status
        FROM attendance
        WHERE id = ? AND student_id = ?
    """, (record_id, session_student_id))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Attendance record not found or access denied"}), 404

    data = dict(row)
    data["time_display"] = format_local_timestamp(data["timestamp"], include_year=True)
    data["photo_url"] = f"/{data['geotagged_photo_path']}" if data.get("geotagged_photo_path") else None
    return jsonify(data)
