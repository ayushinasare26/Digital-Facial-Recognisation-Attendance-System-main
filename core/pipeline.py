import os
import time
import uuid
import datetime
import numpy as np

from config import Config, get_utc_now, get_utc_iso, get_local_now, format_local_date, format_local_timestamp
from core.db import get_db_connection, get_setting
from core.face_engine import (
    decode_image_bytes,
    detect_face_and_embedding,
    match_face_embedding
)
from core.liveness import verify_liveness
from core.geotag import reverse_geocode, stamp_geotag
from core.geofence import validate_geofence
from core.shift_engine import (
    evaluate_check_in,
    evaluate_check_out,
    calculate_shift_hours
)

def log_pipeline_stage(run_id, stage_name, status, message, attendance_id=None):
    """
    Inserts a record into the pipeline_logs table for audit and real-time visualization.
    """
    conn = get_db_connection()
    c = conn.cursor()
    now = get_utc_iso()
    try:
        c.execute("""
            INSERT INTO pipeline_logs (attendance_id, run_id, stage, status, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (attendance_id, run_id, stage_name, status, message, now))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def check_duplicate_event(employee_id, event_type="check_in", cooldown_seconds=None):
    """
    Checks if this employee has already performed the same event_type within the cooldown window.
    Returns: (is_duplicate: bool, previous_record: dict)
    """
    if cooldown_seconds is None:
        try:
            cooldown_seconds = int(get_setting("duplicate_cooldown_seconds", Config.DUPLICATE_COOLDOWN_SECONDS))
        except Exception:
            cooldown_seconds = Config.DUPLICATE_COOLDOWN_SECONDS

    if cooldown_seconds <= 0:
        return False, None

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, event_type
        FROM attendance_events
        WHERE employee_id = ? AND event_type = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (employee_id, event_type))
    row = c.fetchone()
    
    # Also check legacy attendance table
    if not row:
        c.execute("""
            SELECT id, timestamp, 'check_in' as event_type
            FROM attendance
            WHERE student_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (employee_id,))
        row = c.fetchone()
        
    conn.close()

    if not row:
        return False, None

    last_ts_str = row["timestamp"]
    try:
        last_dt = datetime.datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.timezone.utc)
        now_dt = get_utc_now()
        elapsed = (now_dt - last_dt).total_seconds()
        if 0 <= elapsed < cooldown_seconds:
            remaining_mins = max(1, int((cooldown_seconds - elapsed) / 60))
            return True, {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "elapsed_seconds": int(elapsed),
                "remaining_minutes": remaining_mins
            }
    except Exception:
        pass

    return False, None

def get_employee_site_and_shift(employee_id):
    """
    Fetches assigned site and shift configuration for an employee.
    Falls back to system defaults if not assigned.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT e.id, e.name, e.site_id, e.shift_id,
               s.name AS site_name, s.latitude AS site_lat, s.longitude AS site_lon,
               s.geofence_radius_meters, s.geofencing_enabled,
               sh.name AS shift_name, sh.start_time, sh.end_time,
               sh.grace_period_minutes, sh.break_duration_minutes
        FROM employees e
        LEFT JOIN sites s ON e.site_id = s.id
        LEFT JOIN shifts sh ON e.shift_id = sh.id
        WHERE e.id = ?
    """, (employee_id,))
    emp_meta = c.fetchone()
    conn.close()

    site = {
        "name": emp_meta["site_name"] if emp_meta and emp_meta["site_name"] else "Headquarters",
        "latitude": emp_meta["site_lat"] if emp_meta and emp_meta["site_lat"] is not None else 19.0657,
        "longitude": emp_meta["site_lon"] if emp_meta and emp_meta["site_lon"] is not None else 72.8687,
        "geofence_radius_meters": emp_meta["geofence_radius_meters"] if emp_meta and emp_meta["geofence_radius_meters"] is not None else 200.0,
        "geofencing_enabled": bool(emp_meta["geofencing_enabled"]) if emp_meta and emp_meta["geofencing_enabled"] is not None else True,
        "id": emp_meta["site_id"] if emp_meta and emp_meta["site_id"] else 1
    }

    shift = {
        "name": emp_meta["shift_name"] if emp_meta and emp_meta["shift_name"] else "General Shift",
        "start_time": emp_meta["start_time"] if emp_meta and emp_meta["start_time"] else "09:00",
        "end_time": emp_meta["end_time"] if emp_meta and emp_meta["end_time"] else "18:00",
        "grace_period_minutes": emp_meta["grace_period_minutes"] if emp_meta and emp_meta["grace_period_minutes"] is not None else 15,
        "break_duration_minutes": emp_meta["break_duration_minutes"] if emp_meta and emp_meta["break_duration_minutes"] is not None else 60,
        "id": emp_meta["shift_id"] if emp_meta and emp_meta["shift_id"] else 1
    }

    return site, shift

def run_attendance_pipeline(
    primary_image_data,
    latitude=None,
    longitude=None,
    liveness_frames=None,
    challenge_type="blink",
    is_demo=False,
    bypass_cooldown=False,
    event_type="check_in",
    expected_employee_id=None
):
    """
    Executes the 9-stage Enterprise Geo-Verified Attendance Pipeline:
    1. Photo Received (Intake)
    2. Liveness Check
    3. Face Detection
    4. Embedding Extraction
    5. Embedding Match
    6. Geolocation Capture (& Geofence Verification)
    7. Reverse Geocoding
    8. Geotag Stamping
    9. Record Saved (& Shift Engine Evaluation)
    """
    event_type = "check_out" if str(event_type).lower() in ("check_out", "checkout") else "check_in"
    run_id = f"pipe_{uuid.uuid4().hex[:10]}"
    stages_log = []
    
    def record_stage(stage, status, message):
        stages_log.append({
            "stage": stage,
            "status": status,
            "message": message,
            "timestamp": get_utc_iso()
        })
        log_pipeline_stage(run_id, stage, status, message)

    # ---------------- STAGE 1: Photo Received (Intake) ----------------
    has_primary = primary_image_data is not None and not (isinstance(primary_image_data, (bytes, str, list)) and len(primary_image_data) == 0)
    if not has_primary and is_demo:
        demo_folder = os.path.join(Config.DATASET_DIR, "29")
        if os.path.isdir(demo_folder):
            demo_files = [f for f in os.listdir(demo_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if demo_files:
                with open(os.path.join(demo_folder, demo_files[0]), "rb") as df:
                    primary_image_data = df.read()
                    has_primary = True

    if not has_primary:
        record_stage("Photo Received", "Failed", "No image data payload received from client")
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Photo Received",
            "message": "Missing capture photo in request",
            "stages": stages_log
        }
        
    try:
        rgb_image = decode_image_bytes(primary_image_data)
        record_stage("Photo Received", "Completed", f"Image received and decoded ({rgb_image.shape[1]}x{rgb_image.shape[0]} px)")
    except Exception as e:
        record_stage("Photo Received", "Failed", f"Failed to decode photo: {str(e)}")
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Photo Received",
            "message": "Corrupted or unsupported image file format",
            "stages": stages_log
        }

    # ---------------- STAGE 2: Liveness Check ----------------
    frames_to_check = liveness_frames if (liveness_frames and len(liveness_frames) > 0) else [rgb_image]
    liveness_res = verify_liveness(frames_to_check, challenge_type=challenge_type, is_demo=is_demo)
    
    if not liveness_res["passed"]:
        record_stage("Liveness Check", "Failed", liveness_res["reason"])
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Liveness Check",
            "message": liveness_res["reason"],
            "liveness_passed": False,
            "liveness_score": liveness_res["score"],
            "stages": stages_log
        }
    record_stage("Liveness Check", "Completed", f"{liveness_res['reason']} (Score: {liveness_res['score']})")

    # ---------------- STAGE 3: Face Detection ----------------
    decoded_candidate_frames = [rgb_image]
    if liveness_frames:
        for lf in liveness_frames:
            try:
                decoded_candidate_frames.append(decode_image_bytes(lf))
            except Exception:
                pass

    best_frame = rgb_image
    best_loc = None
    best_emb = None
    best_err = None
    best_face_area = 0

    for candidate_rgb in decoded_candidate_frames:
        loc, emb, err = detect_face_and_embedding(candidate_rgb)
        if loc is not None and emb is not None:
            top, right, bottom, left = loc
            area = (bottom - top) * (right - left)
            if area > best_face_area:
                best_face_area = area
                best_frame = candidate_rgb
                best_loc = loc
                best_emb = emb
                best_err = None
        elif best_err is None and err:
            best_err = err

    if best_emb is None or best_loc is None:
        record_stage("Face Detection", "Failed", best_err or "No valid face found in capture")
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Face Detection",
            "message": best_err or "Face detection failed",
            "stages": stages_log
        }

    rgb_image = best_frame
    location = best_loc
    embedding = best_emb
    record_stage("Face Detection", "Completed", f"Face localized at {location} (size: {location[2]-location[0]}x{location[1]-location[3]} px)")

    # ---------------- STAGE 4: Embedding Extraction ----------------
    if embedding is None or len(embedding) != 128:
        record_stage("Embedding Extraction", "Failed", "Invalid embedding length extracted")
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Embedding Extraction",
            "message": "Embedding extraction failed",
            "stages": stages_log
        }
    record_stage("Embedding Extraction", "Completed", "Extracted 128-dimensional deep ResNet embedding vector")

    # ---------------- STAGE 5: Embedding Match ----------------
    match_res = match_face_embedding(embedding, expected_id=expected_employee_id)
    if not match_res["matched"]:
        record_stage("Embedding Match", "Failed", match_res["reason"])
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Embedding Match",
            "message": match_res["reason"],
            "confidence": match_res["confidence"],
            "distance": match_res["distance"],
            "stages": stages_log
        }

    student_id = match_res["student_id"]
    student_name = match_res["name"]
    confidence = match_res["confidence"]
    record_status = match_res["status"]
    record_stage("Embedding Match", "Completed", f"Matched person: {student_name} (#{student_id}) with {round(confidence*100, 1)}% confidence")

    # Cooldown check
    if not bypass_cooldown:
        is_dup, prev = check_duplicate_event(student_id, event_type=event_type)
        if is_dup:
            msg = f"Duplicate attendance prevented: {student_name} marked attendance {prev['elapsed_seconds']}s ago. Cooldown remaining: {prev['remaining_minutes']}m."
            record_stage("Duplicate Check", "Failed", msg)
            return {
                "success": False,
                "run_id": run_id,
                "error_stage": "Duplicate Check",
                "message": msg,
                "student_id": student_id,
                "student_name": student_name,
                "confidence": confidence,
                "stages": stages_log
            }

    # ---------------- STAGE 6: Geolocation Capture (& Geofence Verification) ----------------
    assigned_site, assigned_shift = get_employee_site_and_shift(student_id)

    has_valid_coords = False
    lat_val, lon_val = None, None
    if latitude is not None and longitude is not None:
        try:
            lat_val = float(latitude)
            lon_val = float(longitude)
            if -90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0:
                has_valid_coords = True
        except (ValueError, TypeError):
            pass

    geofence_res = validate_geofence(
        employee_lat=lat_val,
        employee_lon=lon_val,
        site_lat=assigned_site["latitude"],
        site_lon=assigned_site["longitude"],
        radius_meters=assigned_site["geofence_radius_meters"],
        geofencing_enabled=assigned_site["geofencing_enabled"],
        site_name=assigned_site["name"]
    )

    within_geofence = geofence_res["within_geofence"]
    dist_meters = geofence_res["distance_meters"]
    is_flagged = geofence_res["flagged"]
    flag_reason = geofence_res["flag_reason"]

    if has_valid_coords:
        if within_geofence:
            record_stage("Geolocation Capture", "Completed", f"GPS acquired: Lat {lat_val:.5f}, Lon {lon_val:.5f}. Geofence verified ({dist_meters}m from {assigned_site['name']}).")
        else:
            record_stage("Geolocation Capture", "Completed", f"GPS acquired: Lat {lat_val:.5f}, Lon {lon_val:.5f}. Remote location logged ({dist_meters}m from {assigned_site['name']}).")
    else:
        if assigned_site.get("geofencing_enabled", 0):
            record_status = "flagged"
            is_flagged = True
            record_stage("Geolocation Capture", "Failed", "GPS coordinates missing or denied by client. Flagged for audit.")
        else:
            within_geofence = 1
            is_flagged = False
            flag_reason = None
            record_stage("Geolocation Capture", "Completed", "Remote workforce policy: Location logged without radius restriction.")

    # ---------------- STAGE 7: Reverse Geocoding ----------------
    resolved_address = "Location Permission Denied"
    if has_valid_coords:
        resolved_address = reverse_geocode(lat_val, lon_val)
        if "Fallback" in resolved_address or "Pending" in resolved_address:
            record_stage("Reverse Geocoding", "Fallback", f"OSM Nominatim offline; fallback: {resolved_address}")
        else:
            record_stage("Reverse Geocoding", "Completed", f"Address: {resolved_address}")
    else:
        record_stage("Reverse Geocoding", "Skipped", "Skipped reverse geocoding because coordinates were absent")

    # Shift punctuality evaluation
    local_now = get_local_now()
    now_dt = local_now.replace(tzinfo=None)
    hours_worked = 0.0
    overtime_hours = 0.0

    if event_type == "check_in":
        shift_eval = evaluate_check_in(now_dt, assigned_shift)
        event_status = shift_eval["status"]
        if event_status == "late" and not flag_reason:
            flag_reason = shift_eval.get("flag_reason")
        # Only override event_status to flagged if face verification failed or geofencing strictly required
        if is_flagged and assigned_site.get("geofencing_enabled", 0):
            event_status = "flagged"
    else:
        conn = get_db_connection()
        c = conn.cursor()
        today_date = local_now.date().isoformat()
        c.execute("""
            SELECT timestamp FROM attendance_events
            WHERE employee_id = ? AND event_type = 'check_in'
            ORDER BY timestamp DESC
        """, (student_id,))
        checkin_rows = [r["timestamp"] for r in c.fetchall() if format_local_date(r["timestamp"]) == today_date]
        conn.close()

        # Use the latest check-in of the current active session
        check_in_ts = checkin_rows[0] if checkin_rows else None
        hours_res = calculate_shift_hours(check_in_ts, now_dt, assigned_shift)
        event_status = hours_res["status"]
        hours_worked = hours_res.get("regular_hours", 0.0)
        overtime_hours = hours_res.get("overtime_hours", 0.0)

        if is_flagged and assigned_site.get("geofencing_enabled", 0):
            event_status = "flagged"
        elif hours_res.get("flagged") and not assigned_site.get("geofencing_enabled", 0):
            # Do not flag active in-progress shift or remote workers
            pass
        elif hours_res.get("flagged"):
            is_flagged = True
            flag_reason = (flag_reason + "; " if flag_reason else "") + (hours_res.get("flag_reason") or "")

    # ---------------- STAGE 8: Geotag Stamping ----------------
    timestamp_display = local_now.strftime("%b %d, %Y, %I:%M:%S %p")
    photo_rel_path = None
    try:
        photo_rel_path = stamp_geotag(
            image_input=rgb_image,
            student_name=student_name,
            student_id=student_id,
            timestamp_str=timestamp_display,
            latitude=lat_val,
            longitude=lon_val,
            address=resolved_address,
            confidence=confidence,
            status=event_status if not is_flagged else "flagged",
            event_type=event_type,
            site_name=assigned_site["name"],
            distance_meters=dist_meters,
            within_geofence=within_geofence
        )
        record_stage("Geotag Stamping", "Completed", f"Proof photo generated: {photo_rel_path}")
    except Exception as e:
        record_stage("Geotag Stamping", "Failed", f"Watermark stamping error: {str(e)}")
        photo_rel_path = None

    # ---------------- STAGE 9: Record Saved ----------------
    conn = get_db_connection()
    c = conn.cursor()
    iso_timestamp = get_utc_iso()
    try:
        # 1. Save to attendance_events table
        c.execute("""
            INSERT INTO attendance_events (
                employee_id, event_type, timestamp, latitude, longitude,
                address, site_id, distance_from_site_meters, within_geofence,
                confidence, liveness_passed, geotagged_photo_path,
                status, flagged, flag_reason, attendance_method, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            event_type,
            iso_timestamp,
            lat_val,
            lon_val,
            resolved_address,
            assigned_site["id"],
            dist_meters,
            1 if within_geofence else 0,
            confidence,
            1 if liveness_res["passed"] else 0,
            photo_rel_path,
            event_status,
            1 if is_flagged else 0,
            flag_reason,
            "face",
            iso_timestamp
        ))
        event_id = c.lastrowid

        # 2. Mirror to attendance table
        legacy_status = "flagged" if is_flagged else ("success" if event_status in ("on_time", "overtime", "success") else event_status)
        c.execute("""
            INSERT INTO attendance (
                student_id, name, timestamp, latitude, longitude,
                address, confidence, liveness_passed, geotagged_photo_path,
                status, event_type, site_id, distance_from_site_meters,
                within_geofence, flagged, flag_reason, attendance_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            student_name,
            iso_timestamp,
            lat_val,
            lon_val,
            resolved_address,
            confidence,
            1 if liveness_res["passed"] else 0,
            photo_rel_path,
            legacy_status,
            event_type,
            assigned_site["id"],
            dist_meters,
            1 if within_geofence else 0,
            1 if is_flagged else 0,
            flag_reason,
            "face"
        ))
        legacy_att_id = c.lastrowid
        conn.commit()

        status_msg = f"{event_type.replace('_', ' ').title()} logged to database with ID #{legacy_att_id} (Status: {legacy_status.upper()})"
        record_stage("Record Saved", "Completed", status_msg)
        c.execute("UPDATE pipeline_logs SET attendance_id=? WHERE run_id=?", (legacy_att_id, run_id))
        conn.commit()
    except Exception as e:
        record_stage("Record Saved", "Failed", f"DB insert error: {str(e)}")
        conn.close()
        return {
            "success": False,
            "run_id": run_id,
            "error_stage": "Record Saved",
            "message": f"Database logging error: {str(e)}",
            "stages": stages_log
        }
    finally:
        conn.close()

    return {
        "success": True,
        "run_id": run_id,
        "event_id": event_id,
        "attendance_id": legacy_att_id,
        "employee_id": student_id,
        "student_id": student_id,
        "student_name": student_name,
        "employee_name": student_name,
        "attendance_method": "face",
        "event_type": event_type,
        "confidence": confidence,
        "liveness_passed": True,
        "status": legacy_status,
        "event_status": event_status,
        "within_geofence": within_geofence,
        "distance_from_site_meters": dist_meters,
        "site_name": assigned_site["name"],
        "shift_name": assigned_shift["name"],
        "flagged": is_flagged,
        "flag_reason": flag_reason,
        "hours_worked": hours_worked,
        "overtime_hours": overtime_hours,
        "latitude": lat_val,
        "longitude": lon_val,
        "address": resolved_address,
        "timestamp": iso_timestamp,
        "timestamp_display": timestamp_display,
        "photo_url": f"/{photo_rel_path}" if photo_rel_path else None,
        "stages": stages_log
    }
