"""
core/shift_engine.py - Industrial Shift Scheduling & Punctuality Engine
Matches check-in and check-out events against shift schedules to automatically
compute: on-time, late, early leave, overtime, absent, and hours worked.
"""

import datetime
from typing import Dict, Any, Optional, Tuple

def parse_time_str(time_str: str) -> datetime.time:
    """Parses 'HH:MM' or 'HH:MM:SS' string into a datetime.time object."""
    if not time_str:
        return datetime.time(9, 0)
    parts = time_str.strip().split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0
    return datetime.time(hour, minute, second)

def parse_iso_datetime(dt_val) -> Optional[datetime.datetime]:
    """Parses ISO string or returns datetime object."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime.datetime):
        return dt_val
    try:
        clean_str = str(dt_val).replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(clean_str)
    except Exception:
        return None

def normalize_datetimes_to_naive(dt1: datetime.datetime, dt2: Optional[datetime.datetime] = None):
    """Normalizes one or two datetimes to comparable naive datetimes (in local or UTC)."""
    if dt1.tzinfo is not None:
        # Convert to local naive
        dt1 = dt1.astimezone().replace(tzinfo=None)
    if dt2 is not None:
        if dt2.tzinfo is not None:
            dt2 = dt2.astimezone().replace(tzinfo=None)
        return dt1, dt2
    return dt1

def evaluate_check_in(
    check_in_dt: Any,
    shift: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates a check-in event against the assigned shift schedule.
    
    Shift dict should contain:
        - start_time: 'HH:MM' (e.g. '09:00')
        - end_time: 'HH:MM' (e.g. '18:00')
        - grace_period_minutes: int (e.g. 15)
        
    Returns:
        {
            "status": "on_time" | "late",
            "minutes_late": float,
            "scheduled_start": str,
            "grace_limit": str,
            "flagged": bool,
            "flag_reason": Optional[str],
            "message": str
        }
    """
    dt = parse_iso_datetime(check_in_dt)
    if dt is None:
        return {
            "status": "unknown",
            "minutes_late": 0.0,
            "scheduled_start": "09:00",
            "grace_limit": "09:15",
            "flagged": False,
            "flag_reason": None,
            "message": "Invalid check-in timestamp"
        }

    dt = normalize_datetimes_to_naive(dt)

    start_t = parse_time_str(shift.get("start_time", "09:00"))
    grace_mins = int(shift.get("grace_period_minutes", 15))

    scheduled_start = datetime.datetime.combine(dt.date(), start_t)
    grace_limit = scheduled_start + datetime.timedelta(minutes=grace_mins)

    if dt <= grace_limit:
        mins_early = max(0.0, (scheduled_start - dt).total_seconds() / 60.0)
        return {
            "status": "on_time",
            "minutes_late": 0.0,
            "scheduled_start": scheduled_start.strftime("%I:%M %p"),
            "grace_limit": grace_limit.strftime("%I:%M %p"),
            "flagged": False,
            "flag_reason": None,
            "message": f"Checked in on time (within {grace_mins}m grace period)"
        }
    else:
        minutes_late = round((dt - scheduled_start).total_seconds() / 60.0, 1)
        reason = f"Late check-in: {minutes_late} minutes past scheduled start ({scheduled_start.strftime('%I:%M %p')})"
        return {
            "status": "late",
            "minutes_late": minutes_late,
            "scheduled_start": scheduled_start.strftime("%I:%M %p"),
            "grace_limit": grace_limit.strftime("%I:%M %p"),
            "flagged": False,  # Late is a punctuality status, not necessarily a security flag
            "flag_reason": reason,
            "message": f"Late check-in by {int(minutes_late)} minutes"
        }

def evaluate_check_out(
    check_out_dt: Any,
    shift: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates a check-out event against the assigned shift schedule.
    
    Returns:
        {
            "status": "on_time" | "early_leave" | "overtime",
            "minutes_early": float,
            "minutes_overtime": float,
            "scheduled_end": str,
            "flagged": bool,
            "flag_reason": Optional[str],
            "message": str
        }
    """
    dt = parse_iso_datetime(check_out_dt)
    if dt is None:
        return {
            "status": "unknown",
            "minutes_early": 0.0,
            "minutes_overtime": 0.0,
            "scheduled_end": "18:00",
            "flagged": False,
            "flag_reason": None,
            "message": "Invalid check-out timestamp"
        }

    dt = normalize_datetimes_to_naive(dt)

    end_t = parse_time_str(shift.get("end_time", "18:00"))
    scheduled_end = datetime.datetime.combine(dt.date(), end_t)

    # Shift crosses midnight? (e.g. 22:00 to 06:00)
    start_t = parse_time_str(shift.get("start_time", "09:00"))
    if end_t < start_t:
        # Scheduled end is on the next day
        scheduled_end += datetime.timedelta(days=1)

    if dt < scheduled_end:
        minutes_early = round((scheduled_end - dt).total_seconds() / 60.0, 1)
        # Tolerance of 5 minutes before shift end is considered acceptable
        if minutes_early > 5.0:
            return {
                "status": "early_leave",
                "minutes_early": minutes_early,
                "minutes_overtime": 0.0,
                "scheduled_end": scheduled_end.strftime("%I:%M %p"),
                "flagged": False,
                "flag_reason": f"Early leave: {int(minutes_early)}m before scheduled end ({scheduled_end.strftime('%I:%M %p')})",
                "message": f"Early departure by {int(minutes_early)} minutes"
            }
        else:
            return {
                "status": "on_time",
                "minutes_early": 0.0,
                "minutes_overtime": 0.0,
                "scheduled_end": scheduled_end.strftime("%I:%M %p"),
                "flagged": False,
                "flag_reason": None,
                "message": "Checked out at scheduled shift end"
            }
    else:
        minutes_overtime = round((dt - scheduled_end).total_seconds() / 60.0, 1)
        status = "overtime" if minutes_overtime >= 15.0 else "on_time"
        return {
            "status": status,
            "minutes_early": 0.0,
            "minutes_overtime": minutes_overtime,
            "scheduled_end": scheduled_end.strftime("%I:%M %p"),
            "flagged": False,
            "flag_reason": None,
            "message": f"Shift completed ({int(minutes_overtime)}m extra logged)" if minutes_overtime >= 15.0 else "Shift completed on time"
        }

def calculate_shift_hours(
    check_in_dt: Any,
    check_out_dt: Any,
    shift: Dict[str, Any],
    allow_in_progress: bool = False
) -> Dict[str, Any]:
    """
    Computes regular hours, overtime hours, net duration, and overall daily status.
    
    Edge cases handled:
        - In-progress shift: When allow_in_progress=True and check_out is None, calculates elapsed hours and retains check-in punctuality
        - Missing check-out: status='missing_checkout', hours=0
        - Check-out earlier than check-in: status='invalid_sequence', flagged=True
        - Break duration deduction: Deducted only if gross duration >= break duration
        - Overtime calculation: Worked time exceeding scheduled shift duration
    """
    in_dt = parse_iso_datetime(check_in_dt)
    out_dt = parse_iso_datetime(check_out_dt)

    if in_dt is None and out_dt is None:
        return {
            "status": "absent",
            "gross_hours": 0.0,
            "net_hours": 0.0,
            "regular_hours": 0.0,
            "overtime_hours": 0.0,
            "flagged": False,
            "flag_reason": "No check-in or check-out recorded for the day"
        }

    if in_dt is not None and out_dt is None:
        in_eval = evaluate_check_in(in_dt, shift)
        if allow_in_progress:
            in_naive = normalize_datetimes_to_naive(in_dt)
            now_dt = datetime.datetime.now()
            elapsed_sec = max(0.0, (now_dt - in_naive).total_seconds())
            elapsed_hours = round(elapsed_sec / 3600.0, 1)
            elapsed_hours = min(elapsed_hours, 12.0)
            return {
                "status": in_eval["status"],
                "shift_state": "in_progress",
                "gross_hours": elapsed_hours,
                "net_hours": elapsed_hours,
                "regular_hours": elapsed_hours,
                "overtime_hours": 0.0,
                "check_in_status": in_eval["status"],
                "flagged": False,
                "flag_reason": None,
                "is_active_shift": True
            }
        return {
            "status": "missing_checkout",
            "gross_hours": 0.0,
            "net_hours": 0.0,
            "regular_hours": 0.0,
            "overtime_hours": 0.0,
            "check_in_status": in_eval["status"],
            "flagged": True,
            "flag_reason": "Incomplete attendance: Checked in but check-out timestamp is missing",
            "is_active_shift": False
        }

    if in_dt is None and out_dt is not None:
        return {
            "status": "missing_checkin",
            "gross_hours": 0.0,
            "net_hours": 0.0,
            "regular_hours": 0.0,
            "overtime_hours": 0.0,
            "flagged": True,
            "flag_reason": "Incomplete attendance: Check-out recorded without matching check-in"
        }

    # Normalize both
    in_dt, out_dt = normalize_datetimes_to_naive(in_dt, out_dt)

    # Edge Case: check_out before check_in
    if out_dt < in_dt:
        return {
            "status": "invalid_sequence",
            "gross_hours": 0.0,
            "net_hours": 0.0,
            "regular_hours": 0.0,
            "overtime_hours": 0.0,
            "flagged": True,
            "flag_reason": f"Corrupted sequence: check_out ({out_dt.strftime('%H:%M')}) is earlier than check_in ({in_dt.strftime('%H:%M')})"
        }

    # Gross duration in hours
    gross_seconds = (out_dt - in_dt).total_seconds()
    gross_hours = round(gross_seconds / 3600.0, 2)

    # Break duration in hours
    break_mins = int(shift.get("break_duration_minutes", 60))
    break_hours = round(break_mins / 60.0, 2)

    # Only deduct break if shift was long enough (>= 4 hours)
    if gross_hours >= 4.0:
        net_hours = max(0.0, round(gross_hours - break_hours, 2))
    else:
        net_hours = gross_hours

    # Scheduled shift duration
    start_t = parse_time_str(shift.get("start_time", "09:00"))
    end_t = parse_time_str(shift.get("end_time", "18:00"))
    ref_date = in_dt.date()
    sch_start = datetime.datetime.combine(ref_date, start_t)
    sch_end = datetime.datetime.combine(ref_date, end_t)
    if sch_end < sch_start:
        sch_end += datetime.timedelta(days=1)
        
    sch_gross_hours = (sch_end - sch_start).total_seconds() / 3600.0
    sch_regular_hours = max(0.0, round(sch_gross_hours - (break_hours if sch_gross_hours >= 4.0 else 0.0), 2))

    # Overtime & Regular split
    if net_hours > sch_regular_hours:
        regular_hours = sch_regular_hours
        overtime_hours = round(net_hours - sch_regular_hours, 2)
    else:
        regular_hours = net_hours
        overtime_hours = 0.0

    in_eval = evaluate_check_in(in_dt, shift)
    out_eval = evaluate_check_out(out_dt, shift)

    # Synthesize daily status:
    # Priority: flagged -> overtime -> late -> early_leave -> on_time
    if overtime_hours > 0.0:
        daily_status = "overtime"
    elif in_eval["status"] == "late":
        daily_status = "late"
    elif out_eval["status"] == "early_leave":
        daily_status = "early_leave"
    else:
        daily_status = "on_time"

    flagged = in_eval["flagged"] or out_eval["flagged"]
    flag_reasons = [r for r in [in_eval.get("flag_reason"), out_eval.get("flag_reason")] if r]

    return {
        "status": daily_status,
        "check_in_status": in_eval["status"],
        "check_out_status": out_eval["status"],
        "gross_hours": gross_hours,
        "net_hours": net_hours,
        "regular_hours": regular_hours,
        "overtime_hours": overtime_hours,
        "scheduled_regular_hours": sch_regular_hours,
        "minutes_late": in_eval["minutes_late"],
        "minutes_early": out_eval["minutes_early"],
        "flagged": flagged,
        "flag_reason": "; ".join(flag_reasons) if flag_reasons else None
    }
