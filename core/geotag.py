import os
import io
import time
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config import Config

# Simple memory cache for reverse geocoding: (rounded_lat, rounded_lon) -> address
_GEOCODE_CACHE = {}

def reverse_geocode(latitude, longitude, timeout=3.0):
    """
    Reverse-geocodes latitude and longitude into a human-readable address
    using OpenStreetMap Nominatim with strict timeout and offline fallback.
    """
    if latitude is None or longitude is None:
        return "Unknown Location (No GPS provided)"
        
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (ValueError, TypeError):
        return f"Invalid coordinates ({latitude}, {longitude})"

    # Cache lookup (rounded to ~11 meters)
    cache_key = (round(lat, 4), round(lon, 4))
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    headers = {
        "User-Agent": Config.NOMINATIM_USER_AGENT,
        "Accept": "application/json"
    }
    url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=18&addressdetails=1"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            display_name = data.get("display_name")
            if display_name:
                parts = [p.strip() for p in display_name.split(",")]
                short_address = ", ".join(parts[:4])
                _GEOCODE_CACHE[cache_key] = short_address
                return short_address
    except Exception:
        pass

    fallback_str = f"Lat: {lat:.4f}°, Lon: {lon:.4f}° (Lookup Pending)"
    return fallback_str

def stamp_geotag(
    image_input,
    student_name,
    student_id,
    timestamp_str,
    latitude,
    longitude,
    address,
    confidence,
    status="success",
    output_dir=None,
    event_type="check_in",
    site_name=None,
    distance_meters=None,
    within_geofence=True
):
    """
    Stamps enterprise geotag watermark directly onto proof-of-attendance photo.
    Displays: Employee Name & ID, Event Type (Check-in/out), Timestamp,
    Site Geofence Status, Confidence, and GPS Coordinates with Resolved Address.
    """
    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        if image_input.startswith("data:image"):
            import base64
            img = Image.open(io.BytesIO(base64.b64decode(image_input.split(",", 1)[1])))
        else:
            img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif hasattr(image_input, "read"):
        img = Image.open(image_input)
    else:
        img = image_input

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    width, height = img.size
    
    # Calculate banner dimensions
    banner_height = max(120, int(height * 0.26))
    banner_y0 = height - banner_height

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Solid dark slate banner with ~88% opacity
    draw_overlay.rectangle([(0, banner_y0), (width, height)], fill=(15, 23, 42, 225))
    
    # Accent top border: Green for success/on-time, Amber for late/early, Orange for flagged, Blue for overtime
    if status == "flagged" or not within_geofence:
        border_color = (239, 68, 68, 255) # Red/Orange
    elif status == "overtime":
        border_color = (59, 130, 246, 255) # Blue
    elif status in ("late", "early_leave"):
        border_color = (245, 158, 11, 255) # Amber
    else:
        border_color = (34, 197, 94, 255) # Green
        
    draw_overlay.rectangle([(0, banner_y0), (width, banner_y0 + 5)], fill=border_color)

    composed = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(composed)

    try:
        font_large = ImageFont.truetype("arial.ttf", max(15, int(banner_height * 0.17)))
        font_mid = ImageFont.truetype("arial.ttf", max(11, int(banner_height * 0.12)))
        font_badge = ImageFont.truetype("arialbd.ttf", max(12, int(banner_height * 0.14)))
    except Exception:
        font_large = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_badge = ImageFont.load_default()

    pad_x = max(14, int(width * 0.03))
    line1_y = banner_y0 + 10
    line2_y = line1_y + int(banner_height * 0.25)
    line3_y = line2_y + int(banner_height * 0.24)
    line4_y = line3_y + int(banner_height * 0.22)

    # Line 1: Employee Name, ID & Event Type
    event_label = "CHECK-IN" if event_type == "check_in" else "CHECK-OUT"
    conf_pct = round(confidence * 100, 1) if confidence is not None else 0.0
    name_text = f"{student_name} (ID: #{student_id}) • [{event_label}]"
    draw.text((pad_x, line1_y), name_text, fill=(255, 255, 255, 255), font=font_large)

    # Line 2: Punctuality / Geofence Status Badge & Confidence
    geo_badge = "GEOFENCE VERIFIED" if within_geofence else "GEOFENCE VIOLATION"
    dist_str = f" • Dist: {int(distance_meters)}m" if distance_meters is not None else ""
    status_str = f"STATUS: {status.upper()} • [{geo_badge}{dist_str}] • Match: {conf_pct}%"
    draw.text((pad_x, line2_y), status_str, fill=border_color, font=font_badge)

    # Line 3: Timestamp & Site Info
    site_label = f" • Site: {site_name}" if site_name else ""
    ts_text = f"Time: {timestamp_str}{site_label}"
    draw.text((pad_x, line3_y), ts_text, fill=(203, 213, 225, 255), font=font_mid)

    # Line 4: Geotag Coordinates & Address
    if latitude is not None and longitude is not None:
        geo_text = f"GPS: {latitude:.5f}°, {longitude:.5f}° | {address}"
    else:
        geo_text = f"GPS: Location Unavailable | {address}"
    draw.text((pad_x, line4_y), geo_text, fill=(148, 163, 184, 255), font=font_mid)

    # Save to attendance_photos
    output_dir = output_dir or Config.ATTENDANCE_PHOTOS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    safe_ts = "".join(c if c.isalnum() else "_" for c in timestamp_str)
    filename = f"{student_id}_{event_type}_{safe_ts}.jpg"
    file_path = os.path.join(output_dir, filename)

    final_rgb = composed.convert("RGB")
    final_rgb.save(file_path, "JPEG", quality=90)
    
    rel_path = f"attendance_photos/{filename}"
    return rel_path
