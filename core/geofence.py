"""
core/geofence.py - Enterprise Geofencing & Haversine Distance Engine
Calculates precise geographic distance between submitted check-in coordinates
and registered office/site coordinates, enforcing work-site geofencing policies.
"""

import math

# Radius of Earth in meters
EARTH_RADIUS_METERS = 6371000.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth's surface
    using the Haversine formula.
    
    Parameters:
        lat1, lon1: Coordinates of point 1 in decimal degrees.
        lat2, lon2: Coordinates of point 2 in decimal degrees.
        
    Returns:
        float: Distance between points in meters.
    """
    if None in (lat1, lon1, lat2, lon2):
        return None
        
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (ValueError, TypeError):
        return None

    # Convert decimal degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    
    # Clip a to [0.0, 1.0] to guard against rare floating point rounding inaccuracies
    a = max(0.0, min(1.0, a))
    
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    distance = EARTH_RADIUS_METERS * c
    return round(distance, 2)

def validate_geofence(
    employee_lat,
    employee_lon,
    site_lat,
    site_lon,
    radius_meters=200.0,
    geofencing_enabled=True,
    site_name=None
):
    """
    Validates whether an employee's submitted coordinates satisfy a site's geofencing policy.
    
    Parameters:
        employee_lat, employee_lon: Submitted coordinates from browser GPS.
        site_lat, site_lon: Registered site/office coordinates.
        radius_meters: Permitted circular boundary in meters (default: 200m).
        geofencing_enabled: If False, geofencing is bypassed (e.g. for remote/field staff).
        site_name: Optional human-readable site name for logging and warning messages.
        
    Returns:
        dict: {
            "within_geofence": bool,
            "distance_meters": float or None,
            "radius_meters": float,
            "geofencing_enabled": bool,
            "flagged": bool,
            "flag_reason": str or None,
            "message": str
        }
    """
    label = site_name or "assigned work site"
    
    # Policy 1: Remote / Field Employees with geofencing disabled
    if not geofencing_enabled:
        dist = None
        if employee_lat is not None and employee_lon is not None and site_lat is not None and site_lon is not None:
            dist = haversine_distance(employee_lat, employee_lon, site_lat, site_lon)
            
        return {
            "within_geofence": True,
            "distance_meters": dist,
            "radius_meters": radius_meters,
            "geofencing_enabled": False,
            "flagged": False,
            "flag_reason": None,
            "message": f"Geofence disabled for {label} (Field/Remote Policy: Coordinates logged without radius restriction)."
        }

    # Policy 2: Geofencing enabled, but coordinates are missing or unreadable
    if employee_lat is None or employee_lon is None:
        return {
            "within_geofence": False,
            "distance_meters": None,
            "radius_meters": radius_meters,
            "geofencing_enabled": True,
            "flagged": True,
            "flag_reason": f"Missing GPS coordinates for geofenced site ({label})",
            "message": "GPS coordinates were not provided or location access was denied."
        }

    # Policy 3: Site itself has undefined coordinates
    if site_lat is None or site_lon is None:
        return {
            "within_geofence": True,
            "distance_meters": None,
            "radius_meters": radius_meters,
            "geofencing_enabled": False,
            "flagged": False,
            "flag_reason": None,
            "message": f"Site {label} has no configured geographic coordinates."
        }

    # Policy 4: Compute Haversine distance and evaluate against radius boundary
    dist = haversine_distance(employee_lat, employee_lon, site_lat, site_lon)
    if dist is None:
        return {
            "within_geofence": False,
            "distance_meters": None,
            "radius_meters": radius_meters,
            "geofencing_enabled": True,
            "flagged": True,
            "flag_reason": "Invalid coordinate formatting encountered during geofence calculation",
            "message": "Invalid GPS coordinate format."
        }

    # Boundary check: Exactly on boundary or inside
    allowed_radius = float(radius_meters)
    within = (dist <= allowed_radius)

    if within:
        return {
            "within_geofence": True,
            "distance_meters": dist,
            "radius_meters": allowed_radius,
            "geofencing_enabled": True,
            "flagged": False,
            "flag_reason": None,
            "message": f"Verified within geofence: {dist}m from {label} (Limit: {allowed_radius}m)."
        }
    else:
        excess = round(dist - allowed_radius, 1)
        reason = f"Geofence violation: {int(dist)}m from {label} (exceeds {allowed_radius}m limit by {int(excess)}m)"
        return {
            "within_geofence": False,
            "distance_meters": dist,
            "radius_meters": allowed_radius,
            "geofencing_enabled": True,
            "flagged": True,
            "flag_reason": reason,
            "message": f"Location warning: You are {int(dist)}m from {label} (allowed radius: {int(allowed_radius)}m). This event is flagged for administrative review."
        }
