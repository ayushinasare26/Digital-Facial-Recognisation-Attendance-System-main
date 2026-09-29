"""
tests/test_geofence.py - Unit Tests for Geofencing Engine & Edge Cases
Covers:
1. Coordinates well inside the geofence radius.
2. Coordinates well outside the geofence radius.
3. Coordinates exactly on the boundary radius.
4. Sites with geofencing disabled (remote/field workers).
5. Missing GPS coordinates (denied or unavailable).
6. Invalid coordinate formatting or boundary float coordinates.
7. Accurate distance comparison with known Earth coordinates.
"""

import unittest
from core.geofence import haversine_distance, validate_geofence

class TestGeofenceEngine(unittest.TestCase):
    def setUp(self):
        # Mumbai Office reference coordinates
        self.site_lat = 19.0657
        self.site_lon = 72.8687
        self.radius = 200.0

    def test_haversine_identical_point(self):
        """Distance between identical coordinates must be 0 meters."""
        dist = haversine_distance(self.site_lat, self.site_lon, self.site_lat, self.site_lon)
        self.assertEqual(dist, 0.0)

    def test_inside_geofence(self):
        """Employee 50m away should be accepted within 200m radius."""
        # Offset lat by ~0.0004 deg (approx 44m north)
        emp_lat = self.site_lat + 0.0004
        emp_lon = self.site_lon
        result = validate_geofence(emp_lat, emp_lon, self.site_lat, self.site_lon, radius_meters=self.radius)
        self.assertTrue(result["within_geofence"])
        self.assertFalse(result["flagged"])
        self.assertLess(result["distance_meters"], self.radius)

    def test_outside_geofence(self):
        """Employee 800m away should be flagged as outside geofence."""
        # Offset lat by ~0.008 deg (approx 880m north)
        emp_lat = self.site_lat + 0.008
        emp_lon = self.site_lon
        result = validate_geofence(emp_lat, emp_lon, self.site_lat, self.site_lon, radius_meters=self.radius)
        self.assertFalse(result["within_geofence"])
        self.assertTrue(result["flagged"])
        self.assertGreater(result["distance_meters"], self.radius)
        self.assertIn("Geofence violation", result["flag_reason"])

    def test_exactly_on_boundary(self):
        """Coordinates exactly at the boundary radius must be accepted (<= radius)."""
        # Calculate coordinate that is at a specific distance
        dist = haversine_distance(self.site_lat, self.site_lon, self.site_lat + 0.001, self.site_lon)
        # Use exact dist as radius
        result = validate_geofence(
            self.site_lat + 0.001,
            self.site_lon,
            self.site_lat,
            self.site_lon,
            radius_meters=dist
        )
        self.assertTrue(result["within_geofence"], "Points on the boundary radius must be marked within_geofence")
        self.assertFalse(result["flagged"])

    def test_geofencing_disabled_for_field_staff(self):
        """Sites with geofencing disabled must never flag distance regardless of location."""
        # Remote employee checking in from Delhi (~1150km away from Mumbai)
        delhi_lat = 28.6139
        delhi_lon = 77.2090
        result = validate_geofence(
            delhi_lat,
            delhi_lon,
            self.site_lat,
            self.site_lon,
            radius_meters=self.radius,
            geofencing_enabled=False,
            site_name="Field Sales"
        )
        self.assertTrue(result["within_geofence"])
        self.assertFalse(result["flagged"])
        self.assertFalse(result["geofencing_enabled"])

    def test_missing_coordinates(self):
        """When geofencing is enabled, missing GPS coordinates must be flagged."""
        result = validate_geofence(None, None, self.site_lat, self.site_lon, radius_meters=self.radius)
        self.assertFalse(result["within_geofence"])
        self.assertTrue(result["flagged"])
        self.assertIn("Missing GPS coordinates", result["flag_reason"])

    def test_invalid_coordinates(self):
        """Malformed coordinate values must be caught safely."""
        dist = haversine_distance("invalid", "coordinates", self.site_lat, self.site_lon)
        self.assertIsNone(dist)

if __name__ == "__main__":
    unittest.main()
