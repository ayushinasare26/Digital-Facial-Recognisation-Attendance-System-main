import os
import unittest
import numpy as np
from PIL import Image
from core.geotag import reverse_geocode, stamp_geotag

class TestGeotag(unittest.TestCase):
    def test_reverse_geocode_fallback_on_invalid(self):
        # When None or invalid coordinates are passed
        res = reverse_geocode(None, None)
        self.assertIn("Unknown Location", res)

    def test_reverse_geocode_valid_or_fallback(self):
        # Mumbai coordinates - should either resolve via OSM or return fallback gracefully
        res = reverse_geocode(19.0760, 72.8777)
        self.assertIsInstance(res, str)
        self.assertGreater(len(res), 5)

    def test_stamp_geotag_creation(self):
        # Create a test blank image
        test_img = np.zeros((300, 400, 3), dtype=np.uint8)
        test_img[:, :] = [100, 150, 200]
        
        output_rel = stamp_geotag(
            image_input=test_img,
            student_name="Test Student",
            student_id=999,
            timestamp_str="2026-09-27 21:00:00 UTC",
            latitude=19.0760,
            longitude=72.8777,
            address="Mumbai, Maharashtra, India",
            confidence=0.95,
            status="success"
        )
        self.assertTrue(os.path.exists(output_rel))
        with Image.open(output_rel) as img:
            self.assertEqual(img.size, (400, 300))

    def tearDown(self):
        if os.path.exists("attendance_photos"):
            for f in os.listdir("attendance_photos"):
                if f.startswith("999_"):
                    try:
                        os.remove(os.path.join("attendance_photos", f))
                    except OSError:
                        pass

if __name__ == "__main__":
    unittest.main()
