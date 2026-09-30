import io
import unittest
import numpy as np
from PIL import Image

import core.face_engine as fe
import core.liveness as lm
from core.liveness import verify_liveness
from core.face_engine import detect_face_and_embedding, match_face_embedding
from core.pipeline import run_attendance_pipeline

class TestServerlessResilience(unittest.TestCase):
    def setUp(self):
        self.orig_fe_avail = fe.FACE_RECOGNITION_AVAILABLE
        self.orig_fe_fr = fe.face_recognition
        self.orig_lm_fr = lm.face_recognition

    def tearDown(self):
        fe.FACE_RECOGNITION_AVAILABLE = self.orig_fe_avail
        fe.face_recognition = self.orig_fe_fr
        lm.face_recognition = self.orig_lm_fr

    def test_liveness_in_serverless_environment(self):
        # Force serverless condition (no dlib / face_recognition)
        lm.face_recognition = None

        # Simulate live camera frames with natural sensor noise
        f1 = np.random.randint(60, 180, (240, 320, 3), dtype=np.uint8)
        f2 = f1.copy() + np.random.randint(-3, 4, (240, 320, 3)).astype(np.uint8)
        f3 = f2.copy() + np.random.randint(-3, 4, (240, 320, 3)).astype(np.uint8)

        res = verify_liveness([f1, f2, f3], challenge_type="blink", is_demo=False)
        self.assertTrue(res["passed"], f"Liveness check should pass for live frames: {res}")
        self.assertIn("Live camera", res["reason"])

        # Simulate static photo spoof attempt (identical frames)
        res_spoof = verify_liveness([f1, f1, f1], challenge_type="blink", is_demo=False)
        self.assertFalse(res_spoof["passed"], "Static identical frames should be flagged as spoof")
        self.assertIn("Static image", res_spoof["reason"])

    def test_pipeline_in_serverless_environment(self):
        fe.FACE_RECOGNITION_AVAILABLE = False
        fe.face_recognition = None
        lm.face_recognition = None

        f1 = np.random.randint(60, 180, (240, 320, 3), dtype=np.uint8)
        f2 = f1.copy() + np.random.randint(-3, 4, (240, 320, 3)).astype(np.uint8)
        f3 = f2.copy() + np.random.randint(-3, 4, (240, 320, 3)).astype(np.uint8)

        buf1 = io.BytesIO()
        Image.fromarray(f1).save(buf1, format='JPEG')
        b1 = buf1.getvalue()
        buf2 = io.BytesIO()
        Image.fromarray(f2).save(buf2, format='JPEG')
        b2 = buf2.getvalue()
        buf3 = io.BytesIO()
        Image.fromarray(f3).save(buf3, format='JPEG')
        b3 = buf3.getvalue()

        result = run_attendance_pipeline(
            primary_image_data=b1,
            latitude="19.0657",
            longitude="72.8687",
            liveness_frames=[b1, b2, b3],
            challenge_type="blink",
            is_demo=False,
            bypass_cooldown=True,
            event_type="check_in",
            expected_employee_id=24
        )
        self.assertTrue(result.get("success"), f"Pipeline failed: {result.get('message')}")
        self.assertEqual(result.get("student_id"), 24)
        self.assertEqual(result.get("student_name"), "Ayushi Nasare")
        self.assertEqual(len(result.get("stages", [])), 9)

if __name__ == "__main__":
    unittest.main()
