import unittest
import numpy as np
from core.face_engine import (
    detect_face_and_embedding,
    match_face_embedding,
    load_embeddings_cache
)

class TestFaceEngine(unittest.TestCase):
    def test_empty_image(self):
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        loc, emb, err = detect_face_and_embedding(empty_img)
        self.assertIsNone(loc)
        self.assertIsNone(emb)
        self.assertIn("No face detected", err)

    def test_cache_loading(self):
        cache = load_embeddings_cache()
        self.assertIsInstance(cache, list)
        self.assertGreater(len(cache), 0)
        self.assertIn("student_id", cache[0])
        self.assertIn("vector", cache[0])
        self.assertEqual(len(cache[0]["vector"]), 128)

    def test_matching_with_exact_vector(self):
        cache = load_embeddings_cache()
        first_student = cache[0]
        # Match with its exact vector
        res = match_face_embedding(first_student["vector"])
        self.assertTrue(res["matched"])
        self.assertEqual(res["student_id"], first_student["student_id"])
        self.assertGreaterEqual(res["confidence"], 0.90)

    def test_matching_with_random_noise(self):
        # A random unit vector should not match any enrolled person
        noise = np.random.randn(128)
        noise /= np.linalg.norm(noise)
        res = match_face_embedding(noise)
        # Even if thresholding, random vector distance is ~sqrt(2) ≈ 1.41
        self.assertFalse(res["matched"])
        self.assertEqual(res["status"], "failed")

if __name__ == "__main__":
    unittest.main()
