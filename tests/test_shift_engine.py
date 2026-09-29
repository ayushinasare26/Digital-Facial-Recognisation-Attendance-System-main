"""
tests/test_shift_engine.py - Unit Tests for Industrial Shift Engine
Covers:
1. On-time check-in (within grace period).
2. Late check-in (exceeds grace period).
3. Early check-out (leaves before scheduled end).
4. Normal on-time check-out.
5. Overtime calculation for extended hours.
6. Edge case: Check-out before check-in (invalid sequence).
7. Edge case: Missing check-out (incomplete event).
8. Break deduction for shifts >= 4 hours.
"""

import unittest
import datetime
from core.shift_engine import (
    evaluate_check_in,
    evaluate_check_out,
    calculate_shift_hours
)

class TestShiftEngine(unittest.TestCase):
    def setUp(self):
        # Standard Shift: 09:00 to 18:00 (9 hours gross, 60m break -> 8h regular scheduled)
        self.shift = {
            "name": "General Shift",
            "start_time": "09:00",
            "end_time": "18:00",
            "grace_period_minutes": 15,
            "break_duration_minutes": 60
        }

    def test_on_time_check_in_exact(self):
        """Checking in at 09:00 exact should be on_time."""
        check_in = "2026-09-29T09:00:00"
        res = evaluate_check_in(check_in, self.shift)
        self.assertEqual(res["status"], "on_time")
        self.assertEqual(res["minutes_late"], 0.0)

    def test_on_time_check_in_within_grace(self):
        """Checking in at 09:12 (within 15m grace) should still be on_time."""
        check_in = "2026-09-29T09:12:00"
        res = evaluate_check_in(check_in, self.shift)
        self.assertEqual(res["status"], "on_time")
        self.assertEqual(res["minutes_late"], 0.0)

    def test_late_check_in(self):
        """Checking in at 09:35 (20 mins past grace limit) must be marked late."""
        check_in = "2026-09-29T09:35:00"
        res = evaluate_check_in(check_in, self.shift)
        self.assertEqual(res["status"], "late")
        self.assertEqual(res["minutes_late"], 35.0)

    def test_early_leave(self):
        """Checking out at 16:30 (1.5 hours before 18:00) must be early_leave."""
        check_out = "2026-09-29T16:30:00"
        res = evaluate_check_out(check_out, self.shift)
        self.assertEqual(res["status"], "early_leave")
        self.assertEqual(res["minutes_early"], 90.0)

    def test_overtime_calculation(self):
        """Working from 09:00 to 20:00 (11 hours gross - 1h break = 10h net -> 2h overtime)."""
        check_in = "2026-09-29T09:00:00"
        check_out = "2026-09-29T20:00:00"
        res = calculate_shift_hours(check_in, check_out, self.shift)
        self.assertEqual(res["gross_hours"], 11.0)
        self.assertEqual(res["net_hours"], 10.0)
        self.assertEqual(res["regular_hours"], 8.0)
        self.assertEqual(res["overtime_hours"], 2.0)
        self.assertEqual(res["status"], "overtime")

    def test_checkout_before_checkin_edge_case(self):
        """Check-out timestamp earlier than check-in must be flagged as invalid sequence."""
        check_in = "2026-09-29T14:00:00"
        check_out = "2026-09-29T09:00:00"
        res = calculate_shift_hours(check_in, check_out, self.shift)
        self.assertEqual(res["status"], "invalid_sequence")
        self.assertTrue(res["flagged"])
        self.assertIn("Corrupted sequence", res["flag_reason"])

    def test_missing_checkout_edge_case(self):
        """Check-in without corresponding check-out must return missing_checkout."""
        check_in = "2026-09-29T09:00:00"
        res = calculate_shift_hours(check_in, None, self.shift)
        self.assertEqual(res["status"], "missing_checkout")
        self.assertTrue(res["flagged"])
        self.assertIn("missing", res["flag_reason"].lower())

    def test_break_deduction(self):
        """Shift of 8 hours gross should have 1 hour break deducted -> 7 net hours."""
        check_in = "2026-09-29T09:00:00"
        check_out = "2026-09-29T17:00:00"
        res = calculate_shift_hours(check_in, check_out, self.shift)
        self.assertEqual(res["gross_hours"], 8.0)
        self.assertEqual(res["net_hours"], 7.0)

if __name__ == "__main__":
    unittest.main()
