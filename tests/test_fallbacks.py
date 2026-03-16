import unittest

from controllers.fallbacks import FallbackConfig, FallbackManager


class TestFallbacks(unittest.TestCase):
    def test_no_error_no_fallback(self):
        f = FallbackManager()
        out = f.resolve_mode(False, "controller")
        self.assertFalse(out["fallback_applied"])
        self.assertEqual(out["mode"], "controller")

    def test_controller_error_to_native(self):
        f = FallbackManager()
        out = f.resolve_mode(True, "controller")
        self.assertTrue(out["fallback_applied"])
        self.assertEqual(out["mode"], "native_eagle3")

    def test_native_error_to_no_sd(self):
        f = FallbackManager()
        out = f.resolve_mode(True, "native_eagle3")
        self.assertTrue(out["fallback_applied"])
        self.assertEqual(out["mode"], "no_sd")

    def test_force_mode(self):
        f = FallbackManager(FallbackConfig(force_mode="observe_only"))
        out = f.resolve_mode(False, "controller")
        self.assertTrue(out["fallback_applied"])
        self.assertEqual(out["mode"], "observe_only")


if __name__ == "__main__":
    unittest.main()
