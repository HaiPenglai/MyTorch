import unittest

class TestAdd(unittest.TestCase):
    def test_add_integers(self):
        self.assertEqual(1 + 2, 3)
        
    def test_add_floats(self):
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)