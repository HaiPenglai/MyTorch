import unittest

class TestUpper(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')