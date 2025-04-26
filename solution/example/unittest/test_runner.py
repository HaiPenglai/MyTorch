import unittest
import os

def run_tests():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    loader = unittest.TestLoader()
    
    math_suite = loader.discover(
        start_dir=os.path.join(script_dir, 'math_tests'),
        pattern='test_*.py'
    )
    
    string_suite = loader.discover(
        start_dir=os.path.join(script_dir, 'string_tests'),
        pattern='test_*.py'
    )
    
    combined_suite = unittest.TestSuite()
    combined_suite.addTests(math_suite)
    combined_suite.addTests(string_suite)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    passed = result.testsRun - len(result.failures) - len(result.errors)
    pass_rate = (passed / result.testsRun) * 100 if result.testsRun > 0 else 0
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED: {passed}/{result.testsRun} ({pass_rate:.1f}%)")
    else:
        print(f"❌ TEST FAILURES: {passed}/{result.testsRun} ({pass_rate:.1f}%)")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_tests()