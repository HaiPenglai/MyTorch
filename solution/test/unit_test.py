import unittest
import os
import sys
import io
import argparse
import glob
import mytorch

class CustomTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:  # __test_verbosity__= 1、2
            self.stream.write("✅\n")
        elif self.dots:
            self.stream.write("✅")
            self.stream.flush()

class CustomTestRunner(unittest.TextTestRunner):
    resultclass = CustomTestResult

class TestRunner:
    def __init__(self, verbose=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.verbose = verbose if verbose is not None else mytorch.__test_verbosity__
        
    def cleanup_pth_files(self):
        for pth_file in glob.glob(os.path.join(self.base_dir, '*.pth')):
            try:
                os.remove(pth_file)
                if self.verbose:
                    print(f"Removed temporary file: {os.path.basename(pth_file)}")
            except OSError as e:
                print(f"Error removing {pth_file}: {e}")
        
    def find_test_files(self):
        test_files = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        return test_files
        
    def run_tests(self):
        self.cleanup_pth_files()
        test_files = self.find_test_files()
        
        original_stdout = sys.stdout
        if not self.verbose:
            sys.stdout = io.StringIO()
        
        all_suites = unittest.TestSuite()
        
        for test_file in test_files:
            print(f"\n🔬 Running {os.path.relpath(test_file, self.base_dir)}")
            suite = unittest.TestLoader().discover(
                os.path.dirname(test_file),
                pattern=os.path.basename(test_file)
            )
            all_suites.addTests(suite)
        
        runner = CustomTestRunner(verbosity=mytorch.__test_verbosity__)
        result = runner.run(all_suites)
        
        if not self.verbose:
            sys.stdout = original_stdout
        
        passed = result.testsRun - len(result.failures) - len(result.errors)
        pass_rate = (passed / result.testsRun) * 100 if result.testsRun > 0 else 0
        
        if result.wasSuccessful():
            print(f"\n✅ ALL TESTS PASSED: {passed}/{result.testsRun} ({pass_rate:.1f}%)")
        else:
            print(f"\n❌ TEST FAILURES: {passed}/{result.testsRun} ({pass_rate:.1f}%)")
        
        self.cleanup_pth_files()
        return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run All MyTorch Tests")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed test output")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    success = runner.run_tests()
    sys.exit(0 if success else 1)