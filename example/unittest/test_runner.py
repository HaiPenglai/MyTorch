import unittest
import os

def run_tests():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 自动发现所有测试
    loader = unittest.TestLoader()
    
    # 发现math_tests目录下的测试
    math_suite = loader.discover(
        start_dir=os.path.join(script_dir, 'math_tests'),
        pattern='test_*.py'
    )
    
    # 发现string_tests目录下的测试
    string_suite = loader.discover(
        start_dir=os.path.join(script_dir, 'string_tests'),
        pattern='test_*.py'
    )
    
    # 合并测试套件
    combined_suite = unittest.TestSuite()
    combined_suite.addTests(math_suite)
    combined_suite.addTests(string_suite)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # 输出总结
    passed = result.testsRun - len(result.failures) - len(result.errors)
    pass_rate = (passed / result.testsRun) * 100 if result.testsRun > 0 else 0
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED: {passed}/{result.testsRun} ({pass_rate:.1f}%)")
    else:
        print(f"❌ TEST FAILURES: {passed}/{result.testsRun} ({pass_rate:.1f}%)")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_tests()