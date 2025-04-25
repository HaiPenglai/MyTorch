import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomNorm

class TestCustomNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.test_cases = [
            # (p, dim, keepdim, input_shape)
            (1, None, False, (5,)),          # 1范数，向量
            (1, 0, True, (3, 4)),            # 1范数，矩阵，沿dim=0
            (2, None, False, (3, 4)),        # 2范数，矩阵
            (2, 1, True, (2, 3, 4)),         # 2范数，沿dim=1
            (3, -1, True, (2, 2, 2)),        # 3范数，最后一维
            (0, None, False, (5,))           # 0范数
        ]
        
    def test_norm_values(self):
        for p, dim, keepdim, shape in self.test_cases:
            with self.subTest(p=p, dim=dim, keepdim=keepdim):
                x = torch.randn(shape)
                custom_norm = CustomNorm(p=p, dim=dim, keepdim=keepdim)
                
                if p == 0:
                    # 自定义0范数实现
                    if dim is not None:
                        with self.assertRaises(ValueError):
                            custom_norm(x)
                        continue
                    expected = torch.sum(x != 0).to(x.dtype)
                else:
                    # 自定义实现
                    if p == 1:
                        expected = torch.sum(torch.abs(x), dim=dim, keepdim=keepdim)
                    elif p == 2:
                        expected = torch.sqrt(torch.sum(torch.pow(x, 2), dim=dim, keepdim=keepdim))
                    else:
                        expected = torch.pow(
                            torch.sum(torch.pow(torch.abs(x), p), dim=dim, keepdim=keepdim),
                            1/p
                        )
                
                actual = custom_norm(x)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-6),
                    f"Failed for p={p}, dim={dim}, keepdim={keepdim}\n"
                    f"Expected: {expected}\nActual: {actual}"
                )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)