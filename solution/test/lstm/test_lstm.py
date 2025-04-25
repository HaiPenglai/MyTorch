import unittest
import torch
import torch.nn as nn
import mytorch
from mytorch.mynn import CustomLSTM

class TestCustomLSTM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_size = 3
        self.hidden_size = 10
        self.num_layers = 2
        self.batch_size = 2
        self.seq_len = 5
        
        self.lstm = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)
        torch.save(self.lstm.state_dict(), 'lstm_model.pth')
        
        self.custom_lstm = CustomLSTM(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers)
        self.custom_lstm.load_state_dict(torch.load('lstm_model.pth', weights_only=True))
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
    def test_CustomLSTM_loading(self):
        native_state = self.lstm.state_dict()
        custom_state = self.custom_lstm.state_dict()
        
        self.assertEqual(set(native_state.keys()), set(custom_state.keys()),
                       "Parameter names don't match")
        
        for name in native_state:
            self.assertTrue(torch.allclose(native_state[name], custom_state[name], atol=1e-6),
                          f"Parameter {name} values don't match")
            
    def test_CustomLSTM_forward(self):
        output_native, (hidden_native, cell_native) = self.lstm(self.x)
        output_custom, (hidden_custom, cell_custom) = self.custom_lstm(self.x)
        
        self.assertTrue(
            torch.allclose(output_custom, output_native, atol=1e-6),
            "Forward pass outputs differ"
        )
        
        self.assertTrue(
            torch.allclose(hidden_custom, hidden_native, atol=1e-6),
            "Hidden states differ"
        )
        
        self.assertTrue(
            torch.allclose(cell_custom, cell_native, atol=1e-6),
            "Cell states differ"
        )

if __name__ == '__main__':
    unittest.main(verbosity=mytorch.__test_verbosity__)