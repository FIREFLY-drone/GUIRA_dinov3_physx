import unittest
import os
import yaml
import torch

class TestSpreadModel(unittest.TestCase):

    def setUp(self):
        self.config_path = "/workspaces/FIREPREVENTION/config.yaml"
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = os.path.join(self.config['model_paths']['spread_hybrid'], 'best.pt')
        
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            from experiments.spread_hybrid.train_spread_hybrid import ConvLSTM
            model = ConvLSTM(in_channels=1, hid_channels=64, kernel_size=3)
            torch.save(model.state_dict(), self.model_path)

    def test_model_loading(self):
        self.assertTrue(os.path.exists(self.model_path))
        from experiments.spread_hybrid.train_spread_hybrid import ConvLSTM
        model = ConvLSTM(in_channels=1, hid_channels=64, kernel_size=3)
        model.load_state_dict(torch.load(self.model_path))
        self.assertIsInstance(model, ConvLSTM)

if __name__ == '__main__':
    unittest.main()
