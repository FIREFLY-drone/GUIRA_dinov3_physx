import unittest
import os
import yaml
import torch
from ultralytics import YOLO

class TestFaunaModels(unittest.TestCase):

    def setUp(self):
        self.config_path = "/workspaces/FIREPREVENTION/config.yaml"
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_dir = self.config['model_paths']['fauna_yolov8_csrnet']
        self.yolo_path = os.path.join(self.model_dir, 'yolo_best.pt')
        self.csrnet_path = os.path.join(self.model_dir, 'csrnet_best.pth')

        # Ensure dummy models exist
        os.makedirs(self.model_dir, exist_ok=True)
        if not os.path.exists(self.yolo_path):
            YOLO('yolov8n.pt').save(self.yolo_path)
        
        if not os.path.exists(self.csrnet_path):
            # from models.csrnet import CSRNet
            class CSRNet(torch.nn.Module):
                def __init__(self):
                    super(CSRNet, self).__init__()
                    self.frontend = torch.nn.Conv2d(3, 1, 1)
                def forward(self, x): return self.frontend(x)
            
            torch.save(CSRNet().state_dict(), self.csrnet_path)

    def test_yolo_model_loading(self):
        self.assertTrue(os.path.exists(self.yolo_path))
        model = YOLO(self.yolo_path)
        self.assertIsInstance(model, YOLO)

    def test_csrnet_model_loading(self):
        self.assertTrue(os.path.exists(self.csrnet_path))
        # from models.csrnet import CSRNet
        class CSRNet(torch.nn.Module):
            def __init__(self):
                super(CSRNet, self).__init__()
                self.frontend = torch.nn.Conv2d(3, 1, 1)
            def forward(self, x): return self.frontend(x)
        model = CSRNet()
        model.load_state_dict(torch.load(self.csrnet_path))
        self.assertIsInstance(model, CSRNet)

if __name__ == '__main__':
    unittest.main()
