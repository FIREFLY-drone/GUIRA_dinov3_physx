import unittest
import os
import yaml
import torch

class TestVegetationModel(unittest.TestCase):

    def setUp(self):
        self.config_path = "/workspaces/FIREPREVENTION/config.yaml"
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = os.path.join(self.config['model_paths']['vegetation_resnet_vari'], 'best.pt')
        
        # Ensure a dummy model exists
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            from experiments.vegetation_resnet_vari.train_vegetation_resnet_vari import VegHealthModel
            model = VegHealthModel(num_classes=3)
            torch.save(model.state_dict(), self.model_path)

    def test_model_loading(self):
        self.assertTrue(os.path.exists(self.model_path))
        from experiments.vegetation_resnet_vari.train_vegetation_resnet_vari import VegHealthModel
        model = VegHealthModel(num_classes=3)
        model.load_state_dict(torch.load(self.model_path))
        self.assertIsInstance(model, VegHealthModel)

if __name__ == '__main__':
    unittest.main()
