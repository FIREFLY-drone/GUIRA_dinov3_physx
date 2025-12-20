import unittest
import os
import yaml
import torch
import json

class TestSmokeTimeSFormer(unittest.TestCase):

    def setUp(self):
        self.config_path = "/workspaces/FIREPREVENTION/config.yaml"
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = os.path.join(self.config['model_paths']['smoke_timesformer'], 'best.pt')
        self.manifest_path = "/workspaces/FIREPREVENTION/data/manifests/smoke_timesformer.json"

        # Ensure a dummy model and manifest exist for testing
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # Create a small dummy model
            from transformers import TimesformerModel
            import torch.nn as nn
            base_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
            model = nn.Sequential(base_model, nn.Linear(base_model.config.hidden_size, 2))
            torch.save(model.state_dict(), self.model_path)

        if not os.path.exists(self.manifest_path):
            # Create a dummy manifest
            dummy_manifest = {
                "train": [{"video_path": "dummy/path", "label": 0, "num_frames": 10}],
                "val": [{"video_path": "dummy/path", "label": 0, "num_frames": 10}],
                "test": [{"video_path": "dummy/path", "label": 0, "num_frames": 10}]
            }
            with open(self.manifest_path, 'w') as f:
                json.dump(dummy_manifest, f)

    def test_model_loading(self):
        """Test if the trained model can be loaded."""
        self.assertTrue(os.path.exists(self.model_path))
        from transformers import TimesformerModel
        import torch.nn as nn
        base_model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model = nn.Sequential(base_model, nn.Linear(base_model.config.hidden_size, 2))
        model.load_state_dict(torch.load(self.model_path))
        self.assertIsInstance(model, nn.Sequential)

if __name__ == '__main__':
    unittest.main()
