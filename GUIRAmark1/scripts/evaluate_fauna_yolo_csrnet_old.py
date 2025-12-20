import yaml
from ultralytics import YOLO
import torch
import os
import json
from sklearn.metrics import mean_absolute_error

# from models.csrnet import CSRNet
class CSRNet(torch.nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.backend = torch.nn.Conv2d(64, 1, 1)
    def forward(self, x):
        x = self.frontend(x)
        return self.backend(x)

def evaluate_fauna_models():
    with open("/workspaces/FIREPREVENTION/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    fauna_config = config['training']['fauna_yolov8_csrnet']
    model_dir = config['model_paths']['fauna_yolov8_csrnet']

    # --- Evaluate YOLOv8 ---
    print("--- Evaluating YOLOv8 ---")
    yolo_model = YOLO(os.path.join(model_dir, 'yolo_best.pt'))
    yolo_model.val(data=os.path.abspath(config['data_dir'] + '/manifests/fauna_yolov8.yaml'))

    # --- Evaluate CSRNet ---
    print("\n--- Evaluating CSRNet ---")
    csrnet_model = CSRNet()
    csrnet_model.load_state_dict(torch.load(os.path.join(model_dir, 'csrnet_best.pth')))
    device = torch.device(config['device'])
    csrnet_model.to(device)
    csrnet_model.eval()

    # Dummy evaluation
    # A proper dataloader for the test set would be needed.
    # with open(os.path.join(config['data_dir'], 'manifests/fauna_csrnet.json'), 'r') as f:
    #     manifest = json.load(f)
    # test_dataset = CSRNetDataset(manifest['val'], ...)
    # test_loader = DataLoader(test_dataset, ...)
    
    mae = 0
    # with torch.no_grad():
    #     for img, target in test_loader:
    #         img = img.to(device)
    #         output = csrnet_model(img)
    #         gt_count = target.sum()
    #         et_count = output.sum()
    #         mae += abs(gt_count - et_count)
    # print(f"CSRNet MAE: {mae / len(test_loader)}")
    print("CSRNet MAE: (dummy value) 5.0")


if __name__ == '__main__':
    evaluate_fauna_models()
