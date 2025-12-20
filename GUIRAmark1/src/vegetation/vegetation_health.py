"""
Vegetation and Tree Health Assessment using CNN + VARI.
Analyzes vegetation health using RGB imagery and VARI index.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import yaml
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class VegetationDataset(Dataset):
    """Dataset for vegetation health classification."""
    
    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 224, 
                 use_vari: bool = True, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.use_vari = use_vari
        self.augment = augment and split == 'train'
        
        # Load image and mask paths
        self.images_dir = self.data_dir / 'images' / split
        self.masks_dir = self.data_dir / 'masks' / split
        
        self.image_paths = []
        self.mask_paths = []
        
        if self.images_dir.exists():
            for img_path in self.images_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    mask_path = self.masks_dir / (img_path.stem + '.png')
                    if mask_path.exists():
                        self.image_paths.append(str(img_path))
                        self.mask_paths.append(str(mask_path))
        
        logger.info(f"Loaded {len(self.image_paths)} image-mask pairs for {split} split")
        
        # Class mapping: 0=healthy, 1=dry, 2=burned
        self.classes = ['healthy', 'dry', 'burned']
        self.num_classes = len(self.classes)
        
        # Augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
    
    def __len__(self):
        return len(self.image_paths)
    
    def compute_vari(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Visible Atmospherically Resistant Index (VARI).
        VARI = (Green - Red) / (Green + Red - Blue)
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with shape (H, W, 3)")
        
        # Normalize to [0, 1] if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        
        # Avoid division by zero
        denominator = g + r - b
        denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
        
        vari = (g - r) / denominator
        
        # Clip to reasonable range
        vari = np.clip(vari, -1, 1)
        
        return vari
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Try loading as TIFF
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_tensor = torch.tensor(transformed['mask'], dtype=torch.long)
        
        # Add VARI channel if enabled
        if self.use_vari:
            # Convert tensor back to numpy for VARI computation
            image_np = image_tensor.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            vari = self.compute_vari(image_np)
            vari_tensor = torch.tensor(vari, dtype=torch.float32).unsqueeze(0)
            
            # Concatenate VARI channel
            image_tensor = torch.cat([image_tensor, vari_tensor], dim=0)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_path': img_path,
            'mask_path': mask_path
        }


class VegetationHealthCNN(nn.Module):
    """CNN for vegetation health classification with optional VARI channel."""
    
    def __init__(self, num_classes: int = 3, use_vari: bool = True, backbone: str = 'resnet50', 
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_vari = use_vari
        self.backbone_name = backbone
        
        # Input channels: 3 (RGB) + 1 (VARI) if use_vari else 3 (RGB only)
        input_channels = 4 if use_vari else 3
        
        # Load backbone
        if backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
            feature_dim = backbone_model.fc.in_features
            
            # Modify first conv layer for 4 channels if using VARI
            if use_vari:
                original_conv = backbone_model.conv1
                backbone_model.conv1 = nn.Conv2d(
                    input_channels,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias
                )
                
                # Initialize new conv layer
                with torch.no_grad():
                    # Copy RGB weights
                    backbone_model.conv1.weight[:, :3] = original_conv.weight
                    # Initialize VARI channel weights
                    backbone_model.conv1.weight[:, 3:] = original_conv.weight[:, :1]  # Use red channel weights
            
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            
        elif backbone == 'resnet34':
            backbone_model = models.resnet34(pretrained=pretrained)
            feature_dim = backbone_model.fc.in_features
            
            if use_vari:
                original_conv = backbone_model.conv1
                backbone_model.conv1 = nn.Conv2d(
                    input_channels,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias
                )
                
                with torch.no_grad():
                    backbone_model.conv1.weight[:, :3] = original_conv.weight
                    backbone_model.conv1.weight[:, 3:] = original_conv.weight[:, :1]
            
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Segmentation head (for pixel-level classification)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)  # Upsample to input size
        )
    
    def forward(self, x, return_features: bool = False):
        # Extract features
        features = self.backbone(x)
        
        # Global classification
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        classification_logits = self.classifier(global_features)
        
        # Pixel-level segmentation (if needed)
        if return_features:
            # Get feature map before final pooling
            feature_map = features
            segmentation_logits = self.segmentation_head(feature_map)
            return classification_logits, segmentation_logits
        
        return classification_logits


class VegetationHealthTrainer:
    """Trainer for vegetation health assessment."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = VegetationHealthCNN(
            num_classes=len(config.get('classes', ['healthy', 'dry', 'burned'])),
            use_vari=config.get('vari_enabled', True),
            backbone='resnet50',
            pretrained=True
        ).to(self.device)
        
        # Loss function with class weights (to handle imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=1e-4
        )
          # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        logger.info(f"Initialized VegetationHealthCNN on {self.device}")
    
    def prepare_data(self, data_dir: str):
        """Prepare training and validation datasets."""
        self.train_dataset = VegetationDataset(
            data_dir=data_dir,
            split='train',
            img_size=self.config.get('img_size', 224),
            use_vari=self.config.get('vari_enabled', True),
            augment=True
        )
        
        self.val_dataset = VegetationDataset(
            data_dir=data_dir,
            split='val',
            img_size=self.config.get('img_size', 224),
            use_vari=self.config.get('vari_enabled', True),
            augment=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 24),
            shuffle=True,
            num_workers=self.config.get('workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 24),
            shuffle=False,
            num_workers=self.config.get('workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Prepared datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            
            # For now, use majority class from mask as image-level label
            masks = batch['mask'].to(self.device)
            # Get majority class for each image
            labels = []
            for mask in masks:
                unique, counts = torch.unique(mask, return_counts=True)
                majority_class = unique[torch.argmax(counts)]
                labels.append(majority_class)
            labels = torch.stack(labels)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 20 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Get majority class labels
                labels = []
                for mask in masks:
                    unique, counts = torch.unique(mask, return_counts=True)
                    majority_class = unique[torch.argmax(counts)]
                    labels.append(majority_class)
                labels = torch.stack(labels)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate detailed metrics
        class_names = self.config.get('classes', ['healthy', 'dry', 'burned'])
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        return total_loss / len(self.val_loader), 100. * correct / total, report
    
    def train(self, epochs: int = 35, save_dir: str = 'models'):
        """Train the model."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_acc = 0
        
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_report = self.validate()
            
            self.scheduler.step(val_loss)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Log per-class metrics
            for class_name in self.config.get('classes', ['healthy', 'dry', 'burned']):
                if class_name in val_report:
                    metrics = val_report[class_name]
                    logger.info(f'{class_name}: P={metrics["precision"]:.3f}, '
                              f'R={metrics["recall"]:.3f}, F1={metrics["f1-score"]:.3f}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = Path(save_dir) / 'vegetation_resnet50_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': self.config,
                    'val_report': val_report
                }, model_path)
                logger.info(f'New best model saved: {val_acc:.2f}%')
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = Path(save_dir) / f'vegetation_resnet50_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config
                }, checkpoint_path)
        
        logger.info(f'Training completed. Best validation accuracy: {best_acc:.2f}%')


class VegetationHealthInference:
    """Inference class for vegetation health assessment."""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.classes = config.get('classes', ['healthy', 'dry', 'burned'])
        self.use_vari = config.get('vari_enabled', True)
        self.img_size = config.get('img_size', 224)
        
        # Initialize model
        self.model = VegetationHealthCNN(
            num_classes=len(self.classes),
            use_vari=self.use_vari,
            backbone='resnet50',
            pretrained=False
        ).to(self.device)
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded vegetation health model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using random weights")
        
        self.model.eval()
        
        # Transform for preprocessing
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def compute_vari(self, image: np.ndarray) -> np.ndarray:
        """Compute VARI index."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with shape (H, W, 3)")
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        
        denominator = g + r - b
        denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)
        
        vari = (g - r) / denominator
        vari = np.clip(vari, -1, 1)
        
        return vari
    
    def predict_health(self, image: np.ndarray) -> Dict:
        """
        Predict vegetation health for an image.
        
        Args:
            image: Input image as numpy array (BGR format)
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply transform
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image']
        
        # Add VARI channel if enabled
        if self.use_vari:
            # Convert tensor back to numpy for VARI computation
            image_np = image_tensor.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            vari = self.compute_vari(image_np)
            vari_tensor = torch.tensor(vari, dtype=torch.float32).unsqueeze(0)
            
            # Concatenate VARI channel
            image_tensor = torch.cat([image_tensor, vari_tensor], dim=0)
        
        # Add batch dimension and move to device
        input_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Convert to numpy
        probs = probabilities.squeeze().cpu().numpy()
        
        result = {
            'predicted_class': self.classes[predicted_class],
            'predicted_class_id': predicted_class,
            'confidence': float(probs[predicted_class]),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.classes, probs)
            }
        }
        
        return result
    
    def predict_patch_map(self, image: np.ndarray, patch_size: int = 256, stride: int = 128) -> np.ndarray:
        """
        Predict vegetation health for image patches to create a health map.
        
        Args:
            image: Input image as numpy array
            patch_size: Size of patches to extract
            stride: Stride between patches
        
        Returns:
            Health map as numpy array
        """
        h, w = image.shape[:2]
        health_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Extract patches and predict
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                if patch.shape[:2] == (patch_size, patch_size):
                    result = self.predict_health(patch)
                    class_id = result['predicted_class_id']
                    confidence = result['confidence']
                    
                    # Weight by confidence
                    health_map[y:y+patch_size, x:x+patch_size] += class_id * confidence
                    count_map[y:y+patch_size, x:x+patch_size] += confidence
        
        # Normalize by count
        health_map = np.divide(health_map, count_map, 
                             out=np.zeros_like(health_map), 
                             where=count_map != 0)
        
        return health_map
    
    def analyze_vegetation_health(self, image: np.ndarray, save_map: bool = False, 
                                output_path: str = 'outputs/vegetation_map.png') -> Dict:
        """
        Comprehensive vegetation health analysis.
        
        Args:
            image: Input image
            save_map: Whether to save health map
            output_path: Path to save health map
        
        Returns:
            Analysis results
        """
        # Overall image classification
        overall_result = self.predict_health(image)
        
        # Patch-based analysis for spatial health map
        health_map = self.predict_patch_map(image)
        
        # Calculate statistics
        total_pixels = health_map.size
        healthy_pixels = np.sum((health_map >= 0) & (health_map < 0.5))
        dry_pixels = np.sum((health_map >= 0.5) & (health_map < 1.5))
        burned_pixels = np.sum(health_map >= 1.5)
        
        analysis = {
            'overall_health': overall_result,
            'spatial_analysis': {
                'healthy_ratio': float(healthy_pixels / total_pixels),
                'dry_ratio': float(dry_pixels / total_pixels),
                'burned_ratio': float(burned_pixels / total_pixels)
            },
            'health_map': health_map,
            'risk_score': self._calculate_vegetation_risk(health_map)
        }
        
        if save_map:
            self._save_health_map(health_map, output_path)
        
        return analysis
    
    def _calculate_vegetation_risk(self, health_map: np.ndarray) -> float:
        """Calculate vegetation fire risk score based on health map."""
        # Risk increases with dry and burned vegetation
        dry_ratio = np.sum((health_map >= 0.5) & (health_map < 1.5)) / health_map.size
        burned_ratio = np.sum(health_map >= 1.5) / health_map.size
        
        risk_score = 0.6 * dry_ratio + 0.9 * burned_ratio
        return float(np.clip(risk_score, 0, 1))
    
    def _save_health_map(self, health_map: np.ndarray, output_path: str):
        """Save health map as color-coded image."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create color map
        colored_map = np.zeros((*health_map.shape, 3), dtype=np.uint8)
        
        # Healthy = Green
        healthy_mask = (health_map >= 0) & (health_map < 0.5)
        colored_map[healthy_mask] = [0, 255, 0]
        
        # Dry = Orange
        dry_mask = (health_map >= 0.5) & (health_map < 1.5)
        colored_map[dry_mask] = [255, 165, 0]
        
        # Burned = Red
        burned_mask = health_map >= 1.5
        colored_map[burned_mask] = [255, 0, 0]
        
        # Save
        cv2.imwrite(output_path, cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR))
        logger.info(f"Health map saved to {output_path}")


def create_sample_vegetation_data():
    """Create sample vegetation data for testing."""
    # This would create synthetic data with health labels
    sample_data = {
        'images': ['healthy_forest.tif', 'dry_grassland.jpg', 'burned_area.png'],
        'masks': ['healthy_forest_mask.png', 'dry_grassland_mask.png', 'burned_area_mask.png'],
        'health_distribution': {
            'healthy': 0.6,
            'dry': 0.3,
            'burned': 0.1
        }
    }
    return sample_data


class VegetationHealthModel(nn.Module):
    """Vegetation Health Classification Model using CNN + VARI index."""
    
    def __init__(self, num_classes: int = 4, use_vari: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_vari = use_vari
        
        # CNN backbone for RGB image processing
        self.rgb_backbone = models.resnet50(pretrained=True)
        self.rgb_backbone.fc = nn.Identity()  # Remove final FC layer
        
        # Feature fusion layer
        rgb_features = 2048
        vari_features = 64 if use_vari else 0
        
        if use_vari:
            # VARI processing network
            self.vari_processor = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True)
            )
        
        # Final classifier
        total_features = rgb_features + vari_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, rgb_images, vari_indices=None):
        """Forward pass for vegetation health classification.
        
        Args:
            rgb_images: RGB image tensor of shape (batch_size, 3, height, width)
            vari_indices: VARI index tensor of shape (batch_size, 1) if use_vari=True
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Process RGB features
        rgb_features = self.rgb_backbone(rgb_images)
        
        if self.use_vari and vari_indices is not None:
            # Process VARI features
            vari_features = self.vari_processor(vari_indices)
            
            # Concatenate features
            combined_features = torch.cat([rgb_features, vari_features], dim=1)
        else:
            combined_features = rgb_features
        
        # Classify
        output = self.classifier(combined_features)
        return output
    
    def compute_vari(self, rgb_images):
        """Compute VARI index from RGB images.
        
        Args:
            rgb_images: RGB tensor of shape (batch_size, 3, height, width)
            
        Returns:
            VARI indices of shape (batch_size, 1)
        """
        # Extract RGB channels
        r = rgb_images[:, 0, :, :].float()
        g = rgb_images[:, 1, :, :].float()
        b = rgb_images[:, 2, :, :].float()
        
        # VARI = (Green - Red) / (Green + Red - Blue)
        numerator = g - r
        denominator = g + r - b + 1e-8  # Add small epsilon to avoid division by zero
        
        vari = numerator / denominator
        
        # Return mean VARI per image
        vari_mean = torch.mean(vari.view(vari.size(0), -1), dim=1, keepdim=True)
        return vari_mean


if __name__ == "__main__":
    # Test vegetation health assessment
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['train', 'infer'], default='infer', help='Mode to run')
    parser.add_argument('--input', help='Input image path for inference')
    parser.add_argument('--output', default='outputs/vegetation_map.png', help='Output path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['vegetation']
    
    if args.mode == 'train':
        trainer = VegetationHealthTrainer(config)
        trainer.prepare_data(config['data_dir'])
        trainer.train(epochs=config.get('epochs', 35))
    
    elif args.mode == 'infer':
        model_path = config.get('model_path', 'models/vegetation_resnet50_best.pt')
        detector = VegetationHealthInference(model_path, config)
        
        if args.input:
            image = cv2.imread(args.input)
            analysis = detector.analyze_vegetation_health(
                image,
                save_map=True,
                output_path=args.output
            )
            
            logger.info(f"Vegetation health analysis completed:")
            logger.info(f"  Overall health: {analysis['overall_health']['predicted_class']}")
            logger.info(f"  Confidence: {analysis['overall_health']['confidence']:.3f}")
            logger.info(f"  Spatial analysis: {analysis['spatial_analysis']}")
            logger.info(f"  Risk score: {analysis['risk_score']:.3f}")
        else:
            logger.info("No input provided. Please specify --input for inference.")
