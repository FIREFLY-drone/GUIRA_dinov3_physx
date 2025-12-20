"""
Smoke Detection using TimeSFormer for temporal video analysis.
Analyzes sequences of frames to detect smoke patterns.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from transformers import TimesformerModel, TimesformerConfig
import yaml
from loguru import logger
import math


class SmokeVideoDataset(Dataset):
    """Dataset for smoke detection in video sequences."""
    
    def __init__(self, data_dir: str, split: str = 'train', sequence_length: int = 8, 
                 img_size: int = 224, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Load video paths and annotations
        self.videos_dir = self.data_dir / 'videos' / split
        self.annotations_file = self.data_dir / f'annotations_{split}.csv'
        
        self.video_sequences = []
        self.labels = []
        
        if self.annotations_file.exists():
            df = pd.read_csv(self.annotations_file)
            
            # Group by video and create sequences
            for video_name in df['video_name'].unique():
                video_path = self.videos_dir / video_name
                if video_path.exists():
                    video_data = df[df['video_name'] == video_name].sort_values('frame_index')
                    
                    # Create overlapping sequences
                    for i in range(0, len(video_data) - sequence_length + 1, sequence_length // 2):
                        sequence_frames = video_data.iloc[i:i + sequence_length]
                        if len(sequence_frames) == sequence_length:
                            self.video_sequences.append({
                                'video_path': str(video_path),
                                'frame_indices': sequence_frames['frame_index'].tolist(),
                                'start_frame': sequence_frames['frame_index'].iloc[0],
                                'end_frame': sequence_frames['frame_index'].iloc[-1]
                            })
                            # Use majority vote for sequence label
                            smoke_votes = sequence_frames['smoke_flag'].sum()
                            self.labels.append(1 if smoke_votes > sequence_length / 2 else 0)
        
        logger.info(f"Loaded {len(self.video_sequences)} sequences for {split} split")
        
        # Setup augmentations
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(0, 25), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.video_sequences)
    
    def load_video_frames(self, video_path: str, frame_indices: List[int]) -> List[np.ndarray]:
        """Load specific frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame not available, duplicate last frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    # Create black frame as fallback
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        sequence_info = self.video_sequences[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_frames(
            sequence_info['video_path'],
            sequence_info['frame_indices']
        )
        
        # Apply transformations to each frame
        transformed_frames = []
        for frame in frames:
            transformed = self.transform(image=frame)
            transformed_frames.append(transformed['image'])
        
        # Stack frames into tensor [T, C, H, W]
        video_tensor = torch.stack(transformed_frames)
        
        return {
            'video': video_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'sequence_info': sequence_info
        }


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for TimeSFormer."""
    
    def __init__(self, channels: int, height: int, width: int, time: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.time = time
        
        # Create positional encoding
        pe = torch.zeros(time, height, width, channels)
        
        # Time encoding
        t_pos = torch.arange(0, time).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        
        # Height encoding
        h_pos = torch.arange(0, height).unsqueeze(0).unsqueeze(1).unsqueeze(1).float()
        
        # Width encoding
        w_pos = torch.arange(0, width).unsqueeze(0).unsqueeze(0).unsqueeze(1).float()
        
        # Apply sinusoidal encoding
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                           -(math.log(10000.0) / channels))
        
        # Time dimension
        pe[:, :, :, 0::2] = torch.sin(t_pos * div_term)
        pe[:, :, :, 1::2] = torch.cos(t_pos * div_term)
        
        self.register_buffer('pe', pe.permute(3, 0, 1, 2))  # [C, T, H, W]
    
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        return x + self.pe.unsqueeze(0)


class SmokeTimeSFormer(nn.Module):
    """TimeSFormer model adapted for smoke detection."""
    
    def __init__(self, num_frames: int = 8, img_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12, 
                 num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Calculate number of patches
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.total_patches = num_frames * self.num_patches_per_frame
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights if available."""
        try:
            # Try to load TimeSFormer pretrained weights
            # This would require downloading actual pretrained weights
            logger.info("Pretrained TimeSFormer weights would be loaded here")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        B, C, T, H, W = x.shape
        
        # Patch embedding: [B, C, T, H, W] -> [B, embed_dim, T, H//patch_size, W//patch_size]
        x = self.patch_embed(x)
        
        # Flatten spatial and temporal dimensions: [B, embed_dim, total_patches]
        x = x.flatten(2).transpose(1, 2)  # [B, total_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, total_patches + 1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification using CLS token
        x = self.ln(x[:, 0])  # [B, embed_dim]
        x = self.head(x)      # [B, num_classes]
        
        return x


class SmokeDetectionTrainer:
    """Trainer for smoke detection model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = SmokeTimeSFormer(
            num_frames=config.get('sequence_length', 8),
            img_size=config.get('img_size', 224),
            patch_size=config.get('patch_size', 16),
            embed_dim=config.get('embed_dim', 768),
            depth=config.get('depth', 12),
            num_heads=config.get('num_heads', 12),
            num_classes=2,  # smoke/no_smoke
            pretrained=True
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=0.05
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30)
        )
        
        logger.info(f"Initialized SmokeTimeSFormer on {self.device}")
    
    def prepare_data(self, data_dir: str):
        """Prepare training and validation datasets."""
        self.train_dataset = SmokeVideoDataset(
            data_dir=data_dir,
            split='train',
            sequence_length=self.config.get('sequence_length', 8),
            img_size=self.config.get('img_size', 224),
            augment=True
        )
        
        self.val_dataset = SmokeVideoDataset(
            data_dir=data_dir,
            split='val',
            sequence_length=self.config.get('sequence_length', 8),
            img_size=self.config.get('img_size', 224),
            augment=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 8),
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
            videos = batch['video'].to(self.device)  # [B, T, C, H, W]
            labels = batch['label'].to(self.device)
            
            # Rearrange to [B, C, T, H, W] for TimeSFormer
            videos = videos.transpose(1, 2)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                videos = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)
                
                videos = videos.transpose(1, 2)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, epochs: int = 30, save_dir: str = 'models'):
        """Train the model."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_acc = 0
        
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = Path(save_dir) / 'smoke_timesformer_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': self.config
                }, model_path)
                logger.info(f'New best model saved: {val_acc:.2f}%')
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = Path(save_dir) / f'smoke_timesformer_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config
                }, checkpoint_path)
        
        logger.info(f'Training completed. Best validation accuracy: {best_acc:.2f}%')


class SmokeDetectionInference:
    """Inference class for smoke detection."""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.sequence_length = config.get('sequence_length', 8)
        self.img_size = config.get('img_size', 224)
        
        # Initialize model
        self.model = SmokeTimeSFormer(
            num_frames=self.sequence_length,
            img_size=self.img_size,
            patch_size=config.get('patch_size', 16),
            embed_dim=config.get('embed_dim', 768),
            depth=config.get('depth', 12),
            num_heads=config.get('num_heads', 12),
            num_classes=2
        ).to(self.device)
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded smoke detection model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using random weights")
        
        self.model.eval()
        
        # Transform for preprocessing
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def predict_sequence(self, frames: List[np.ndarray]) -> float:
        """
        Predict smoke probability for a sequence of frames.
        
        Args:
            frames: List of frames as numpy arrays (RGB format)
        
        Returns:
            Smoke probability (0-1)
        """
        if len(frames) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {len(frames)}")
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            transformed = self.transform(image=frame)
            processed_frames.append(transformed['image'])
        
        # Stack into tensor [T, C, H, W]
        video_tensor = torch.stack(processed_frames).unsqueeze(0)  # [1, T, C, H, W]
        video_tensor = video_tensor.transpose(1, 2).to(self.device)  # [1, C, T, H, W]
        
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probabilities = F.softmax(outputs, dim=1)
            smoke_prob = probabilities[0, 1].item()  # Probability of smoke class
        
        return smoke_prob
    
    def predict_video(self, video_path: str, output_path: str, stride: int = 1):
        """
        Predict smoke for entire video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save CSV results
            stride: Frame stride for sampling
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        frame_buffer = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)
                
                if len(frame_buffer) == self.sequence_length:
                    # Predict for current sequence
                    smoke_prob = self.predict_sequence(frame_buffer)
                    
                    # Record result for middle frame of sequence
                    middle_frame = frame_idx - (self.sequence_length // 2) * stride
                    results.append({
                        'frame_index': middle_frame,
                        'smoke_prob': smoke_prob
                    })
                    
                    # Slide window (remove first frame)
                    frame_buffer.pop(0)
            
            frame_idx += 1
            
            if frame_idx % 1000 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        
        # Save results
        df = pd.DataFrame(results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Smoke detection completed. Results saved to {output_path}")
        return results


class SmokeDetectionModel(nn.Module):
    """Smoke Detection Model using TimeSFormer architecture."""
    
    def __init__(self, num_classes: int = 2, sequence_length: int = 8, img_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Use a simplified CNN backbone for testing
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """Forward pass for video sequences.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Reshape from (B, T, C, H, W) to (B, C, T, H, W)
        x = x.transpose(1, 2)
        
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output


def create_sample_smoke_annotations():
    """Create sample smoke annotations for testing."""
    sample_data = {
        'video_name': ['sample_video.mp4'] * 100,
        'frame_index': list(range(100)),
        'smoke_flag': [0] * 50 + [1] * 30 + [0] * 20  # Smoke appears in middle
    }
    
    df = pd.DataFrame(sample_data)
    return df


if __name__ == "__main__":
    # Test smoke detection
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['train', 'infer'], default='infer', help='Mode to run')
    parser.add_argument('--input', help='Input video path for inference')
    parser.add_argument('--output', default='outputs/smoke_predictions.csv', help='Output CSV path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['smoke']
    
    if args.mode == 'train':
        trainer = SmokeDetectionTrainer(config)
        trainer.prepare_data(config['data_dir'])
        trainer.train(epochs=config.get('epochs', 30))
    
    elif args.mode == 'infer':
        model_path = config.get('model_path', 'models/smoke_timesformer_best.pt')
        detector = SmokeDetectionInference(model_path, config)
        
        if args.input:
            detector.predict_video(args.input, args.output)
        else:
            logger.info("No input provided. Please specify --input for inference.")
