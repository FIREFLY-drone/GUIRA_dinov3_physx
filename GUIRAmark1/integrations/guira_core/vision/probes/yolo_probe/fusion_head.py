# integrations/guira_core/vision/probes/yolo_probe/fusion_head.py
"""
Fusion Head for integrating DINOv3 embeddings with YOLO detections.

This module provides a simple MLP-based fusion head that can:
1. Take DINOv3 embeddings (768-dim for base model)
2. Combine with YOLO detection features
3. Output enhanced detection tags (e.g., vegetation health, fire intensity)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class FusionHead(nn.Module):
    """MLP fusion head for combining embeddings with detection features.
    
    Args:
        embed_dim: Dimension of DINOv3 embeddings (768 for base, 1024 for large)
        num_classes: Number of detection classes
        hidden_dim: Hidden layer dimension
        num_health_classes: Number of vegetation health classes (healthy, dry, burned)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_health_classes: int = 3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Embedding aggregation layer (pool patches)
        self.embed_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Health classification head
        self.health_classifier = nn.Linear(hidden_dim, num_health_classes)
        
        # Fire intensity regressor (0-1 scale)
        self.intensity_regressor = nn.Linear(hidden_dim, 1)
        
        self.eval()  # Set to eval mode by default
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through fusion head.
        
        Args:
            embeddings: DINOv3 embeddings of shape (batch, num_patches, embed_dim)
            
        Returns:
            Dictionary with health logits and fire intensity predictions
        """
        # Pool embeddings across patches
        # embeddings: (batch, num_patches, embed_dim)
        embeddings_transposed = embeddings.transpose(1, 2)  # (batch, embed_dim, num_patches)
        pooled = self.embed_pool(embeddings_transposed).squeeze(-1)  # (batch, embed_dim)
        
        # Pass through fusion MLP
        features = self.fusion_mlp(pooled)  # (batch, hidden_dim)
        
        # Health classification
        health_logits = self.health_classifier(features)  # (batch, num_health_classes)
        
        # Fire intensity regression
        intensity = torch.sigmoid(self.intensity_regressor(features))  # (batch, 1)
        
        return {
            "health_logits": health_logits,
            "health_probs": torch.softmax(health_logits, dim=-1),
            "fire_intensity": intensity
        }
    
    def predict(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Predict health and intensity from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings (num_patches, embed_dim)
            
        Returns:
            Prediction dictionary with health labels and fire intensity
        """
        health_labels = ["healthy", "dry", "burned"]
        
        # Convert to torch tensor
        if embeddings.ndim == 2:
            embeddings = embeddings[np.newaxis, ...]  # Add batch dimension
        
        embeddings_tensor = torch.from_numpy(embeddings).float()
        
        with torch.no_grad():
            outputs = self.forward(embeddings_tensor)
        
        # Get health prediction
        health_probs = outputs["health_probs"][0].cpu().numpy()
        health_idx = int(np.argmax(health_probs))
        health_label = health_labels[health_idx]
        health_conf = float(health_probs[health_idx])
        
        # Get fire intensity
        fire_intensity = float(outputs["fire_intensity"][0].cpu().numpy())
        
        return {
            "health_label": health_label,
            "health_confidence": health_conf,
            "health_probabilities": {
                label: float(prob) for label, prob in zip(health_labels, health_probs)
            },
            "fire_intensity": fire_intensity
        }
    
    def load_weights(self, path: str):
        """Load pretrained weights.
        
        Args:
            path: Path to saved weights file
        """
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict)
            print(f"Loaded fusion head weights from {path}")
        except Exception as e:
            print(f"Failed to load fusion head weights: {e}")
    
    def save_weights(self, path: str):
        """Save fusion head weights.
        
        Args:
            path: Path to save weights file
        """
        torch.save(self.state_dict(), path)
        print(f"Saved fusion head weights to {path}")


def create_default_fusion_head(embed_dim: int = 768) -> FusionHead:
    """Create a default fusion head with standard configuration.
    
    Args:
        embed_dim: Embedding dimension (768 for dinov2-base, 1024 for dinov2-large)
        
    Returns:
        Initialized FusionHead instance
    """
    return FusionHead(
        embed_dim=embed_dim,
        num_classes=2,  # fire, smoke
        hidden_dim=256,
        num_health_classes=3  # healthy, dry, burned
    )
