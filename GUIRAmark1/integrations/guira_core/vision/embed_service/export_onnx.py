#!/usr/bin/env python3
"""
Export DINOv2 model to ONNX format for edge deployment.

Usage:
    python export_onnx.py --model facebook/dinov2-base --output dinov2_base.onnx
"""

import argparse
import torch
from transformers import AutoModel, AutoImageProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(model_id: str, output_path: str, opset_version: int = 14):
    """Export DINOv2 model to ONNX format.
    
    Args:
        model_id: Hugging Face model ID
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    logger.info(f"Loading model: {model_id}")
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    
    # Create dummy input matching expected image size
    # DINOv2 typically uses 518x518 for optimal performance
    dummy_input = torch.randn(1, 3, 518, 518)
    
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'last_hidden_state': {0: 'batch_size'}
        },
        verbose=False
    )
    
    logger.info(f"Model exported to: {output_path}")
    
    # Verify the model
    logger.info("Verifying exported model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✓ Model verification passed")
    
    # Print model info
    logger.info(f"Input shape: [batch_size, 3, 518, 518]")
    logger.info(f"Output shape: [batch_size, num_patches, embedding_dim]")
    
    return output_path


def test_onnx_inference(onnx_path: str):
    """Test ONNX model inference.
    
    Args:
        onnx_path: Path to ONNX model
    """
    import onnxruntime as ort
    import numpy as np
    
    logger.info("Testing ONNX inference...")
    
    # Create session
    session = ort.InferenceSession(onnx_path)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 518, 518).astype(np.float32)
    
    # Run inference
    inputs = {"pixel_values": dummy_input}
    outputs = session.run(None, inputs)
    
    embeddings = outputs[0]
    logger.info(f"✓ Inference successful")
    logger.info(f"  Output shape: {embeddings.shape}")
    logger.info(f"  Output dtype: {embeddings.dtype}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Export DINOv2 to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dinov2_base.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test ONNX model after export"
    )
    
    args = parser.parse_args()
    
    # Export model
    output_path = export_to_onnx(args.model, args.output, args.opset)
    
    # Test if requested
    if args.test:
        try:
            test_onnx_inference(output_path)
        except ImportError:
            logger.warning("onnxruntime not installed, skipping inference test")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
