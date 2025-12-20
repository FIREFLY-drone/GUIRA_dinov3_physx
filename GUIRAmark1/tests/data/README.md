# Test Data Structure

This directory contains minimal sample data for testing each model:

## Fire Detection Test Data (`fire/`)
- Sample images: fire_001.jpg, fire_002.jpg, smoke_001.jpg
- Sample labels: corresponding .txt files in YOLO format
- Negative samples: background images without fire/smoke

## Smoke Detection Test Data (`smoke/`)
- Sample clips: smoke_clip_001.mp4, no_smoke_clip_001.mp4
- Temporal sequences for testing 16-frame processing
- Mixed RGB and thermal sequences (if available)

## Fauna Detection Test Data (`fauna/`)
- Wildlife images: various species with bounding box annotations
- Density test images: for CSRNet crowd counting validation
- Small object test cases: challenging detection scenarios

## Vegetation Health Test Data (`vegetation/`)
- Crown patches: healthy, stressed, and burned vegetation samples
- VARI computation test images: known RGB values for index validation
- DeepForest style canopy detection samples

## Fire Spread Test Data (`spread/`)
- Raster stacks: multi-channel geographical data
- Temporal sequences: fire progression examples
- Physics validation data: wind and topography effects

Each subdirectory should contain minimal but representative samples
for unit testing and integration validation.