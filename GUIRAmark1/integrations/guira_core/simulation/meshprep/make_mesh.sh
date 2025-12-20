#!/bin/bash
# make_mesh.sh - CLI wrapper for DEM to mesh conversion
#
# Usage:
#   ./make_mesh.sh <dem_file> [output_file] [options]
#
# Examples:
#   ./make_mesh.sh terrain.tif
#   ./make_mesh.sh terrain.tif mesh.obj
#   ./make_mesh.sh terrain.tif mesh.obj --scale 2.0
#   ./make_mesh.sh terrain.tif mesh.obj --veg vegetation.tif

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
DEM_FILE=""
OUTPUT_FILE="tile.obj"
EXTRA_ARGS=""

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dem_file> [output_file] [options]"
    echo ""
    echo "Examples:"
    echo "  $0 terrain.tif"
    echo "  $0 terrain.tif mesh.obj"
    echo "  $0 terrain.tif mesh.obj --scale 2.0"
    echo "  $0 terrain.tif mesh.obj --veg vegetation.tif"
    exit 1
fi

DEM_FILE="$1"
shift

# Check if second argument is a file path (doesn't start with -)
if [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; then
    OUTPUT_FILE="$1"
    shift
fi

# Remaining arguments are passed to Python script
EXTRA_ARGS="$@"

# Check if DEM file exists
if [ ! -f "$DEM_FILE" ]; then
    echo "Error: DEM file not found: $DEM_FILE"
    exit 1
fi

# Run Python script
echo "Converting DEM to mesh..."
echo "  Input:  $DEM_FILE"
echo "  Output: $OUTPUT_FILE"

python3 "$SCRIPT_DIR/convert_dem_to_mesh.py" \
    --dem "$DEM_FILE" \
    --out "$OUTPUT_FILE" \
    $EXTRA_ARGS

echo "Done! Mesh saved to $OUTPUT_FILE"
