#!/bin/bash
# Build script for PhysX gRPC server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Building PhysX gRPC Server"
echo "========================================"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✓ Docker found"
    echo ""
    echo "Building Docker image: guira/physx-server:dev"
    docker build -t guira/physx-server:dev .
    echo ""
    echo "✅ Build complete!"
    echo ""
    echo "To run the server:"
    echo "  docker run --rm -p 50051:50051 guira/physx-server:dev"
    echo ""
else
    echo "❌ Docker not found"
    echo ""
    echo "Please install Docker or build locally with CMake:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .."
    echo "  make -j\$(nproc)"
    exit 1
fi
