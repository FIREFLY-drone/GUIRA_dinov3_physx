#!/bin/bash
# Integration test script for PhysX gRPC server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "PhysX gRPC Server Integration Test"
echo "========================================"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Build the image
echo "Step 1: Building Docker image..."
docker build -t guira/physx-server:dev . > /tmp/physx_build.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed. Check /tmp/physx_build.log"
    exit 1
fi
echo ""

# Generate Python proto files
echo "Step 2: Generating Python proto files..."
pip install -q grpcio grpcio-tools
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/physx.proto
if [ $? -eq 0 ]; then
    echo "✅ Python proto files generated"
else
    echo "❌ Proto generation failed"
    exit 1
fi
echo ""

# Start the server in background
echo "Step 3: Starting PhysX server..."
docker run --rm -d -p 50051:50051 --name physx-server-test guira/physx-server:dev > /dev/null
sleep 3  # Give server time to start
if docker ps | grep -q physx-server-test; then
    echo "✅ Server started successfully"
else
    echo "❌ Server failed to start"
    exit 1
fi
echo ""

# Run the Python test client
echo "Step 4: Testing with Python client..."
python physx_client.py localhost 50051 > /tmp/physx_client.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Client test passed"
else
    echo "❌ Client test failed. Check /tmp/physx_client.log"
    docker stop physx-server-test > /dev/null 2>&1
    exit 1
fi
echo ""

# Verify output file exists in container
echo "Step 5: Verifying output file..."
docker exec physx-server-test test -f /tmp/physx_output/fire_perimeter_test1.geojson
if [ $? -eq 0 ]; then
    echo "✅ Output file created successfully"
else
    echo "❌ Output file not found"
    docker stop physx-server-test > /dev/null 2>&1
    exit 1
fi
echo ""

# Validate GeoJSON structure
echo "Step 6: Validating GeoJSON output..."
OUTPUT=$(docker exec physx-server-test cat /tmp/physx_output/fire_perimeter_test1.geojson)
if echo "$OUTPUT" | python -m json.tool > /dev/null 2>&1; then
    echo "✅ GeoJSON is valid"
else
    echo "❌ GeoJSON is invalid"
    docker stop physx-server-test > /dev/null 2>&1
    exit 1
fi
echo ""

# Stop the server
echo "Step 7: Cleaning up..."
docker stop physx-server-test > /dev/null 2>&1
echo "✅ Server stopped"
echo ""

echo "========================================"
echo "✅ ALL TESTS PASSED"
echo "========================================"
echo ""
echo "Summary:"
echo "  - Docker image builds successfully"
echo "  - Server starts and listens on port 50051"
echo "  - Client can connect and send requests"
echo "  - Server processes requests and returns responses"
echo "  - GeoJSON output is generated and valid"
echo ""
echo "PhysX gRPC server is ready for production integration!"
