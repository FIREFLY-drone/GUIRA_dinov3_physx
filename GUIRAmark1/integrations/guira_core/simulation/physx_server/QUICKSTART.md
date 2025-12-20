# PhysX gRPC Server - Quick Start Guide

Get up and running with the PhysX fire spread simulation server in under 5 minutes.

## Prerequisites

- Docker installed and running
- Python 3.8+ (for testing)
- Port 50051 available

## 1. Build the Server (30 seconds)

```bash
cd integrations/guira_core/simulation/physx_server
docker build -t guira/physx-server:dev .
```

Or use the build script:

```bash
./build.sh
```

## 2. Start the Server (5 seconds)

```bash
docker run --rm -p 50051:50051 guira/physx-server:dev
```

You should see:

```
========================================
PhysX Fire Spread Simulation Server
========================================
Server listening on: 0.0.0.0:50051
Status: Ready to accept simulation requests
Security: Insecure (TODO: implement mTLS for production)
========================================
```

## 3. Test the Server (10 seconds)

In a new terminal:

```bash
# Install Python dependencies
pip install grpcio grpcio-tools

# Generate Python proto files
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/physx.proto

# Run test client
python physx_client.py
```

Expected output:

```
============================================================
Response Received
============================================================
Request ID: test1
Status: completed
Results URI: /tmp/physx_output/fire_perimeter_test1.geojson

✅ Test completed successfully!
```

## 4. Run Integration Tests

Automated end-to-end test:

```bash
./test.sh
```

This will:
- Build the Docker image
- Generate proto files
- Start the server
- Run the test client
- Verify GeoJSON output
- Clean up

## Quick Commands

```bash
# Build
docker build -t guira/physx-server:dev .

# Run
docker run --rm -p 50051:50051 guira/physx-server:dev

# Run in background
docker run -d --name physx-server -p 50051:50051 guira/physx-server:dev

# View logs
docker logs -f physx-server

# Stop
docker stop physx-server

# Test
python physx_client.py localhost 50051

# Full integration test
./test.sh
```

## Example Usage (Python)

```python
import grpc
import physx_pb2
import physx_pb2_grpc

# Connect to server
channel = grpc.insecure_channel("localhost:50051")
stub = physx_pb2_grpc.PhysXSimStub(channel)

# Create simulation request
request = physx_pb2.SimulationRequest(
    request_id="my_simulation",
    terrain_mesh_uri="/path/to/mesh.obj",
    dt=1.0,
    duration_hours=2.0,
    resolution_m=10
)

# Add ignition point
ignition = physx_pb2.Ignition(
    id="ignition_001",
    wkb="POINT(100.0 100.0)",
    confidence=0.95
)
request.ignitions.append(ignition)

# Call server
response = stub.RunSimulation(request)

print(f"Status: {response.status}")
print(f"Results: {response.results_uri}")
```

## Troubleshooting

### "Port 50051 already in use"

```bash
# Find and kill the process
lsof -ti:50051 | xargs kill -9

# Or use a different port
docker run --rm -p 50052:50051 guira/physx-server:dev
python physx_client.py localhost 50052
```

### "Cannot connect to server"

1. Verify server is running: `docker ps | grep physx`
2. Check server logs: `docker logs <container_id>`
3. Test connectivity: `nc -zv localhost 50051`

### "Proto import failed"

```bash
# Regenerate proto files
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/physx.proto
```

### Docker build fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
docker build --no-cache -t guira/physx-server:dev .
```

## What's Next?

- See [README.md](README.md) for full documentation
- Integrate with your orchestrator
- Replace stubbed physics with PhysX SDK
- Add mTLS for production
- Deploy to Kubernetes

## File Structure

```
physx_server/
├── proto/physx.proto       # gRPC service definition
├── src/main.cpp           # C++ server implementation
├── CMakeLists.txt         # Build configuration
├── Dockerfile             # Container definition
├── physx_client.py        # Test client
├── test.sh                # Integration tests
├── build.sh               # Build helper
├── README.md              # Full documentation
└── QUICKSTART.md          # This file
```

## Support

For issues or questions:
1. Check [README.md](README.md) troubleshooting section
2. Review server logs: `docker logs <container_id>`
3. Validate proto files are generated correctly
4. File an issue in the repository

---

**Last Updated:** 2024-10-04  
**Version:** 1.0.0 (Production skeleton with stubbed physics)
