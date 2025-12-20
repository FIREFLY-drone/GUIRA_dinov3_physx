# PhysX gRPC Server (PH-07)

Production gRPC server skeleton for fire spread simulation using PhysX. This server accepts `SimulationRequest` messages and returns `SimulationResponse` with fire perimeter results.

## Overview

This is a production-ready gRPC server skeleton that:
- Accepts simulation requests via gRPC
- Loads terrain mesh and fuel metadata (stubbed)
- Exposes `RunSimulation` RPC to orchestrator
- Returns synthetic fire perimeter results (physics internals stubbed)
- Compiles and runs in Docker
- Includes Python test client

## Architecture

```
physx_server/
├── proto/
│   └── physx.proto          # gRPC service definition
├── src/
│   └── main.cpp             # C++ server implementation
├── CMakeLists.txt           # Build configuration
├── Dockerfile               # Multi-stage Docker build
├── physx_client.py          # Python test client
└── README.md                # This file
```

## Protocol Definition

The server implements the following gRPC service:

```protobuf
service PhysXSim {
  rpc RunSimulation(SimulationRequest) returns (SimulationResponse);
}
```

### Messages

- **SimulationRequest**: Contains ignition points, terrain mesh URI, timestep, duration, and resolution
- **SimulationResponse**: Returns request ID, status, and results URI (path to GeoJSON output)
- **Ignition**: Fire ignition point with ID, WKB geometry, and confidence
- **Vec3**: 3D vector for wind and other vectors
- **Weather**: Weather parameters (timestamp, wind, humidity)

## Building

### Prerequisites

- Docker (recommended)
- OR: CMake 3.15+, gRPC, Protobuf (for local build)

### Docker Build

```bash
cd integrations/guira_core/simulation/physx_server
docker build -t guira/physx-server:dev .
```

### Local Build (without Docker)

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install -y build-essential cmake libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

# Build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Running

### Run with Docker

```bash
docker run --rm -p 50051:50051 guira/physx-server:dev
```

### Run Locally

```bash
./build/physx_server
```

The server will start on port 50051 and display:

```
========================================
PhysX Fire Spread Simulation Server
========================================
Server listening on: 0.0.0.0:50051
Status: Ready to accept simulation requests
Security: Insecure (TODO: implement mTLS for production)
========================================
```

## Testing

### Python Test Client

1. **Generate Python proto files:**

```bash
pip install grpcio grpcio-tools

python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/physx.proto
```

2. **Run the test client:**

```bash
python physx_client.py
```

Or specify custom host/port:

```bash
python physx_client.py localhost 50051
```

### Expected Output

The test client will:
1. Connect to the server
2. Send a simulation request with 2 ignition points
3. Display the response
4. Show the generated GeoJSON content (if accessible)

Example response:

```
PhysX Client - Sending Simulation Request
============================================================
Host: localhost:50051
Request ID: test1
Terrain mesh: /tmp/tile.obj
Timestep (dt): 1.0
Duration: 1.0 hours
Resolution: 10 meters
Number of ignitions: 2
  Ignition 1: id=ignition_001, confidence=0.95
  Ignition 2: id=ignition_002, confidence=0.87

Calling RunSimulation RPC...

============================================================
Response Received
============================================================
Request ID: test1
Status: completed
Results URI: /tmp/physx_output/fire_perimeter_test1.geojson

✅ Test completed successfully!
```

## Output Format

The server generates GeoJSON FeatureCollections with fire perimeter evolution:

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "request_id": "test1",
    "simulation_type": "physx_stub",
    "timestamp": "1696445123",
    "description": "Synthetic fire perimeter output"
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "timestep": 0,
        "time_hours": 0.0,
        "fire_intensity": 0.8
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100.0, 100.0], [110.0, 100.0], ...]]
      }
    }
  ]
}
```

## Security

### Current Implementation

- Uses `InsecureServerCredentials` for development
- No authentication or encryption
- Suitable for testing and development only

### Production Requirements

For production deployment, implement:

1. **mTLS Authentication:**
   ```cpp
   auto creds = grpc::SslServerCredentials(ssl_opts);
   builder.AddListeningPort(server_address, creds);
   ```

2. **Network Policies:**
   - Use Kubernetes NetworkPolicies to restrict access
   - Allow connections only from orchestrator pods
   - Block external traffic

3. **Service Mesh (Optional):**
   - Use Istio or Linkerd for automatic mTLS
   - Enable mutual authentication between services
   - Add observability and tracing

## Integration with Orchestrator

The orchestrator should:

1. Create gRPC channel to physx-server:50051
2. Send `SimulationRequest` with:
   - Unique request_id
   - Ignition polygons from detection pipeline
   - Terrain mesh URI (from blob storage or local path)
   - Simulation parameters (dt, duration, resolution)
3. Receive `SimulationResponse` with:
   - Request ID (for tracking)
   - Status ("completed", "failed", "pending")
   - Results URI (path to GeoJSON output)
4. Retrieve results from the specified URI
5. Process and visualize fire perimeters

## Current Limitations

This is a **skeleton implementation** with stubs:

1. **Physics:** Returns synthetic perimeter data, no real PhysX simulation
2. **Mesh Loading:** Does not actually load terrain meshes
3. **Fuel Metadata:** Does not process vegetation/fuel data
4. **Weather:** Does not incorporate weather parameters

## Future Development

To make this production-ready:

1. **Integrate PhysX SDK:**
   - Add NVIDIA PhysX 5.x libraries
   - Implement particle-based fire spread
   - Load and process terrain meshes

2. **Fuel Model:**
   - Parse vegetation metadata
   - Calculate burn rates and spread factors
   - Implement fuel consumption

3. **Weather Integration:**
   - Use Weather message for wind/humidity
   - Implement CFD wind field simulation
   - Add weather update streaming

4. **Performance:**
   - Add GPU acceleration
   - Implement multi-threaded processing
   - Enable batch simulation requests

5. **Storage:**
   - Upload results to blob storage (Azure/S3)
   - Return signed URLs instead of local paths
   - Add result caching

## Troubleshooting

### "Failed to connect to server"
- Ensure server is running: `docker ps | grep physx-server`
- Check port mapping: server should expose 50051
- Verify firewall rules allow connections

### "Proto import failed"
- Generate Python proto files: see Testing section above
- Ensure `grpcio` and `grpcio-tools` are installed

### "Cannot write output file"
- Check write permissions for `/tmp/physx_output`
- Verify disk space availability

### Docker build fails
- Ensure Docker daemon is running
- Check internet connectivity for base image download
- Verify sufficient disk space

## References

- [gRPC Documentation](https://grpc.io/docs/)
- [Protocol Buffers](https://protobuf.dev/)
- [NVIDIA PhysX](https://developer.nvidia.com/physx-sdk)
- [Fire Spread Models](https://www.fs.usda.gov/pnw/tools/fire-spread-models)

## Contact

For issues or questions, refer to the main GUIRA repository documentation.
