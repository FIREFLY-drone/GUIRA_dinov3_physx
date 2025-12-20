#!/usr/bin/env python3
"""
PhysX gRPC Test Client

Test client for the PhysX fire spread simulation gRPC server.
Sends simulation requests and displays responses.
"""

import sys
import grpc

# Import generated proto files
try:
    import physx_pb2
    import physx_pb2_grpc
except ImportError:
    print("ERROR: Proto files not generated. Please run:")
    print("  python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/physx.proto")
    sys.exit(1)


def run_simulation_test(host="localhost", port=50051):
    """
    Send a test simulation request to the PhysX server.
    
    Args:
        host: Server hostname (default: localhost)
        port: Server port (default: 50051)
    """
    # Create a gRPC channel
    channel = grpc.insecure_channel(f"{host}:{port}")
    
    # Create a stub (client)
    stub = physx_pb2_grpc.PhysXSimStub(channel)
    
    # Create ignition points
    ignition1 = physx_pb2.Ignition(
        id="ignition_001",
        wkb="POINT(100.0 100.0)",
        confidence=0.95
    )
    
    ignition2 = physx_pb2.Ignition(
        id="ignition_002",
        wkb="POINT(120.0 120.0)",
        confidence=0.87
    )
    
    # Create simulation request
    request = physx_pb2.SimulationRequest(
        request_id="test1",
        ignitions=[ignition1, ignition2],
        terrain_mesh_uri="/tmp/tile.obj",
        dt=1.0,
        duration_hours=1.0,
        resolution_m=10
    )
    
    print("=" * 60)
    print("PhysX Client - Sending Simulation Request")
    print("=" * 60)
    print(f"Host: {host}:{port}")
    print(f"Request ID: {request.request_id}")
    print(f"Terrain mesh: {request.terrain_mesh_uri}")
    print(f"Timestep (dt): {request.dt}")
    print(f"Duration: {request.duration_hours} hours")
    print(f"Resolution: {request.resolution_m} meters")
    print(f"Number of ignitions: {len(request.ignitions)}")
    for i, ign in enumerate(request.ignitions):
        print(f"  Ignition {i+1}: id={ign.id}, confidence={ign.confidence:.2f}")
    print()
    
    try:
        # Make the RPC call
        print("Calling RunSimulation RPC...")
        response = stub.RunSimulation(request)
        
        print()
        print("=" * 60)
        print("Response Received")
        print("=" * 60)
        print(f"Request ID: {response.request_id}")
        print(f"Status: {response.status}")
        print(f"Results URI: {response.results_uri}")
        print()
        
        # Try to read and display the results file if local
        if response.results_uri.startswith("/tmp/"):
            try:
                with open(response.results_uri, 'r') as f:
                    print("Results file content (first 500 chars):")
                    print("-" * 60)
                    content = f.read()
                    print(content[:500])
                    if len(content) > 500:
                        print("...")
                    print("-" * 60)
            except FileNotFoundError:
                print(f"Note: Results file not accessible locally (may be in container)")
        
        print()
        print("✅ Test completed successfully!")
        return True
        
    except grpc.RpcError as e:
        print()
        print("❌ RPC Failed:")
        print(f"  Status: {e.code()}")
        print(f"  Details: {e.details()}")
        return False
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        return False
    finally:
        channel.close()


def main():
    """Main entry point for the test client."""
    # Parse command line arguments
    host = "localhost"
    port = 50051
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print()
    print("PhysX gRPC Test Client")
    print()
    
    # Run the test
    success = run_simulation_test(host, port)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
