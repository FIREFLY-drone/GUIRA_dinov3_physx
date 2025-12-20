#!/usr/bin/env python3
"""
Example Orchestrator Integration for PhysX gRPC Server

This demonstrates how the orchestrator should integrate with the PhysX
fire spread simulation server. This is a reference implementation showing
proper request construction, error handling, and result processing.
"""

import grpc
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

# Import generated proto files
# Run: python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/physx.proto
import physx_pb2
import physx_pb2_grpc


@dataclass
class IgnitionPoint:
    """Detected fire ignition point from vision pipeline."""
    id: str
    wkb_geometry: str
    confidence: float
    detection_timestamp: str


class PhysXOrchestrator:
    """
    Orchestrator client for PhysX fire spread simulation server.
    
    This class manages the lifecycle of simulation requests:
    1. Collect ignition points from detection pipeline
    2. Fetch terrain mesh and metadata
    3. Send simulation request to PhysX server
    4. Process and store results
    5. Trigger downstream actions (alerts, visualization)
    """
    
    def __init__(self, physx_server_host: str = "localhost", 
                 physx_server_port: int = 50051,
                 use_secure_channel: bool = False):
        """
        Initialize orchestrator with PhysX server connection.
        
        Args:
            physx_server_host: Hostname of PhysX server
            physx_server_port: Port of PhysX server
            use_secure_channel: Whether to use mTLS (production only)
        """
        self.server_address = f"{physx_server_host}:{physx_server_port}"
        self.use_secure_channel = use_secure_channel
        self._channel = None
        self._stub = None
        
    def connect(self):
        """Establish connection to PhysX server."""
        if self.use_secure_channel:
            # TODO: Load certificates for mTLS
            # credentials = grpc.ssl_channel_credentials(
            #     root_certificates=...,
            #     private_key=...,
            #     certificate_chain=...
            # )
            # self._channel = grpc.secure_channel(self.server_address, credentials)
            raise NotImplementedError("Secure channel not yet implemented")
        else:
            self._channel = grpc.insecure_channel(self.server_address)
        
        self._stub = physx_pb2_grpc.PhysXSimStub(self._channel)
        print(f"✓ Connected to PhysX server at {self.server_address}")
    
    def disconnect(self):
        """Close connection to PhysX server."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            print("✓ Disconnected from PhysX server")
    
    def request_simulation(self,
                          request_id: str,
                          ignition_points: List[IgnitionPoint],
                          terrain_mesh_uri: str,
                          timestep_seconds: float = 1.0,
                          duration_hours: float = 1.0,
                          resolution_meters: int = 10) -> Optional[Dict]:
        """
        Request fire spread simulation from PhysX server.
        
        Args:
            request_id: Unique identifier for this simulation
            ignition_points: List of detected fire ignition points
            terrain_mesh_uri: URI to terrain mesh file (local path or blob URL)
            timestep_seconds: Simulation timestep in seconds
            duration_hours: Simulation duration in hours
            resolution_meters: Grid resolution in meters
            
        Returns:
            Dictionary with simulation results or None on error
        """
        if not self._stub:
            print("❌ Not connected to PhysX server. Call connect() first.")
            return None
        
        # Construct simulation request
        request = physx_pb2.SimulationRequest(
            request_id=request_id,
            terrain_mesh_uri=terrain_mesh_uri,
            dt=timestep_seconds,
            duration_hours=duration_hours,
            resolution_m=resolution_meters
        )
        
        # Add ignition points
        for point in ignition_points:
            ignition = physx_pb2.Ignition(
                id=point.id,
                wkb=point.wkb_geometry,
                confidence=point.confidence
            )
            request.ignitions.append(ignition)
        
        print(f"\n📤 Sending simulation request: {request_id}")
        print(f"   Ignitions: {len(ignition_points)}")
        print(f"   Terrain mesh: {terrain_mesh_uri}")
        print(f"   Duration: {duration_hours} hours")
        print(f"   Resolution: {resolution_meters}m")
        
        try:
            # Call PhysX server
            response = self._stub.RunSimulation(request)
            
            print(f"\n📥 Received response:")
            print(f"   Status: {response.status}")
            print(f"   Results URI: {response.results_uri}")
            
            # Process results
            result = {
                "request_id": response.request_id,
                "status": response.status,
                "results_uri": response.results_uri,
                "timestamp": time.time()
            }
            
            # If successful, fetch and process GeoJSON
            if response.status == "completed":
                self._process_simulation_results(result)
            
            return result
            
        except grpc.RpcError as e:
            print(f"\n❌ RPC Error:")
            print(f"   Code: {e.code()}")
            print(f"   Details: {e.details()}")
            return None
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return None
    
    def _process_simulation_results(self, result: Dict):
        """
        Process simulation results (GeoJSON fire perimeters).
        
        In production, this would:
        1. Download GeoJSON from blob storage if results_uri is a URL
        2. Parse and validate GeoJSON structure
        3. Extract fire perimeter evolution
        4. Store in database for visualization
        5. Trigger alerts if fire spreads to critical areas
        6. Update risk maps
        """
        print(f"\n🔍 Processing simulation results...")
        
        # For now, just log that we received results
        # TODO: Implement full result processing pipeline
        print(f"   ✓ Results available at: {result['results_uri']}")
        print(f"   ✓ Ready for downstream processing")
    
    def __enter__(self):
        """Context manager support."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.disconnect()


def main():
    """Example usage of PhysX orchestrator integration."""
    print("=" * 60)
    print("PhysX Orchestrator Integration Example")
    print("=" * 60)
    
    # Example: Ignition points from detection pipeline
    ignition_points = [
        IgnitionPoint(
            id="det_001",
            wkb_geometry="POINT(100.5 200.3)",
            confidence=0.95,
            detection_timestamp="2024-10-04T15:30:00Z"
        ),
        IgnitionPoint(
            id="det_002",
            wkb_geometry="POINT(120.8 210.5)",
            confidence=0.87,
            detection_timestamp="2024-10-04T15:31:00Z"
        )
    ]
    
    # Example: Terrain mesh from mesh preparation service
    terrain_mesh_uri = "/data/meshes/region_north_forest_2024.obj"
    
    # Use context manager for automatic connection handling
    with PhysXOrchestrator(physx_server_host="localhost", 
                          physx_server_port=50051) as orchestrator:
        
        # Request simulation
        result = orchestrator.request_simulation(
            request_id=f"sim_{int(time.time())}",
            ignition_points=ignition_points,
            terrain_mesh_uri=terrain_mesh_uri,
            timestep_seconds=1.0,
            duration_hours=2.0,
            resolution_meters=10
        )
        
        if result and result["status"] == "completed":
            print("\n" + "=" * 60)
            print("✅ Simulation completed successfully!")
            print("=" * 60)
            print(f"\nNext steps:")
            print(f"  1. Fetch results from: {result['results_uri']}")
            print(f"  2. Parse GeoJSON fire perimeters")
            print(f"  3. Update risk maps and dashboards")
            print(f"  4. Trigger alerts if needed")
        else:
            print("\n❌ Simulation failed or returned error")


if __name__ == "__main__":
    main()
