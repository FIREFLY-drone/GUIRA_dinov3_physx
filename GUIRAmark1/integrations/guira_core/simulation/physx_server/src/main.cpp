/**
 * PhysX Fire Spread Simulation gRPC Server
 * 
 * Implements a production gRPC server skeleton that accepts SimulationRequest
 * and returns SimulationResponse. The physics internals are stubbed initially
 * (returning synthetic perimeter data), but the server compiles, runs in Docker,
 * and includes full gRPC service implementation.
 */

#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "physx.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using physx::SimulationRequest;
using physx::SimulationResponse;
using physx::PhysXSim;

/**
 * Generate a sample GeoJSON output for fire spread simulation.
 * This is a placeholder for the actual PhysX simulation logic.
 */
std::string GenerateSampleGeoJSON(const std::string& request_id) {
    std::ostringstream json;
    
    // Simple GeoJSON FeatureCollection with fire perimeter
    json << "{\n";
    json << "  \"type\": \"FeatureCollection\",\n";
    json << "  \"metadata\": {\n";
    json << "    \"request_id\": \"" << request_id << "\",\n";
    json << "    \"simulation_type\": \"physx_stub\",\n";
    json << "    \"timestamp\": \"" << std::time(nullptr) << "\",\n";
    json << "    \"description\": \"Synthetic fire perimeter output\"\n";
    json << "  },\n";
    json << "  \"features\": [\n";
    json << "    {\n";
    json << "      \"type\": \"Feature\",\n";
    json << "      \"properties\": {\n";
    json << "        \"timestep\": 0,\n";
    json << "        \"time_hours\": 0.0,\n";
    json << "        \"fire_intensity\": 0.8\n";
    json << "      },\n";
    json << "      \"geometry\": {\n";
    json << "        \"type\": \"Polygon\",\n";
    json << "        \"coordinates\": [\n";
    json << "          [\n";
    json << "            [100.0, 100.0],\n";
    json << "            [110.0, 100.0],\n";
    json << "            [110.0, 110.0],\n";
    json << "            [100.0, 110.0],\n";
    json << "            [100.0, 100.0]\n";
    json << "          ]\n";
    json << "        ]\n";
    json << "      }\n";
    json << "    },\n";
    json << "    {\n";
    json << "      \"type\": \"Feature\",\n";
    json << "      \"properties\": {\n";
    json << "        \"timestep\": 1,\n";
    json << "        \"time_hours\": 0.5,\n";
    json << "        \"fire_intensity\": 0.9\n";
    json << "      },\n";
    json << "      \"geometry\": {\n";
    json << "        \"type\": \"Polygon\",\n";
    json << "        \"coordinates\": [\n";
    json << "          [\n";
    json << "            [95.0, 95.0],\n";
    json << "            [115.0, 95.0],\n";
    json << "            [115.0, 115.0],\n";
    json << "            [95.0, 115.0],\n";
    json << "            [95.0, 95.0]\n";
    json << "          ]\n";
    json << "        ]\n";
    json << "      }\n";
    json << "    }\n";
    json << "  ]\n";
    json << "}\n";
    
    return json.str();
}

/**
 * Implementation of the PhysXSim gRPC service.
 */
class PhysXSimServiceImpl final : public PhysXSim::Service {
    Status RunSimulation(ServerContext* context, 
                        const SimulationRequest* request,
                        SimulationResponse* response) override {
        
        std::cout << "[PhysX Server] Received simulation request:" << std::endl;
        std::cout << "  Request ID: " << request->request_id() << std::endl;
        std::cout << "  Ignition count: " << request->ignitions_size() << std::endl;
        std::cout << "  Terrain mesh URI: " << request->terrain_mesh_uri() << std::endl;
        std::cout << "  dt: " << request->dt() << std::endl;
        std::cout << "  Duration (hours): " << request->duration_hours() << std::endl;
        std::cout << "  Resolution (m): " << request->resolution_m() << std::endl;
        
        // Log ignition details
        for (int i = 0; i < request->ignitions_size(); i++) {
            const auto& ignition = request->ignitions(i);
            std::cout << "  Ignition " << i << ": id=" << ignition.id() 
                     << ", confidence=" << ignition.confidence() << std::endl;
        }
        
        // TODO: Replace this stub with actual PhysX simulation
        // For now, generate synthetic GeoJSON output
        std::string geojson = GenerateSampleGeoJSON(request->request_id());
        
        // Create output directory and file path
        std::string output_dir = "/tmp/physx_output";
        std::string output_path = output_dir + "/fire_perimeter_" + 
                                 request->request_id() + ".geojson";
        
        // Create directory if it doesn't exist
        system(("mkdir -p " + output_dir).c_str());
        
        // Write GeoJSON to file
        std::ofstream outfile(output_path);
        if (outfile.is_open()) {
            outfile << geojson;
            outfile.close();
            std::cout << "[PhysX Server] Results written to: " << output_path << std::endl;
        } else {
            std::cerr << "[PhysX Server] ERROR: Could not write output file" << std::endl;
            return Status(grpc::StatusCode::INTERNAL, 
                         "Failed to write simulation results");
        }
        
        // Populate response
        response->set_request_id(request->request_id());
        response->set_status("completed");
        response->set_results_uri(output_path);
        
        std::cout << "[PhysX Server] Simulation completed successfully" << std::endl;
        
        return Status::OK;
    }
};

/**
 * Start the gRPC server and listen for requests.
 */
void RunServer() {
    std::string server_address("0.0.0.0:50051");
    PhysXSimServiceImpl service;
    
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    
    ServerBuilder builder;
    
    // Listen on the given address without authentication (insecure_credentials)
    // TODO: In production, use mTLS with secure credentials
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    
    // Register the service
    builder.RegisterService(&service);
    
    // Build and start the server
    std::unique_ptr<Server> server(builder.BuildAndStart());
    
    std::cout << "========================================" << std::endl;
    std::cout << "PhysX Fire Spread Simulation Server" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Server listening on: " << server_address << std::endl;
    std::cout << "Status: Ready to accept simulation requests" << std::endl;
    std::cout << "Security: Insecure (TODO: implement mTLS for production)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Wait for the server to shutdown
    server->Wait();
}

int main(int argc, char** argv) {
    std::cout << "Starting PhysX gRPC server..." << std::endl;
    RunServer();
    return 0;
}
