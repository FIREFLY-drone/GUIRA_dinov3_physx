cmake_minimum_required(VERSION 3.15)
project(physx_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Threads REQUIRED)

# Find Protobuf installation using MODULE mode (for Ubuntu packages)
find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF ${Protobuf_LIBRARIES})
find_program(_PROTOBUF_PROTOC protoc REQUIRED)

# Find gRPC installation
find_library(GRPC_LIBRARY NAMES grpc++ REQUIRED)
find_library(GRPC_REFLECTION_LIBRARY NAMES grpc++_reflection REQUIRED)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)

set(_GRPC_GRPCPP ${GRPC_LIBRARY})
set(_REFLECTION ${GRPC_REFLECTION_LIBRARY})

# Proto file
get_filename_component(physx_proto "proto/physx.proto" ABSOLUTE)
get_filename_component(physx_proto_path "${physx_proto}" PATH)

# Generated sources
set(physx_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/physx.pb.cc")
set(physx_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/physx.pb.h")
set(physx_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/physx.grpc.pb.cc")
set(physx_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/physx.grpc.pb.h")

# Custom command to generate proto sources
add_custom_command(
      OUTPUT "${physx_proto_srcs}" "${physx_proto_hdrs}" "${physx_grpc_srcs}" "${physx_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${physx_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${physx_proto}"
      DEPENDS "${physx_proto}")

# Include generated *.pb.h files and Protobuf headers
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${Protobuf_INCLUDE_DIRS})

# physx_grpc_proto library
add_library(physx_grpc_proto
  ${physx_grpc_srcs}
  ${physx_grpc_hdrs}
  ${physx_proto_srcs}
  ${physx_proto_hdrs})

target_link_libraries(physx_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Server executable
add_executable(physx_server
  "src/main.cpp")

target_link_libraries(physx_server
  physx_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Installation rules
install(TARGETS physx_server DESTINATION bin)
