# Multi-stage build for PhysX gRPC server
# Stage 1: Build the C++ server with gRPC

FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    cmake \
    git \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install gRPC and Protobuf
# Using the latest stable version from Ubuntu repos for simplicity
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy proto and source files
COPY proto/ ./proto/
COPY src/ ./src/
COPY CMakeLists.txt .

# Create build directory and build the server
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# Stage 2: Runtime image
FROM ubuntu:22.04

WORKDIR /app

# Install runtime dependencies including gRPC and Protobuf libraries
RUN apt-get update && \
    apt-get install -y \
    libssl3 \
    ca-certificates \
    libgrpc++1 \
    libprotobuf23 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built server from builder stage
COPY --from=builder /app/build/physx_server /app/physx_server

# Create output directory for simulation results
RUN mkdir -p /tmp/physx_output

# Expose gRPC port
EXPOSE 50051

# Run the server
CMD ["/app/physx_server"]
