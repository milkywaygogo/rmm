#!/bin/bash
# Build script for RMM allocation examples

set -e

echo "Building RMM Allocation Examples"
echo "================================="

# Check if RMM is built (adjust path since we're in cursor_example/)
RMM_BUILD_DIR="../cpp/build"
RMM_INSTALL_DIR="$RMM_BUILD_DIR/install"

if [ ! -d "$RMM_BUILD_DIR" ]; then
    echo "Error: RMM not built yet. Please build RMM first:"
    echo "  cd .. && ./build.sh librmm"
    exit 1
fi

# Check if RMM is installed
if [ ! -d "$RMM_INSTALL_DIR" ]; then
    echo "Installing RMM..."
    cd "$RMM_BUILD_DIR"
    make install
    cd - > /dev/null
fi

# Create build directory
BUILD_DIR="build_example"
mkdir -p "$BUILD_DIR"

# Create a CMakeLists.txt in build directory for the example
cat > "$BUILD_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(rmm_allocation_example VERSION 1.0 LANGUAGES CXX CUDA)

# Find RMM
find_package(rmm REQUIRED)

# Create executable (source file is in parent directory)
add_executable(rmm_allocation_example ../rmm_allocation_example.cpp)

# Link RMM
target_link_libraries(rmm_allocation_example PRIVATE rmm::rmm)

# Set C++ standard
target_compile_features(rmm_allocation_example PRIVATE cxx_std_17)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(rmm_allocation_example PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80;86"
    CUDA_SEPARABLE_COMPILATION ON)
EOF

cd "$BUILD_DIR"

# Configure CMake
echo ""
echo "Configuring CMake..."
cmake . \
    -DCMAKE_PREFIX_PATH="../../$RMM_INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make rmm_allocation_example

echo ""
echo "Build successful!"
echo ""
echo "To run the example:"
echo "  cd $BUILD_DIR && ./rmm_allocation_example"
echo ""
echo "Or run from here:"
echo "  ./$BUILD_DIR/rmm_allocation_example"

