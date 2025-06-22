# CMakeLists.txt for RMM Example
# This shows how to build a C++/CUDA application using RMM

cmake_minimum_required(VERSION 3.30)

# Initialize CUDA architectures before project declaration
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")

project(rmm_example LANGUAGES CXX CUDA)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Set the path to the RMM build directory (one level up from example_usage)
set(rmm_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../build")

# Find RMM package
find_package(rmm REQUIRED)

# Create executable
add_executable(rmm_example rmm_example.cu)

# Set C++ standard
target_compile_features(rmm_example PRIVATE cxx_std_17)

# Link RMM
target_link_libraries(rmm_example PRIVATE rmm::rmm)

# Enable CUDA separable compilation if needed
set_target_properties(rmm_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)