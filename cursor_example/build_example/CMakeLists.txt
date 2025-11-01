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
