/*
 * Comprehensive RMM (RAPIDS Memory Manager) Example
 * This example demonstrates various memory management techniques using RMM
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// RMM headers
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>

// Simple CUDA kernel for demonstration
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel to double values in managed memory
__global__ void doubleValues(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2;
    }
}

// Kernel to initialize float array
__global__ void initializeArray(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value + idx * 0.1f;
    }
}

// Helper function to print memory usage
void printMemoryUsage(const std::string& label) {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    std::cout << label << " - Free: " << free_memory / (1024*1024) << " MB, "
              << "Used: " << (total_memory - free_memory) / (1024*1024) << " MB\n";
}

// Example 1: Basic device_buffer usage
void example1_device_buffer() {
    std::cout << "\n=== Example 1: Basic device_buffer usage ===\n";
    
    // Create a CUDA stream
    rmm::cuda_stream stream;
    
    // Allocate device memory using device_buffer
    size_t size_bytes = 1000 * sizeof(float);
    rmm::device_buffer buffer(size_bytes, stream);
    
    std::cout << "Allocated " << buffer.size() << " bytes\n";
    std::cout << "Capacity: " << buffer.capacity() << " bytes\n";
    
    // Resize the buffer
    buffer.resize(2000 * sizeof(float), stream);
    std::cout << "After resize - Size: " << buffer.size() << " bytes\n";
    
    // Shrink to fit
    buffer.shrink_to_fit(stream);
    std::cout << "After shrink_to_fit - Capacity: " << buffer.capacity() << " bytes\n";
    
    stream.synchronize();
}

// Example 2: device_uvector usage
void example2_device_uvector() {
    std::cout << "\n=== Example 2: device_uvector usage ===\n";
    
    rmm::cuda_stream stream;
    const size_t n = 1000000;
    
    // Create device vectors
    rmm::device_uvector<float> d_a(n, stream);
    rmm::device_uvector<float> d_b(n, stream);
    rmm::device_uvector<float> d_c(n, stream);
    
    // Initialize data on host
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    
    // Copy data to device
    cudaMemcpyAsync(d_a.data(), h_a.data(), n * sizeof(float), 
                    cudaMemcpyHostToDevice, stream.value());
    cudaMemcpyAsync(d_b.data(), h_b.data(), n * sizeof(float), 
                    cudaMemcpyHostToDevice, stream.value());
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize, 0, stream.value()>>>(
        d_a.data(), d_b.data(), d_c.data(), n);
    
    // Copy result back
    std::vector<float> h_c(n);
    cudaMemcpyAsync(h_c.data(), d_c.data(), n * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream.value());
    
    stream.synchronize();
    
    // Verify result
    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << "\n";
}

// Example 3: Pool memory resource
void example3_pool_resource() {
    std::cout << "\n=== Example 3: Pool Memory Resource ===\n";
    
    printMemoryUsage("Before pool creation");
    
    // Create a pool memory resource with 256 MB initial size
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
        &cuda_mr,
        256 * 1024 * 1024  // initial pool size
    );
    
    // Set as default resource
    auto* old_mr = rmm::mr::set_current_device_resource(&pool_mr);
    
    printMemoryUsage("After pool creation");
    
    // Allocate and deallocate memory using the pool
    rmm::cuda_stream stream;
    std::vector<rmm::device_buffer> buffers;
    
    // Allocate multiple buffers
    for (int i = 0; i < 10; i++) {
        buffers.emplace_back(10 * 1024 * 1024, stream); // 10 MB each
    }
    
    printMemoryUsage("After allocations");
    
    // Clear buffers (memory returns to pool, not OS)
    buffers.clear();
    
    printMemoryUsage("After deallocation (memory in pool)");
    
    // Restore original resource
    rmm::mr::set_current_device_resource(old_mr);
}

// Example 4: Logging memory resource
void example4_logging_resource() {
    std::cout << "\n=== Example 4: Logging Memory Resource ===\n";
    
    // Create base and logging resources
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::logging_resource_adaptor<rmm::mr::cuda_memory_resource> logging_mr(
        &cuda_mr, "memory_log.csv");
    
    // Use the logging resource
    rmm::cuda_stream stream;
    {
        rmm::device_buffer buffer1(1024, stream, &logging_mr);
        rmm::device_buffer buffer2(2048, stream, &logging_mr);
        buffer1.resize(4096, stream);
    } // Buffers deallocated here
    
    stream.synchronize();
    std::cout << "Memory operations logged to memory_log.csv\n";
}

// Example 5: Statistics resource adaptor
void example5_statistics_resource() {
    std::cout << "\n=== Example 5: Statistics Resource Adaptor ===\n";
    
    // Create base and statistics resources
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::statistics_resource_adaptor<rmm::mr::cuda_memory_resource> stats_mr(&cuda_mr);
    
    rmm::cuda_stream stream;
    
    // Perform allocations
    {
        rmm::device_buffer buffer1(1024 * 1024, stream, &stats_mr);      // 1 MB
        rmm::device_buffer buffer2(2 * 1024 * 1024, stream, &stats_mr);  // 2 MB
        rmm::device_buffer buffer3(512 * 1024, stream, &stats_mr);       // 512 KB
    }
    
    stream.synchronize();
    
    // Get statistics
    std::cout << "Peak allocated: " << stats_mr.get_bytes_counter().peak / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Total allocated: " << stats_mr.get_bytes_counter().total / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Current allocated: " << stats_mr.get_bytes_counter().value / (1024.0 * 1024.0) << " MB\n";
}

// Example 6: Managed memory resource
void example6_managed_memory() {
    std::cout << "\n=== Example 6: Managed Memory Resource ===\n";
    
    // Create managed memory resource
    rmm::mr::managed_memory_resource managed_mr;
    
    rmm::cuda_stream stream;
    
    // Allocate managed memory
    rmm::device_buffer managed_buffer(1000 * sizeof(int), stream, &managed_mr);
    
    // Can access from both host and device
    int* data = static_cast<int*>(managed_buffer.data());
    
    // Initialize on host
    for (int i = 0; i < 1000; i++) {
        data[i] = i;
    }
    
    // Use in kernel
    int blockSize = 256;
    int numBlocks = (1000 + blockSize - 1) / blockSize;
    doubleValues<<<numBlocks, blockSize, 0, stream.value()>>>(data, 1000);
    
    std::cout << "Managed memory allocated and accessible from both host and device\n";
    stream.synchronize();
}

// Example 7: Stream pools
void example7_stream_pools() {
    std::cout << "\n=== Example 7: Stream Pools ===\n";
    
    // Create a stream pool
    rmm::cuda_stream_pool pool(4);  // Pool with 4 streams
    
    // Get streams from the pool
    rmm::cuda_stream_view stream1 = pool.get_stream();
    rmm::cuda_stream_view stream2 = pool.get_stream();
    
    // Use streams for concurrent operations
    const size_t n = 1000000;
    rmm::device_uvector<float> vec1(n, stream1);
    rmm::device_uvector<float> vec2(n, stream2);
    
    // Concurrent kernel launches
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Initialize arrays on different streams
    initializeArray<<<numBlocks, blockSize, 0, stream1.value()>>>(vec1.data(), 1.0f, n);
    initializeArray<<<numBlocks, blockSize, 0, stream2.value()>>>(vec2.data(), 2.0f, n);
    
    std::cout << "Using " << pool.get_pool_size() 
              << " streams from pool\n";
    
    // Synchronize streams
    stream1.synchronize();
    stream2.synchronize();
}

int main() {
    try {
        // Check CUDA device
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            std::cerr << "No CUDA devices found!\n";
            return 1;
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using device: " << prop.name << "\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
        
        // Run examples
        example1_device_buffer();
        example2_device_uvector();
        example3_pool_resource();
        example4_logging_resource();
        example5_statistics_resource();
        example6_managed_memory();
        example7_stream_pools();
        
        std::cout << "\nAll examples completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}