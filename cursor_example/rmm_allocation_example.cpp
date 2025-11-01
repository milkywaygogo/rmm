/*
 * RMM Allocation Strategy Example
 * 
 * This example demonstrates:
 * 1. How RMM handles small vs large allocations
 * 2. Different memory resource types
 * 3. When cudaFree is actually called
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <chrono>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <cuda_runtime_api.h>

// Helper to format bytes
std::string format_bytes(std::size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 3) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

// Example 1: Direct CUDA allocations (immediate cudaFree)
void example_direct_cuda_allocations() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Example 1: Direct CUDA Allocations (cuda_memory_resource)\n";
    std::cout << "Note: Each deallocate() immediately calls cudaFree()\n";
    std::cout << std::string(60, '=') << "\n";
    
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::cuda_stream stream;
    
    // Small allocations
    std::vector<rmm::device_uvector<char>> small_allocs;
    std::cout << "\nAllocating small buffers (1KB, 4KB, 16KB):\n";
    for (auto size : {1024, 4096, 16384}) {
        small_allocs.emplace_back(size, stream, cuda_mr);
        std::cout << "  Allocated " << format_bytes(size) 
                  << " at pointer: " << static_cast<void*>(small_allocs.back().data()) << "\n";
    }
    
    // Large allocations
    std::vector<rmm::device_uvector<char>> large_allocs;
    std::cout << "\nAllocating large buffers (1MB, 10MB, 100MB):\n";
    for (auto size : {1024*1024, 10*1024*1024, 100*1024*1024}) {
        large_allocs.emplace_back(size, stream, cuda_mr);
        std::cout << "  Allocated " << format_bytes(size) 
                  << " at pointer: " << static_cast<void*>(large_allocs.back().data()) << "\n";
    }
    
    stream.synchronize();
    std::cout << "\nWhen these buffers go out of scope, cudaFree() is called immediately.\n";
}

// Example 2: Pool allocations (delayed cudaFree)
void example_pool_allocations() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Example 2: Pool Memory Resource\n";
    std::cout << "Note: Memory goes to pool free list, NO immediate cudaFree()\n";
    std::cout << std::string(60, '=') << "\n";
    
    rmm::mr::cuda_memory_resource upstream_mr;
    
    // Create a pool with initial size of 256MB, max 1GB
    constexpr std::size_t initial_pool = 256 * 1024 * 1024;
    constexpr std::size_t max_pool = 1024 * 1024 * 1024;
    
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> 
        pool_mr(&upstream_mr, initial_pool, max_pool);
    
    std::cout << "\nPool created:\n";
    std::cout << "  Initial pool size: " << format_bytes(initial_pool) << "\n";
    std::cout << "  Maximum pool size: " << format_bytes(max_pool) << "\n";
    std::cout << "  Current pool size: " << format_bytes(pool_mr.pool_size()) << "\n";
    
    rmm::cuda_stream stream;
    
    // Allocate many small buffers
    std::cout << "\nAllocating 10 buffers of varying sizes from pool:\n";
    std::vector<rmm::device_uvector<char>> buffers;
    for (int i = 0; i < 10; ++i) {
        auto size = (i + 1) * 1024 * 1024;  // 1MB to 10MB
        buffers.emplace_back(size, stream, pool_mr);
        std::cout << "  Buffer " << i+1 << ": " << format_bytes(size) 
                  << " (pool size: " << format_bytes(pool_mr.pool_size()) << ")\n";
    }
    
    stream.synchronize();
    std::cout << "\nCurrent pool size after allocations: " 
              << format_bytes(pool_mr.pool_size()) << "\n";
    
    std::cout << "\nFreeing all buffers...\n";
    buffers.clear();  // Returns memory to pool free list
    
    std::cout << "Pool size after deallocation: " 
              << format_bytes(pool_mr.pool_size()) << "\n";
    std::cout << "Note: cudaFree() NOT called yet - memory remains in pool!\n";
    std::cout << "cudaFree() will only be called when pool_mr is destroyed.\n";
}

// Example 3: Binning memory resource (optimized for different sizes)
void example_binning_allocations() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Example 3: Binning Memory Resource\n";
    std::cout << "Routes allocations to different resources based on size\n";
    std::cout << std::string(60, '=') << "\n";
    
    rmm::mr::cuda_memory_resource upstream_mr;
    
    // Create binning resource with fixed-size bins for small allocations
    // Bins: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB
    rmm::mr::binning_memory_resource<rmm::mr::cuda_memory_resource> 
        binning_mr(&upstream_mr, 10, 22);  // 2^10 = 1KB to 2^22 = 4MB
    
    // Add a pool for allocations larger than 4MB
    auto pool_for_large = std::make_unique<
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            &upstream_mr, 100 * 1024 * 1024, 500 * 1024 * 1024);
    binning_mr.add_bin(4 * 1024 * 1024 + 1, pool_for_large.get());
    
    rmm::cuda_stream stream;
    
    std::cout << "\nAllocation size routing:\n";
    std::cout << "  Small (< 1KB):      Fixed-size bin\n";
    std::cout << "  Medium (1KB-4MB):   Fixed-size bins\n";
    std::cout << "  Large (> 4MB):      Pool memory resource\n";
    
    // Small allocations - use fixed-size bins
    std::cout << "\nSmall allocations (use fixed-size bins):\n";
    std::vector<rmm::device_uvector<char>> small_buffers;
    for (auto size : {512, 2048, 8192, 32768, 131072}) {
        small_buffers.emplace_back(size, stream, binning_mr);
        std::cout << "  " << format_bytes(size) << " -> Fixed-size bin\n";
    }
    
    // Large allocations - use pool
    std::cout << "\nLarge allocations (use pool resource):\n";
    std::vector<rmm::device_uvector<char>> large_buffers;
    for (auto size : {5*1024*1024, 20*1024*1024, 50*1024*1024}) {
        large_buffers.emplace_back(size, stream, binning_mr);
        std::cout << "  " << format_bytes(size) << " -> Pool resource\n";
    }
    
    stream.synchronize();
    
    std::cout << "\nWhen buffers are freed:\n";
    std::cout << "  Small buffers: Returned to fixed-size bin free lists\n";
    std::cout << "  Large buffers: Returned to pool free list\n";
    std::cout << "  cudaFree() only called when resources are destroyed\n";
}

// Example 4: Fixed-size memory resource (very fast for same-sized allocations)
void example_fixed_size_allocations() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Example 4: Fixed-Size Memory Resource\n";
    std::cout << "O(1) allocation for fixed-size blocks\n";
    std::cout << std::string(60, '=') << "\n";
    
    rmm::mr::cuda_memory_resource upstream_mr;
    
    // Create fixed-size resource for 64KB blocks
    constexpr std::size_t block_size = 64 * 1024;
    rmm::mr::fixed_size_memory_resource<rmm::mr::cuda_memory_resource>
        fixed_mr(&upstream_mr, block_size, 128);  // 128 blocks pre-allocated
    
    std::cout << "\nFixed-size resource:\n";
    std::cout << "  Block size: " << format_bytes(block_size) << "\n";
    std::cout << "  Blocks pre-allocated: 128\n";
    std::cout << "  Total pre-allocated: " << format_bytes(block_size * 128) << "\n";
    
    rmm::cuda_stream stream;
    
    // Allocate many blocks of the same size
    std::cout << "\nAllocating 20 blocks (should be very fast - O(1)):\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<rmm::device_uvector<char>> blocks;
    for (int i = 0; i < 20; ++i) {
        blocks.emplace_back(block_size, stream, fixed_mr);
    }
    
    stream.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Allocated 20 blocks in " << duration.count() << " microseconds\n";
    std::cout << "  Average: " << (duration.count() / 20.0) << " microseconds per allocation\n";
    
    std::cout << "\nFreeing all blocks (also O(1)):\n";
    start = std::chrono::high_resolution_clock::now();
    blocks.clear();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Freed 20 blocks in " << duration.count() << " microseconds\n";
    std::cout << "  Memory returned to fixed-size free list\n";
    std::cout << "  cudaFree() only called when fixed_mr is destroyed\n";
}

// Example 5: CUDA async memory resource (uses CUDA's built-in pool)
void example_cuda_async_allocations() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Example 5: CUDA Async Memory Resource\n";
    std::cout << "Uses CUDA driver's built-in memory pool\n";
    std::cout << std::string(60, '=') << "\n";
    
    // Create async resource using 50% of free GPU memory
    rmm::mr::cuda_async_memory_resource async_mr{
        rmm::percent_of_free_device_memory(50)};
    
    rmm::cuda_stream stream;
    
    std::cout << "\nCUDA async memory resource uses CUDA driver's memory pool.\n";
    std::cout << "Allocations are stream-ordered and very efficient.\n\n";
    
    // Allocate buffers of different sizes
    std::vector<rmm::device_uvector<float>> buffers;
    std::cout << "Allocating buffers:\n";
    for (auto size_mb : {1, 5, 10, 25}) {
        auto size = size_mb * 1024 * 1024 / sizeof(float);
        buffers.emplace_back(size, stream, async_mr);
        std::cout << "  " << size_mb << " MB buffer (" << size << " floats)\n";
    }
    
    stream.synchronize();
    
    std::cout << "\nWhen freed, memory returns to CUDA driver's pool.\n";
    std::cout << "No cudaFree() called - CUDA driver manages the pool.\n";
    buffers.clear();
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     RMM Allocation Strategy Examples                     ║\n";
    std::cout << "║     Understanding Large vs Small Allocations             ║\n";
    std::cout << "║     When cudaFree() is Actually Called                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    try {
        // Show GPU info
        auto device_id = rmm::get_current_cuda_device();
        auto [free_mem, total_mem] = rmm::available_device_memory();
        
        // Get device name
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id.value());
        
        std::cout << "\nGPU Information:\n";
        std::cout << "  Device ID: " << device_id.value() << "\n";
        std::cout << "  Device Name: " << prop.name << "\n";
        std::cout << "  Total Memory: " << format_bytes(total_mem) << "\n";
        std::cout << "  Free Memory: " << format_bytes(free_mem) << "\n";
        
        // Run examples
        example_direct_cuda_allocations();
        example_pool_allocations();
        example_binning_allocations();
        example_fixed_size_allocations();
        example_cuda_async_allocations();
        
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Summary:\n";
        std::cout << "  1. cuda_memory_resource: Immediate cudaFree()\n";
        std::cout << "  2. pool_memory_resource: Delayed cudaFree() (on destruction)\n";
        std::cout << "  3. binning_memory_resource: Routes to appropriate resource\n";
        std::cout << "  4. fixed_size_memory_resource: O(1) for fixed sizes\n";
        std::cout << "  5. cuda_async_memory_resource: Uses CUDA driver pool\n";
        std::cout << std::string(60, '=') << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

