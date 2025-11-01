/*
 * RMM Allocation Strategy Example
 * 
 * This example demonstrates:
 * 1. How RMM handles small vs large allocations
 * 2. Different memory resource types
 * 3. When cudaFree is actually called
 * 4. Performance benchmarking of alloc/dealloc patterns
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
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

// ============================================================================
// BENCHMARKING INFRASTRUCTURE
// ============================================================================

struct BenchmarkResult {
    std::string resource_name;
    std::string pattern_name;
    double alloc_time_us;      // Total allocation time in microseconds
    double dealloc_time_us;    // Total deallocation time in microseconds
    double avg_alloc_time_us;  // Average per-allocation time
    double avg_dealloc_time_us; // Average per-deallocation time
    std::size_t num_operations;
    std::size_t total_bytes;
};

// Benchmark pattern types
enum class PatternType {
    SMALL_ALLOCS,      // Many small allocations (1KB-64KB)
    LARGE_ALLOCS,      // Fewer large allocations (1MB-100MB)
    MIXED,             // Mixed small and large
    SEQUENTIAL,        // Alloc then dealloc immediately
    BATCH,             // Alloc many, then dealloc all
    RANDOM_SIZES       // Random allocation sizes
};

// Generic benchmark template
template<typename MemoryResource>
BenchmarkResult benchmark_pattern(MemoryResource& mr, 
                                   PatternType pattern,
                                   int iterations,
                                   const std::string& resource_name,
                                   const std::string& pattern_name) {
    BenchmarkResult result;
    result.resource_name = resource_name;
    result.pattern_name = pattern_name;
    result.num_operations = 0;
    result.total_bytes = 0;
    
    rmm::cuda_stream stream;
    std::mt19937 rng(42); // Fixed seed for reproducibility
    
    // Pattern-specific allocation/deallocation
    if (pattern == PatternType::SMALL_ALLOCS) {
        // Many small allocations: 1KB, 4KB, 16KB, 64KB
        std::vector<std::size_t> sizes = {1024, 4096, 16384, 65536};
        std::vector<rmm::device_uvector<char>> buffers;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto size = sizes[i % sizes.size()];
            buffers.emplace_back(size, stream, mr);
            result.total_bytes += size;
            result.num_operations++;
        }
        stream.synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        result.alloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        buffers.clear();
        end = std::chrono::high_resolution_clock::now();
        result.dealloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
    } else if (pattern == PatternType::LARGE_ALLOCS) {
        // Fewer large allocations: 1MB, 10MB, 50MB, 100MB
        std::vector<std::size_t> sizes = {1024*1024, 10*1024*1024, 50*1024*1024, 100*1024*1024};
        std::vector<rmm::device_uvector<char>> buffers;
        
        auto start = std::chrono::high_resolution_clock::now();
        int num_allocs = std::min(iterations / 4, 10); // Limit large allocations
        for (int i = 0; i < num_allocs; ++i) {
            auto size = sizes[i % sizes.size()];
            buffers.emplace_back(size, stream, mr);
            result.total_bytes += size;
            result.num_operations++;
        }
        stream.synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        result.alloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        buffers.clear();
        end = std::chrono::high_resolution_clock::now();
        result.dealloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
    } else if (pattern == PatternType::MIXED) {
        // Mixed small and large
        std::vector<rmm::device_uvector<char>> buffers;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::size_t size;
            if (i % 3 == 0) {
                size = 1024; // Small
            } else if (i % 3 == 1) {
                size = 64 * 1024; // Medium
            } else {
                size = 5 * 1024 * 1024; // Large
            }
            buffers.emplace_back(size, stream, mr);
            result.total_bytes += size;
            result.num_operations++;
        }
        stream.synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        result.alloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        buffers.clear();
        end = std::chrono::high_resolution_clock::now();
        result.dealloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
    } else if (pattern == PatternType::SEQUENTIAL) {
        // Alloc then dealloc immediately (many alloc/dealloc pairs)
        double alloc_sum = 0, dealloc_sum = 0;
        
        for (int i = 0; i < iterations; ++i) {
            std::size_t size = 1024 * (1 + (i % 16)); // 1KB to 16KB
            
            auto a_start = std::chrono::high_resolution_clock::now();
            auto a_end = std::chrono::high_resolution_clock::now();
            auto d_start = std::chrono::high_resolution_clock::now();
            
            {
                rmm::device_uvector<char> buffer(size, stream, mr);
                stream.synchronize();
                a_end = std::chrono::high_resolution_clock::now();
                
                d_start = std::chrono::high_resolution_clock::now();
                // Buffer goes out of scope here and deallocates
            }
            auto d_end = std::chrono::high_resolution_clock::now();
            
            alloc_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(a_end - a_start).count() / 1000.0;
            dealloc_sum += std::chrono::duration_cast<std::chrono::nanoseconds>(d_end - d_start).count() / 1000.0;
            
            result.total_bytes += size;
            result.num_operations++;
        }
        result.alloc_time_us = alloc_sum;
        result.dealloc_time_us = dealloc_sum;
        
    } else if (pattern == PatternType::BATCH) {
        // Alloc many, then dealloc all
        std::vector<rmm::device_uvector<char>> buffers;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            std::size_t size = 1024 * 1024; // 1MB each
            buffers.emplace_back(size, stream, mr);
            result.total_bytes += size;
            result.num_operations++;
        }
        stream.synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        result.alloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        buffers.clear();
        end = std::chrono::high_resolution_clock::now();
        result.dealloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
    } else if (pattern == PatternType::RANDOM_SIZES) {
        // Random allocation sizes between 1KB and 10MB
        std::uniform_int_distribution<std::size_t> size_dist(1024, 10 * 1024 * 1024);
        std::vector<rmm::device_uvector<char>> buffers;
        
        auto start = std::chrono::high_resolution_clock::now();
        int num_allocs = std::min(iterations, 50); // Limit for safety
        for (int i = 0; i < num_allocs; ++i) {
            std::size_t size = size_dist(rng);
            buffers.emplace_back(size, stream, mr);
            result.total_bytes += size;
            result.num_operations++;
        }
        stream.synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        result.alloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        buffers.clear();
        end = std::chrono::high_resolution_clock::now();
        result.dealloc_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    
    if (result.num_operations > 0) {
        result.avg_alloc_time_us = result.alloc_time_us / result.num_operations;
        result.avg_dealloc_time_us = result.dealloc_time_us / result.num_operations;
    }
    
    return result;
}

// Run benchmarks for all memory resources
void run_benchmarks() {
    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Memory Allocation/Deallocation Benchmarks        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    std::vector<BenchmarkResult> results;
    
    // Define patterns to test
    struct PatternConfig {
        PatternType type;
        std::string name;
        int iterations;
    };
    
    std::vector<PatternConfig> patterns = {
        {PatternType::SMALL_ALLOCS, "Small Allocs (1-64KB)", 100},
        {PatternType::LARGE_ALLOCS, "Large Allocs (1-100MB)", 10},
        {PatternType::MIXED, "Mixed Sizes", 50},
        {PatternType::SEQUENTIAL, "Sequential Alloc/Dealloc", 100},
        {PatternType::BATCH, "Batch (Alloc Many → Dealloc All)", 50},
        {PatternType::RANDOM_SIZES, "Random Sizes (1KB-10MB)", 50}
    };
    
    // Benchmark 1: Direct CUDA
    std::cout << "\nBenchmarking cuda_memory_resource...\n";
    rmm::mr::cuda_memory_resource cuda_mr;
    for (const auto& pat : patterns) {
        auto result = benchmark_pattern(cuda_mr, pat.type, pat.iterations, 
                                        "CUDA Direct", pat.name);
        results.push_back(result);
    }
    
    // Benchmark 2: Pool
    std::cout << "Benchmarking pool_memory_resource...\n";
    rmm::mr::cuda_memory_resource upstream_pool;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> 
        pool_mr(&upstream_pool, 256 * 1024 * 1024, 1024 * 1024 * 1024);
    for (const auto& pat : patterns) {
        auto result = benchmark_pattern(pool_mr, pat.type, pat.iterations, 
                                        "Pool", pat.name);
        results.push_back(result);
    }
    
    // Benchmark 3: Binning
    std::cout << "Benchmarking binning_memory_resource...\n";
    rmm::mr::cuda_memory_resource upstream_binning;
    rmm::mr::binning_memory_resource<rmm::mr::cuda_memory_resource> 
        binning_mr(&upstream_binning, 10, 22);
    auto pool_for_binning = std::make_unique<
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            &upstream_binning, 100 * 1024 * 1024, 500 * 1024 * 1024);
    binning_mr.add_bin(4 * 1024 * 1024 + 1, pool_for_binning.get());
    for (const auto& pat : patterns) {
        auto result = benchmark_pattern(binning_mr, pat.type, pat.iterations, 
                                        "Binning", pat.name);
        results.push_back(result);
    }
    
    // Benchmark 4: Fixed-size (for patterns with fixed sizes)
    // Note: fixed_size_memory_resource only supports allocations smaller than block_size
    std::cout << "Benchmarking fixed_size_memory_resource...\n";
    rmm::mr::cuda_memory_resource upstream_fixed;
    constexpr std::size_t fixed_block_size = 64 * 1024;
    rmm::mr::fixed_size_memory_resource<rmm::mr::cuda_memory_resource>
        fixed_mr(&upstream_fixed, fixed_block_size, 256);
    // Only benchmark patterns that work with fixed-size (allocations must be < block_size)
    for (const auto& pat : patterns) {
        if (pat.type == PatternType::SEQUENTIAL) {  // Sequential uses 1-16KB, which fits
            try {
                auto result = benchmark_pattern(fixed_mr, pat.type, pat.iterations, 
                                                "Fixed-Size", pat.name);
                results.push_back(result);
            } catch (const std::exception& e) {
                std::cout << "  Skipping " << pat.name << " (incompatible with fixed-size)\n";
            }
        }
    }
    
    // Benchmark 5: CUDA Async (if available)
    std::cout << "Benchmarking cuda_async_memory_resource...\n";
    try {
        rmm::mr::cuda_async_memory_resource async_mr{
            rmm::percent_of_free_device_memory(50)};
        for (const auto& pat : patterns) {
            auto result = benchmark_pattern(async_mr, pat.type, pat.iterations, 
                                            "CUDA Async", pat.name);
            results.push_back(result);
        }
    } catch (...) {
        std::cout << "  (CUDA async resource not available, skipping)\n";
    }
    
    // Print results table
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "BENCHMARK RESULTS SUMMARY\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    // Group by pattern
    for (const auto& pat : patterns) {
        std::cout << "\nPattern: " << pat.name << "\n";
        std::cout << std::string(100, '-') << "\n";
        std::cout << std::left 
                  << std::setw(15) << "Resource"
                  << std::setw(12) << "Alloc (us)"
                  << std::setw(12) << "Dealloc (us)"
                  << std::setw(12) << "Avg Alloc"
                  << std::setw(12) << "Avg Dealloc"
                  << std::setw(10) << "Ops"
                  << std::setw(15) << "Total Bytes"
                  << "\n";
        std::cout << std::string(100, '-') << "\n";
        
        for (const auto& r : results) {
            if (r.pattern_name == pat.name) {
                std::cout << std::left 
                          << std::setw(15) << r.resource_name
                          << std::setw(12) << std::fixed << std::setprecision(2) << r.alloc_time_us
                          << std::setw(12) << std::fixed << std::setprecision(2) << r.dealloc_time_us
                          << std::setw(12) << std::fixed << std::setprecision(3) << r.avg_alloc_time_us
                          << std::setw(12) << std::fixed << std::setprecision(3) << r.avg_dealloc_time_us
                          << std::setw(10) << r.num_operations
                          << std::setw(15) << format_bytes(r.total_bytes)
                          << "\n";
            }
        }
    }
    
    // Performance comparison summary
    std::cout << "\n\n" << std::string(100, '=') << "\n";
    std::cout << "PERFORMANCE INSIGHTS\n";
    std::cout << std::string(100, '=') << "\n";
    
    // Find best performers for each pattern
    for (const auto& pat : patterns) {
        std::cout << "\n" << pat.name << ":\n";
        double best_alloc = 1e9;
        double best_dealloc = 1e9;
        std::string best_alloc_resource, best_dealloc_resource;
        
        for (const auto& r : results) {
            if (r.pattern_name == pat.name) {
                if (r.avg_alloc_time_us < best_alloc) {
                    best_alloc = r.avg_alloc_time_us;
                    best_alloc_resource = r.resource_name;
                }
                if (r.avg_dealloc_time_us < best_dealloc) {
                    best_dealloc = r.avg_dealloc_time_us;
                    best_dealloc_resource = r.resource_name;
                }
            }
        }
        
        if (best_alloc_resource == best_dealloc_resource) {
            std::cout << "  Fastest: " << best_alloc_resource 
                      << " (alloc: " << std::fixed << std::setprecision(3) << best_alloc 
                      << " us, dealloc: " << best_dealloc << " us)\n";
        } else {
            std::cout << "  Fastest alloc: " << best_alloc_resource 
                      << " (" << std::fixed << std::setprecision(3) << best_alloc << " us)\n";
            std::cout << "  Fastest dealloc: " << best_dealloc_resource 
                      << " (" << best_dealloc << " us)\n";
        }
    }
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
        
        // Run benchmarks
        run_benchmarks();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

