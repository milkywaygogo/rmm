# RMM (RAPIDS Memory Manager) Study Guide

## Overview

RMM is a GPU memory management library that provides a unified interface for allocating and managing device (GPU) and host memory. It's designed to optimize memory allocation patterns in GPU-accelerated applications.

## Core Concepts

### 1. Memory Resources

The foundation of RMM is the **device_memory_resource** abstract base class. All memory allocation in RMM goes through memory resources.

**Key Memory Resources:**
- `cuda_memory_resource`: Basic wrapper around cudaMalloc/cudaFree
- `pool_memory_resource`: Pre-allocates a pool for fast sub-allocations
- `managed_memory_resource`: Uses CUDA Unified Memory
- `cuda_async_memory_resource`: Uses CUDA's async memory allocator

### 2. Resource Adaptors

Adaptors wrap existing resources to add functionality:
- `logging_resource_adaptor`: Logs all allocations/deallocations
- `statistics_resource_adaptor`: Tracks allocation statistics
- `limiting_resource_adaptor`: Enforces memory limits
- `binning_memory_resource`: Optimizes for different allocation sizes

### 3. Core Data Structures

**device_buffer** (cpp/include/rmm/device_buffer.hpp:82-477)
- RAII wrapper for untyped device memory
- Supports resize, shrink_to_fit operations
- Stream-ordered allocation/deallocation

**device_uvector** (cpp/include/rmm/device_uvector.hpp:77-636)
- Typed, uninitialized device vector
- Similar to std::vector but for GPU memory
- No default initialization (performance optimization)

### 4. Stream-Ordered Memory Allocation

All RMM allocations are stream-ordered:
- Memory allocated on stream A is only valid for use on stream A
- Cross-stream usage requires synchronization
- Enables memory reuse without synchronization overhead

## C++ Usage Pattern

```cpp
// 1. Create a memory resource
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &cuda_mr, 
    initial_pool_size
);

// 2. Set as default (optional)
rmm::mr::set_current_device_resource(&pool_mr);

// 3. Use with data structures
rmm::cuda_stream stream;
rmm::device_uvector<float> vec(1000, stream);
rmm::device_buffer buffer(size_bytes, stream);

// 4. Memory is automatically freed when objects go out of scope
```

## Python API

RMM's Python API provides high-level access to memory management:

```python
import rmm

# Initialize with pool allocator
rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=2**30  # 1 GiB
)

# Create DeviceBuffer
dbuf = rmm.DeviceBuffer(size=1000)

# Integration with CuPy/Numba
from rmm.allocators.cupy import rmm_cupy_allocator
cp.cuda.set_allocator(rmm_cupy_allocator)
```

## Memory Management Strategies

### 1. Pool Allocation
Best for applications with:
- Frequent allocations/deallocations
- Known memory requirements
- Performance-critical paths

### 2. Managed Memory
Use when:
- Need CPU/GPU access to same data
- Prototyping or debugging
- Not performance critical

### 3. Binning + Pool
Optimal for:
- Mixed allocation sizes
- Reducing fragmentation
- Complex allocation patterns

## Python Bindings Architecture

The Python bindings use Cython to wrap C++ classes:

1. **pylibrmm/**: Low-level Cython bindings
2. **rmm/**: High-level Python API
3. **allocators/**: Integration with CuPy, Numba, PyTorch

Key features:
- Automatic memory management via Python GC
- Integration with popular GPU libraries
- Statistics and logging capabilities

## Best Practices

1. **Choose the Right Resource**
   - Start with pool_memory_resource for most applications
   - Use statistics/logging adaptors during development
   - Consider binning for varied allocation sizes

2. **Stream Management**
   - Always specify streams for operations
   - Synchronize when sharing memory across streams
   - Use stream pools for concurrent operations

3. **Error Handling**
   - RMM throws exceptions on allocation failure
   - Check available memory before large allocations
   - Use limiting adaptors to prevent OOM

4. **Performance Tips**
   - Pre-allocate pools based on expected usage
   - Avoid frequent pool resizing
   - Use device_uvector for typed data
   - Leverage stream ordering for efficiency

## Integration Example

```cpp
// Create a resource hierarchy
rmm::mr::cuda_memory_resource cuda_mr;
rmm::mr::pool_memory_resource pool_mr(&cuda_mr, 1ULL << 30);
rmm::mr::statistics_resource_adaptor stats_mr(&pool_mr);
rmm::mr::logging_resource_adaptor log_mr(&stats_mr, "memory.log");

// Use for allocation
rmm::device_buffer buffer(size, stream, &log_mr);
```

This creates a chain: logging → statistics → pool → CUDA, giving you logging, statistics, and pooled allocation in one setup.