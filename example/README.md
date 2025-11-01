# RMM Allocation Strategy Examples

This directory (`cursor_example/`) contains comprehensive examples demonstrating how RMM handles large and small allocations, and when `cudaFree()` is actually called.

## Files

- `rmm_allocation_example.cpp` - C++ example with detailed allocation strategies
- `rmm_allocation_example.py` - Python example with the same concepts
- `build_example.sh` - Automated build script

## Building the C++ Example

### Prerequisites

1. RMM must be built and installed (or available via CMake)
2. CUDA toolkit installed
3. CMake 3.20 or later

### Build Steps

**Option 1: Use the build script (recommended):**

```bash
cd cursor_example
./build_example.sh
cd build_example
./rmm_allocation_example
```

**Option 2: Manual build:**

```bash
cd cursor_example
mkdir build_example && cd build_example
cmake .. -DCMAKE_PREFIX_PATH=../../cpp/build/install
make rmm_allocation_example
./rmm_allocation_example
```

**Option 3: If RMM is installed system-wide:**

```bash
cd cursor_example
mkdir build_example && cd build_example
cmake .. -DCMAKE_PREFIX_PATH=/path/to/rmm/install
make rmm_allocation_example
./rmm_allocation_example
```

**Option 4: Using the existing RMM build system:**

```bash
# If RMM is already built, you can add this example to the existing build
# Add to cpp/examples/CMakeLists.txt or create a new example directory
```

## Running the Python Example

### Prerequisites

1. RMM Python package installed (`pip install rmm` or built from source)

### Run

```bash
cd cursor_example
python rmm_allocation_example.py
```

## What the Examples Demonstrate

### 1. Direct CUDA Allocations (`cuda_memory_resource`)
- **Behavior**: Each `deallocate()` immediately calls `cudaFree()`
- **Use Case**: Simple applications, when you want immediate memory release
- **Performance**: Slower for frequent allocations/deallocations

### 2. Pool Allocations (`pool_memory_resource`)
- **Behavior**: Memory returns to pool free list, NO immediate `cudaFree()`
- **Use Case**: Applications with many allocations/deallocations
- **Performance**: Much faster - O(log n) allocation, O(1) deallocation
- **When cudaFree is called**: Only when the pool resource is destroyed

### 3. Binning Memory Resource (`binning_memory_resource`)
- **Behavior**: Routes allocations to different resources based on size
  - Small allocations (< threshold): Fixed-size bins
  - Large allocations (> threshold): Pool resource
- **Use Case**: Mixed workload with both small and large allocations
- **Performance**: Optimized for each size range

### 4. Fixed-Size Memory Resource (`fixed_size_memory_resource`)
- **Behavior**: O(1) allocation/deallocation for fixed block sizes
- **Use Case**: Many allocations of the same size
- **Performance**: Fastest for same-sized allocations
- **When cudaFree is called**: Only when resource is destroyed

### 5. CUDA Async Memory Resource (`cuda_async_memory_resource`)
- **Behavior**: Uses CUDA driver's built-in memory pool
- **Use Case**: Stream-ordered allocations (CUDA 11.2+)
- **Performance**: Very efficient, managed by CUDA driver
- **When cudaFree is called**: Managed by CUDA driver, not explicit

## Key Takeaways

1. **Small vs Large Allocations**:
   - Small allocations benefit from fixed-size bins or pools
   - Large allocations use pool or direct CUDA allocation
   - Binning resource automatically routes to the best strategy

2. **When cudaFree() is Called**:
   - `cuda_memory_resource`: Immediately on deallocate()
   - `pool_memory_resource`: Only on resource destruction
   - `fixed_size_memory_resource`: Only on resource destruction
   - `cuda_async_memory_resource`: Managed by CUDA driver

3. **Performance Implications**:
   - Pool-based resources are 10-100x faster for repeated allocations
   - Memory stays in pool for reuse, reducing fragmentation
   - Trade-off: Memory is not immediately available to other processes

## Memory Lifecycle Example

```
1. Application allocates 1MB buffer
   ↓
2. Pool memory resource checks free list
   ↓
3. If available: Returns block from free list (FAST, no CUDA call)
   If not: Expands pool by allocating from upstream (cudaMalloc)
   ↓
4. Application uses buffer
   ↓
5. Application deallocates buffer
   ↓
6. Buffer returned to pool free list (NO cudaFree yet)
   ↓
7. Pool resource destroyed (e.g., program exit)
   ↓
8. ALL pool memory freed to upstream (cudaFree called here)
```

## Troubleshooting

### C++ Build Issues

- **Cannot find RMM**: Set `CMAKE_PREFIX_PATH` to RMM install directory
- **CUDA architecture errors**: Modify `CUDA_ARCHITECTURES` in CMakeLists.txt for your GPU
- **Link errors**: Ensure RMM is built with the same CUDA version

### Python Runtime Issues

- **Import errors**: Ensure RMM Python package is installed: `python -c "import rmm; print(rmm.__version__)"`
- **GPU errors**: Ensure CUDA is properly installed and GPU is accessible

### Memory Issues

- **Out of memory**: Reduce allocation sizes in examples or increase GPU memory
- **Pool exhausted**: Increase `maximum_pool_size` parameter

## Further Reading

- [RMM Documentation](https://docs.rapids.ai/api/rmm/stable/)
- [RMM README](README.md)
- [CUDA Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)

