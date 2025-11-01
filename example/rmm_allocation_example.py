#!/usr/bin/env python3
"""
RMM Allocation Strategy Example (Python)

This example demonstrates:
1. How RMM handles small vs large allocations
2. Different memory resource types
3. When cudaFree is actually called (through Python's RMM interface)
"""

import rmm
import rmm.mr
import numpy as np
import time

def format_bytes(bytes_val):
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def example_direct_cuda_allocations():
    """Example 1: Direct CUDA allocations (immediate cudaFree)"""
    print("\n" + "="*60)
    print("Example 1: Direct CUDA Allocations (CudaMemoryResource)")
    print("Note: Each deallocate() immediately calls cudaFree()")
    print("="*60)
    
    # Use default CUDA memory resource
    cuda_mr = rmm.mr.CudaMemoryResource()
    rmm.mr.set_current_device_resource(cuda_mr)
    
    print("\nAllocating small buffers (1KB, 4KB, 16KB):")
    small_buffers = []
    for size in [1024, 4096, 16384]:
        buf = rmm.DeviceBuffer(size=size)
        small_buffers.append(buf)
        print(f"  Allocated {format_bytes(size)} at pointer: {buf.ptr}")
    
    print("\nAllocating large buffers (1MB, 10MB, 100MB):")
    large_buffers = []
    for size in [1024*1024, 10*1024*1024, 100*1024*1024]:
        buf = rmm.DeviceBuffer(size=size)
        large_buffers.append(buf)
        print(f"  Allocated {format_bytes(size)} at pointer: {buf.ptr}")
    
    print("\nWhen these buffers are deleted, cudaFree() is called immediately.")
    del small_buffers, large_buffers

def example_pool_allocations():
    """Example 2: Pool allocations (delayed cudaFree)"""
    print("\n" + "="*60)
    print("Example 2: Pool Memory Resource")
    print("Note: Memory goes to pool free list, NO immediate cudaFree()")
    print("="*60)
    
    # Create upstream resource
    upstream_mr = rmm.mr.CudaMemoryResource()
    
    # Create pool with initial 256MB, max 1GB
    # Note: sizes can be strings like "256MB" or integers
    pool_mr = rmm.mr.PoolMemoryResource(
        upstream_mr,  # First argument is the upstream resource
        initial_pool_size="256MB",
        maximum_pool_size="1GB"
    )
    
    print(f"\nPool created:")
    print(f"  Initial pool size: 256MB")
    print(f"  Maximum pool size: 1GB")
    
    rmm.mr.set_current_device_resource(pool_mr)
    
    print("\nAllocating 10 buffers of varying sizes from pool:")
    buffers = []
    for i in range(10):
        size = (i + 1) * 1024 * 1024  # 1MB to 10MB
        buf = rmm.DeviceBuffer(size=size)
        buffers.append(buf)
        print(f"  Buffer {i+1}: {format_bytes(size)}")
    
    print("\nFreeing all buffers...")
    del buffers
    
    print("Note: cudaFree() NOT called yet - memory remains in pool!")
    print("cudaFree() will only be called when pool_mr is destroyed.")

def example_binning_allocations():
    """Example 3: Binning memory resource (optimized for different sizes)"""
    print("\n" + "="*60)
    print("Example 3: Binning Memory Resource")
    print("Routes allocations to different resources based on size")
    print("="*60)
    
    upstream_mr = rmm.mr.CudaMemoryResource()
    
    # Create binning resource
    # Small allocations will use fixed-size bins
    # Large allocations will use upstream resource
    binning_mr = rmm.mr.BinningMemoryResource(
        upstream_mr,  # First argument is the upstream resource
        min_size_exponent=10,  # 2^10 = 1KB
        max_size_exponent=22   # 2^22 = 4MB
    )
    
    # For allocations > 4MB, use the pool
    # Note: Python API may not expose add_bin directly, so we use a simplified version
    rmm.mr.set_current_device_resource(binning_mr)
    
    print("\nAllocation size routing:")
    print("  Small (< 1KB):      Fixed-size bin")
    print("  Medium (1KB-4MB):   Fixed-size bins")
    print("  Large (> 4MB):      Upstream resource (or configured pool)")
    
    print("\nSmall allocations (use fixed-size bins):")
    small_buffers = []
    for size in [512, 2048, 8192, 32768, 131072]:
        buf = rmm.DeviceBuffer(size=size)
        small_buffers.append(buf)
        print(f"  {format_bytes(size)} -> Fixed-size bin")
    
    print("\nLarge allocations:")
    large_buffers = []
    for size in [5*1024*1024, 20*1024*1024]:
        buf = rmm.DeviceBuffer(size=size)
        large_buffers.append(buf)
        print(f"  {format_bytes(size)} -> Pool/upstream resource")
    
    print("\nWhen buffers are freed:")
    print("  Small buffers: Returned to fixed-size bin free lists")
    print("  Large buffers: Returned to pool/upstream")
    print("  cudaFree() only called when resources are destroyed")
    
    del small_buffers, large_buffers

def example_reinitialize_pool():
    """Example 4: Using reinitialize for easy pool setup"""
    print("\n" + "="*60)
    print("Example 4: Easy Pool Setup with reinitialize()")
    print("="*60)
    
    # Simple way to set up RMM with a pool
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size="256MB",
        maximum_pool_size="1GB"
    )
    
    print("\nRMM initialized with pool:")
    print("  Pool allocator: Enabled")
    print("  Initial pool size: 256MB")
    print("  Maximum pool size: 1GB")
    
    print("\nAllocating buffers (all use the pool):")
    buffers = []
    start_time = time.time()
    
    for i in range(20):
        size = (i % 5 + 1) * 1024 * 1024  # 1MB to 5MB
        buf = rmm.DeviceBuffer(size=size)
        buffers.append(buf)
    
    alloc_time = time.time() - start_time
    print(f"  Allocated {len(buffers)} buffers in {alloc_time*1000:.2f} ms")
    
    num_buffers = len(buffers)
    start_time = time.time()
    del buffers
    free_time = time.time() - start_time
    print(f"  Freed {num_buffers} buffers in {free_time*1000:.2f} ms")
    
    print("\nNote: Memory is in the pool, ready for reuse!")
    print("No cudaFree() called until RMM is finalized or pool is destroyed.")

def example_statistics_tracking():
    """Example 5: Track allocation statistics"""
    print("\n" + "="*60)
    print("Example 5: Statistics Resource Adaptor")
    print("Track memory allocation patterns")
    print("="*60)
    
    upstream_mr = rmm.mr.CudaMemoryResource()
    
    # Wrap with statistics tracking
    # StatisticsResourceAdaptor takes upstream as first argument
    stats_mr = rmm.mr.StatisticsResourceAdaptor(upstream_mr)
    rmm.mr.set_current_device_resource(stats_mr)
    
    print("\nAllocating buffers with statistics tracking:")
    buffers = []
    
    for i, size in enumerate([1024, 4096, 1024*1024, 10*1024*1024]):
        buf = rmm.DeviceBuffer(size=size)
        buffers.append(buf)
        print(f"  Buffer {i+1}: {format_bytes(size)}")
    
    # Get statistics (Statistics object with attributes, not a dict)
    stats = stats_mr.allocation_counts
    print("\nAllocation Statistics:")
    print(f"  Current bytes allocated: {format_bytes(stats.current_bytes)}")
    print(f"  Current allocation count: {stats.current_count}")
    print(f"  Peak bytes allocated: {format_bytes(stats.peak_bytes)}")
    print(f"  Peak allocation count: {stats.peak_count}")
    print(f"  Total bytes allocated: {format_bytes(stats.total_bytes)}")
    print(f"  Total allocation count: {stats.total_count}")
    
    print("\nFreeing buffers...")
    del buffers
    
    stats = stats_mr.allocation_counts
    print("\nStatistics after freeing:")
    print(f"  Current bytes allocated: {format_bytes(stats.current_bytes)}")
    print(f"  Current allocation count: {stats.current_count}")
    print(f"  Peak bytes was: {format_bytes(stats.peak_bytes)}")

def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     RMM Allocation Strategy Examples (Python)            ║")
    print("║     Understanding Large vs Small Allocations             ║")
    print("║     When cudaFree() is Actually Called                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    try:
        # Show GPU info
        import rmm._cuda.gpu as gpu
        device_id = gpu.getDevice()
        print(f"\nGPU Information:")
        print(f"  Device ID: {device_id}")
        
        # Run examples
        example_direct_cuda_allocations()
        example_pool_allocations()
        example_reinitialize_pool()
        example_statistics_tracking()
        
        print("\n" + "="*60)
        print("Summary:")
        print("  1. CudaMemoryResource: Immediate cudaFree()")
        print("  2. PoolMemoryResource: Delayed cudaFree() (on destruction)")
        print("  3. BinningMemoryResource: Routes to appropriate resource")
        print("  4. reinitialize(): Easy way to set up pool allocation")
        print("  5. StatisticsResourceAdaptor: Track allocation patterns")
        print("="*60)
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

