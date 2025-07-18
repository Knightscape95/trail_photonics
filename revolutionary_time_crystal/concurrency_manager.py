"""
Safe Concurrency Manager for Time-Crystal Simulations
=====================================================

Critical fix for code review blocking issue #4: Concurrency hazards
with `ProcessPoolExecutor` spawning `parallel_workers = mp.cpu_count()`
with heavy NumPy objects leading to oversubscribed RAM.

This module provides:
- Intelligent worker count calculation
- Shared memory arrays for large data
- Memory-aware parallelization
- Process pool hygiene
- Resource throttling

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Code Review Critical Fix #4
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
import time
import functools
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory constants
GB = 1024**3
MB = 1024**2

# Default limits
DEFAULT_MAX_WORKERS = 8
DEFAULT_MEMORY_PER_WORKER_GB = 4.0
DEFAULT_SHARED_MEMORY_THRESHOLD_MB = 100


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency management."""
    max_workers: Optional[int] = None
    memory_per_worker_gb: float = DEFAULT_MEMORY_PER_WORKER_GB
    use_shared_memory: bool = True
    shared_memory_threshold_mb: float = DEFAULT_SHARED_MEMORY_THRESHOLD_MB
    chunk_size: Optional[int] = None
    timeout_seconds: float = 3600.0  # 1 hour default timeout
    use_threads_for_io: bool = True


class SafeProcessPool:
    """
    Memory-aware process pool for heavy computational tasks.
    
    Automatically calculates optimal worker count based on:
    - Available CPU cores
    - Available RAM
    - Memory requirements per worker
    """
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """
        Initialize safe process pool.
        
        Args:
            config: Concurrency configuration
        """
        self.config = config or ConcurrencyConfig()
        self.shared_arrays = {}  # Store shared memory arrays
        self.executor = None
        
        # Calculate optimal worker count
        self.optimal_workers = self._calculate_optimal_workers()
        
        logger.info(f"🔧 Concurrency Configuration:")
        logger.info(f"   Optimal workers: {self.optimal_workers}")
        logger.info(f"   Memory per worker: {self.config.memory_per_worker_gb:.1f} GB")
        logger.info(f"   Shared memory: {self.config.use_shared_memory}")
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker processes."""
        
        # Get system resources
        cpu_count = mp.cpu_count()
        physical_cores = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / GB
        
        logger.debug(f"System resources:")
        logger.debug(f"  CPU cores: {cpu_count} logical, {physical_cores} physical")
        logger.debug(f"  Available memory: {available_memory_gb:.1f} GB")
        
        # Memory-limited worker count
        memory_limited_workers = int(available_memory_gb / self.config.memory_per_worker_gb)
        
        # CPU-limited worker count (prefer physical cores for CPU-bound tasks)
        cpu_limited_workers = physical_cores or cpu_count
        
        # Apply user limits
        user_limited_workers = self.config.max_workers or float('inf')
        
        # Take minimum of all constraints
        optimal = min(
            memory_limited_workers,
            cpu_limited_workers,
            user_limited_workers,
            DEFAULT_MAX_WORKERS  # Safety cap
        )
        
        # Ensure at least 1 worker
        optimal = max(1, optimal)
        
        logger.debug(f"Worker constraints:")
        logger.debug(f"  Memory limited: {memory_limited_workers}")
        logger.debug(f"  CPU limited: {cpu_limited_workers}")
        logger.debug(f"  User limited: {user_limited_workers}")
        logger.debug(f"  Final optimal: {optimal}")
        
        return optimal
    
    def create_shared_array(self, 
                           name: str, 
                           shape: Tuple[int, ...], 
                           dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Create shared memory array accessible by all workers.
        
        Args:
            name: Unique name for the array
            shape: Array shape
            dtype: NumPy data type
        
        Returns:
            NumPy array backed by shared memory
        """
        
        # Calculate memory size
        array_size = np.prod(shape) * np.dtype(dtype).itemsize
        size_mb = array_size / MB
        
        logger.info(f"📝 Creating shared array '{name}': {shape} ({size_mb:.1f} MB)")
        
        if not self.config.use_shared_memory:
            logger.warning("Shared memory disabled - using regular array")
            return np.zeros(shape, dtype=dtype)
        
        try:
            # Create shared memory block
            shm = shared_memory.SharedMemory(create=True, size=array_size, name=name)
            
            # Create NumPy array view
            shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            shared_array.fill(0)  # Initialize with zeros
            
            # Store reference to prevent garbage collection
            self.shared_arrays[name] = {
                'shm': shm,
                'array': shared_array,
                'shape': shape,
                'dtype': dtype
            }
            
            logger.debug(f"✅ Shared array '{name}' created successfully")
            return shared_array
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create shared array '{name}': {e}")
            logger.warning("Falling back to regular array")
            return np.zeros(shape, dtype=dtype)
    
    def get_shared_array(self, name: str) -> Optional[np.ndarray]:
        """
        Get existing shared array by name.
        
        Args:
            name: Name of the shared array
        
        Returns:
            Shared array if exists, None otherwise
        """
        if name in self.shared_arrays:
            return self.shared_arrays[name]['array']
        
        # Try to attach to existing shared memory
        try:
            shm = shared_memory.SharedMemory(name=name)
            logger.warning(f"Found existing shared memory '{name}' - this might cause issues")
            return None
        except FileNotFoundError:
            return None
    
    def map_with_shared_memory(self, 
                              func: Callable,
                              args_list: List[Tuple],
                              shared_data: Optional[Dict[str, np.ndarray]] = None) -> List[Any]:
        """
        Map function over arguments with shared memory support.
        
        Args:
            func: Function to apply (must accept shared_data as first argument)
            args_list: List of argument tuples for each call
            shared_data: Dictionary of shared arrays to pass to workers
        
        Returns:
            List of results
        """
        
        if not args_list:
            return []
        
        # Prepare shared data info for workers
        shared_info = {}
        if shared_data:
            for name, array in shared_data.items():
                if name in self.shared_arrays:
                    shared_info[name] = {
                        'name': name,
                        'shape': self.shared_arrays[name]['shape'],
                        'dtype': self.shared_arrays[name]['dtype']
                    }
        
        # Create worker function that handles shared memory
        def worker_func(args):
            return _shared_memory_worker(func, shared_info, args)
        
        # Execute with process pool
        return self.map(worker_func, args_list)
    
    def map(self, func: Callable, args_list: List[Any], chunk_size: Optional[int] = None) -> List[Any]:
        """
        Map function over arguments using optimal process pool.
        
        Args:
            func: Function to apply
            args_list: List of arguments
            chunk_size: Chunk size for batching (auto-calculated if None)
        
        Returns:
            List of results
        """
        
        if not args_list:
            return []
        
        # Calculate chunk size if not provided
        if chunk_size is None:
            chunk_size = max(1, len(args_list) // (self.optimal_workers * 4))
        
        results = []
        failed_tasks = []
        
        logger.info(f"🚀 Starting parallel execution:")
        logger.info(f"   Tasks: {len(args_list)}")
        logger.info(f"   Workers: {self.optimal_workers}")
        logger.info(f"   Chunk size: {chunk_size}")
        
        try:
            # FIX: Add worker initialization for deterministic execution
            def _worker_initializer():
                """Initialize worker process with deterministic seed."""
                try:
                    # Import here to avoid circular imports
                    from seed_manager import seed_everything, get_global_seed
                    
                    # Get the current global seed and reseed in worker
                    global_seed = get_global_seed()
                    if global_seed is not None:
                        # Use worker-specific seed based on global seed
                        worker_seed = global_seed + mp.current_process().pid
                        seed_everything(worker_seed, deterministic_mode=True, 
                                      context=f"worker_pid_{mp.current_process().pid}")
                        logger.debug(f"Worker {mp.current_process().pid} initialized with seed {worker_seed}")
                except ImportError:
                    logger.warning("seed_manager not available in worker process")
                except Exception as e:
                    logger.warning(f"Worker initialization failed: {e}")
            
            with ProcessPoolExecutor(max_workers=self.optimal_workers, 
                                   initializer=_worker_initializer) as executor:
                # Submit all tasks
                future_to_index = {}
                for i, args in enumerate(args_list):
                    future = executor.submit(func, args)
                    future_to_index[future] = i
                
                # Collect results with timeout
                for future in as_completed(future_to_index.keys(), 
                                         timeout=self.config.timeout_seconds):
                    try:
                        result = future.result()
                        index = future_to_index[future]
                        results.append((index, result))
                        
                    except Exception as e:
                        index = future_to_index[future]
                        logger.warning(f"⚠️  Task {index} failed: {e}")
                        failed_tasks.append(index)
        
        except concurrent.futures.TimeoutError:
            logger.error(f"❌ Parallel execution timed out after {self.config.timeout_seconds}s")
            raise
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]
        
        if failed_tasks:
            logger.warning(f"⚠️  {len(failed_tasks)} tasks failed: {failed_tasks}")
        
        logger.info(f"✅ Parallel execution complete: {len(final_results)}/{len(args_list)} succeeded")
        
        return final_results
    
    def cleanup(self):
        """Clean up shared memory resources."""
        
        logger.info("🧹 Cleaning up shared memory resources...")
        
        for name, info in self.shared_arrays.items():
            try:
                shm = info['shm']
                shm.close()
                shm.unlink()
                logger.debug(f"✅ Cleaned up shared array '{name}'")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup shared array '{name}': {e}")
        
        self.shared_arrays.clear()
        
        # Force garbage collection
        gc.collect()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def _shared_memory_worker(func: Callable, 
                         shared_info: Dict[str, Dict], 
                         args: Tuple) -> Any:
    """
    Worker function that reconstructs shared memory arrays.
    
    Args:
        func: Function to call
        shared_info: Information about shared arrays
        args: Function arguments
    
    Returns:
        Function result
    """
    
    # Reconstruct shared arrays
    shared_data = {}
    for name, info in shared_info.items():
        try:
            shm = shared_memory.SharedMemory(name=info['name'])
            array = np.ndarray(info['shape'], dtype=info['dtype'], buffer=shm.buf)
            shared_data[name] = array
        except Exception as e:
            logger.warning(f"Worker failed to access shared array '{name}': {e}")
    
    # Call function with shared data
    return func(shared_data, *args)


class ThreadSafeCounter:
    """Thread-safe counter for tracking progress."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = mp.Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value


def safe_parallel_map(func: Callable,
                     args_list: List[Any],
                     max_workers: Optional[int] = None,
                     memory_per_worker_gb: float = DEFAULT_MEMORY_PER_WORKER_GB,
                     timeout_seconds: float = 3600.0) -> List[Any]:
    """
    Simple interface for safe parallel mapping.
    
    Args:
        func: Function to apply
        args_list: List of arguments
        max_workers: Maximum number of workers (auto-calculated if None)
        memory_per_worker_gb: Memory budget per worker
        timeout_seconds: Timeout for execution
    
    Returns:
        List of results
    """
    
    config = ConcurrencyConfig(
        max_workers=max_workers,
        memory_per_worker_gb=memory_per_worker_gb,
        timeout_seconds=timeout_seconds
    )
    
    with SafeProcessPool(config) as pool:
        return pool.map(func, args_list)


def estimate_memory_per_task(sample_args: Any, 
                           func: Callable,
                           safety_factor: float = 2.0) -> float:
    """
    Estimate memory usage per task by running a sample.
    
    Args:
        sample_args: Sample arguments for the function
        func: Function to test
        safety_factor: Safety factor for memory estimation
    
    Returns:
        Estimated memory per task in GB
    """
    
    import tracemalloc
    
    # Start memory tracing
    tracemalloc.start()
    
    try:
        # Run sample task
        _ = func(sample_args)
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / GB
        
        # Apply safety factor
        estimated_gb = peak_gb * safety_factor
        
        logger.info(f"📊 Memory estimation:")
        logger.info(f"   Peak usage: {peak_gb:.3f} GB")
        logger.info(f"   Estimated (with safety): {estimated_gb:.3f} GB")
        
        return estimated_gb
        
    except Exception as e:
        logger.warning(f"⚠️  Memory estimation failed: {e}")
        return DEFAULT_MEMORY_PER_WORKER_GB
        
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    # Test concurrency management
    print("🧪 Testing Safe Concurrency Management")
    
    # Test function for heavy computation
    def heavy_task(args):
        """Simulate heavy computational task."""
        n, worker_id = args
        # Simulate some computation
        result = np.sum(np.random.randn(n, n))
        time.sleep(0.1)  # Simulate work
        return f"Worker {worker_id}: {result:.3f}"
    
    # Test parameters
    task_args = [(100, i) for i in range(20)]
    
    print(f"\n📊 Testing {len(task_args)} tasks")
    
    # Test safe parallel execution
    with SafeProcessPool() as pool:
        start_time = time.time()
        results = pool.map(heavy_task, task_args)
        end_time = time.time()
    
    print(f"✅ Completed {len(results)} tasks in {end_time - start_time:.2f} seconds")
    print(f"First few results: {results[:3]}")
    
    # Test shared memory
    print("\n🧪 Testing shared memory arrays")
    
    with SafeProcessPool() as pool:
        # Create shared array
        shared_array = pool.create_shared_array("test_array", (1000, 1000), np.float32)
        shared_array.fill(42.0)
        
        print(f"Created shared array: {shared_array.shape}, mean = {shared_array.mean()}")
    
    print("🎉 Concurrency management testing complete!")
