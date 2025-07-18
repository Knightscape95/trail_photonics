"""
Memory Safety Guards for Electromagnetic Simulations
===================================================

Critical fix for code review blocking issue #3: Unbounded memory use
with `RevolutionaryMEEPEngine._estimate_memory_gb` predicting >100 GB
for default 30 px/µm with 3-D FDTD.

This module provides:
- Memory estimation and validation
- Automatic resolution scaling
- Domain decomposition suggestions
- Memory budget enforcement
- GPU memory guards

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Code Review Critical Fix #3
"""

import numpy as np
import psutil
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory constants (in bytes)
GB = 1024**3
MB = 1024**2
KB = 1024

# Safety margins
DEFAULT_MEMORY_SAFETY_MARGIN = 0.8  # Use only 80% of available memory
GPU_MEMORY_SAFETY_MARGIN = 0.7      # Use only 70% of GPU memory


@dataclass
class MemoryRequirements:
    """Memory requirements for a simulation."""
    total_gb: float
    fields_gb: float
    geometry_gb: float
    pml_gb: float
    sources_gb: float
    monitors_gb: float
    overhead_gb: float


@dataclass
class MemoryBudget:
    """Available memory budget."""
    total_ram_gb: float
    available_ram_gb: float
    total_gpu_gb: float
    available_gpu_gb: float
    swap_gb: float


@dataclass
class SimulationDimensions:
    """Simulation grid dimensions."""
    nx: int
    ny: int
    nz: int
    nt: int = 1
    dtype_size: int = 8  # bytes per float64
    
    @property
    def total_points(self) -> int:
        """Total number of grid points."""
        return self.nx * self.ny * self.nz * self.nt


class MemoryManager:
    """
    Memory management for electromagnetic simulations.
    
    Provides estimation, validation, and optimization of memory usage.
    """
    
    def __init__(self, safety_margin: float = DEFAULT_MEMORY_SAFETY_MARGIN, 
                 max_memory_gb: Optional[float] = None, logger=None):
        """
        Initialize memory manager.
        
        Args:
            safety_margin: Fraction of available memory to use (0.0-1.0)
            max_memory_gb: Maximum memory limit in GB (overrides safety_margin if provided)
            logger: Optional logger instance
        """
        self.safety_margin = safety_margin
        self.max_memory_gb = max_memory_gb
        self.logger = logger or logging.getLogger(__name__)
        self.budget = self._get_memory_budget()
        
    def _get_memory_budget(self) -> MemoryBudget:
        """Get current system memory budget."""
        
        # System RAM
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / GB
        available_ram_gb = memory.available / GB
        
        # Swap
        swap = psutil.swap_memory()
        swap_gb = swap.total / GB
        
        # GPU memory (if available)
        total_gpu_gb = 0.0
        available_gpu_gb = 0.0
        
        try:
            # Try to get GPU memory info
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_gpu_gb = torch.cuda.get_device_properties(device).total_memory / GB
                available_gpu_gb = (torch.cuda.get_device_properties(device).total_memory - 
                                  torch.cuda.memory_allocated(device)) / GB
        except ImportError:
            try:
                import cupy
                if cupy.cuda.is_available():
                    mempool = cupy.get_default_memory_pool()
                    total_gpu_gb = mempool.total_bytes() / GB
                    available_gpu_gb = (mempool.total_bytes() - mempool.used_bytes()) / GB
            except ImportError:
                logger.debug("No GPU memory info available")
        
        budget = MemoryBudget(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            total_gpu_gb=total_gpu_gb,
            available_gpu_gb=available_gpu_gb,
            swap_gb=swap_gb
        )
        
        # Override available RAM if max_memory_gb is specified
        if self.max_memory_gb is not None:
            budget.available_ram_gb = min(self.max_memory_gb, budget.available_ram_gb)
            self.logger.info(f"💾 Memory limit set to {self.max_memory_gb:.1f} GB")
        
        self.logger.info(f"💾 Memory Budget:")
        self.logger.info(f"   RAM: {budget.available_ram_gb:.1f} GB available / {total_ram_gb:.1f} GB total")
        if total_gpu_gb > 0:
            self.logger.info(f"   GPU: {available_gpu_gb:.1f} GB available / {total_gpu_gb:.1f} GB total")
        self.logger.info(f"   Safety margin: {self.safety_margin:.0%}")
        
        return budget
    
    def estimate_meep_memory(self, 
                            resolution: float,
                            cell_size: Tuple[float, float, float],
                            runtime: float,
                            pml_thickness: float = 1.0,
                            n_sources: int = 1,
                            n_monitors: int = 5) -> MemoryRequirements:
        """
        Estimate memory requirements for MEEP simulation.
        
        Args:
            resolution: Grid resolution (pixels per unit length)
            cell_size: Simulation cell size (x, y, z)
            runtime: Simulation runtime
            pml_thickness: PML boundary thickness
            n_sources: Number of sources
            n_monitors: Number of field monitors
        
        Returns:
            Detailed memory requirements breakdown
        """
        
        # Grid dimensions
        nx = int(np.ceil(cell_size[0] * resolution))
        ny = int(np.ceil(cell_size[1] * resolution))
        nz = int(np.ceil(cell_size[2] * resolution))
        
        # Time steps (rough estimate based on CFL condition)
        dt = 0.5 / (resolution * np.sqrt(3))  # Conservative CFL
        nt = int(np.ceil(runtime / dt))
        
        dims = SimulationDimensions(nx, ny, nz, nt)
        
        logger.debug(f"Grid dimensions: {nx} × {ny} × {nz} × {nt}")
        logger.debug(f"Total grid points: {dims.total_points:,}")
        
        # MEEP field components (Ex, Ey, Ez, Hx, Hy, Hz = 6 components)
        # Plus additional fields for dispersive materials, nonlinearities, etc.
        n_field_components = 12  # Conservative estimate
        
        fields_gb = (dims.nx * dims.ny * dims.nz * 
                    n_field_components * dims.dtype_size) / GB
        
        # Geometry arrays (permittivity, permeability, conductivity)
        n_material_components = 6
        geometry_gb = (dims.nx * dims.ny * dims.nz * 
                      n_material_components * dims.dtype_size) / GB
        
        # PML arrays (absorbing boundaries)
        pml_points = self._estimate_pml_points(dims, pml_thickness, resolution)
        pml_gb = (pml_points * n_field_components * dims.dtype_size) / GB
        
        # Sources memory (typically small)
        sources_gb = n_sources * 0.001  # 1 MB per source
        
        # Monitors memory (depends on runtime and frequency)
        monitor_points = n_monitors * dims.nx * dims.ny * dims.nz
        monitors_gb = (monitor_points * dims.dtype_size * 10) / GB  # 10 freq points
        
        # MEEP overhead (FFTs, temporary arrays, etc.)
        overhead_gb = max(1.0, fields_gb * 0.2)  # 20% overhead minimum 1GB
        
        total_gb = fields_gb + geometry_gb + pml_gb + sources_gb + monitors_gb + overhead_gb
        
        requirements = MemoryRequirements(
            total_gb=total_gb,
            fields_gb=fields_gb,
            geometry_gb=geometry_gb,
            pml_gb=pml_gb,
            sources_gb=sources_gb,
            monitors_gb=monitors_gb,
            overhead_gb=overhead_gb
        )
        
        logger.debug(f"Memory breakdown:")
        logger.debug(f"  Fields:     {fields_gb:.2f} GB")
        logger.debug(f"  Geometry:   {geometry_gb:.2f} GB")
        logger.debug(f"  PML:        {pml_gb:.2f} GB")
        logger.debug(f"  Sources:    {sources_gb:.2f} GB")
        logger.debug(f"  Monitors:   {monitors_gb:.2f} GB")
        logger.debug(f"  Overhead:   {overhead_gb:.2f} GB")
        logger.debug(f"  TOTAL:      {total_gb:.2f} GB")
        
        return requirements
    
    def _estimate_pml_points(self, 
                            dims: SimulationDimensions,
                            pml_thickness: float,
                            resolution: float) -> int:
        """Estimate number of grid points in PML regions."""
        
        pml_pixels = int(pml_thickness * resolution)
        
        # PML on all six faces of the simulation volume
        pml_volume = (
            2 * pml_pixels * dims.ny * dims.nz +  # x faces
            2 * dims.nx * pml_pixels * dims.nz +  # y faces  
            2 * dims.nx * dims.ny * pml_pixels    # z faces
        )
        
        return pml_volume
    
    def validate_memory_requirements(self, requirements: MemoryRequirements) -> Dict[str, any]:
        """
        Validate if memory requirements can be satisfied.
        
        Args:
            requirements: Estimated memory requirements
        
        Returns:
            Validation results with recommendations
        """
        
        result = {
            'can_fit': False,
            'exceeds_budget': False,
            'recommended_action': None,
            'max_safe_memory_gb': 0.0,
            'memory_reduction_needed': 0.0,
            'suggestions': []
        }
        
        # Calculate maximum safe memory usage
        max_safe_ram = self.budget.available_ram_gb * self.safety_margin
        max_safe_gpu = self.budget.available_gpu_gb * GPU_MEMORY_SAFETY_MARGIN
        
        result['max_safe_memory_gb'] = max(max_safe_ram, max_safe_gpu)
        
        # Check if requirements fit in budget
        if requirements.total_gb <= max_safe_ram:
            result['can_fit'] = True
            result['recommended_action'] = 'proceed'
            logger.info(f"✅ Memory requirements ({requirements.total_gb:.1f} GB) fit in RAM budget")
            
        elif requirements.total_gb <= max_safe_gpu and max_safe_gpu > 0:
            result['can_fit'] = True
            result['recommended_action'] = 'use_gpu'
            logger.info(f"✅ Memory requirements ({requirements.total_gb:.1f} GB) fit in GPU budget")
            
        else:
            result['exceeds_budget'] = True
            result['memory_reduction_needed'] = requirements.total_gb - result['max_safe_memory_gb']
            result['recommended_action'] = 'reduce_memory'
            
            logger.warning(f"⚠️  Memory requirements ({requirements.total_gb:.1f} GB) exceed budget")
            logger.warning(f"    Available RAM: {max_safe_ram:.1f} GB")
            if max_safe_gpu > 0:
                logger.warning(f"    Available GPU: {max_safe_gpu:.1f} GB")
            logger.warning(f"    Reduction needed: {result['memory_reduction_needed']:.1f} GB")
            
            # Generate suggestions
            result['suggestions'] = self._generate_memory_reduction_suggestions(requirements)
        
        return result
    
    def _generate_memory_reduction_suggestions(self, 
                                             requirements: MemoryRequirements) -> List[str]:
        """Generate suggestions for reducing memory usage."""
        
        suggestions = []
        
        # Resolution reduction (most effective)
        if requirements.fields_gb > 5.0:  # Large field arrays
            reduction_factor = np.cbrt(requirements.total_gb / (self.budget.available_ram_gb * 0.5))
            new_resolution_factor = 1.0 / reduction_factor
            suggestions.append(
                f"Reduce resolution by factor {reduction_factor:.1f} "
                f"(multiply resolution by {new_resolution_factor:.2f})"
            )
        
        # Domain decomposition
        if requirements.total_gb > 10.0:
            suggestions.append("Use domain decomposition (MPI parallelization)")
            suggestions.append("Split simulation into smaller chunks")
        
        # Reduce simulation time
        suggestions.append("Reduce simulation runtime if possible")
        
        # Optimize monitors
        if requirements.monitors_gb > 1.0:
            suggestions.append("Reduce number of field monitors")
            suggestions.append("Use lower frequency resolution for monitors")
        
        # Use lower precision
        suggestions.append("Consider using single precision (float32) instead of double")
        
        # Hardware suggestions
        if self.budget.total_ram_gb < 32:
            suggestions.append("Consider upgrading to ≥32 GB RAM")
        
        if self.budget.total_gpu_gb == 0:
            suggestions.append("Consider using GPU acceleration")
        
        return suggestions
    
    def suggest_optimal_resolution(self, 
                                  cell_size: Tuple[float, float, float],
                                  runtime: float,
                                  target_memory_gb: Optional[float] = None) -> float:
        """
        Suggest optimal resolution for given memory budget.
        
        Args:
            cell_size: Simulation cell size
            runtime: Simulation runtime
            target_memory_gb: Target memory usage (uses available if None)
        
        Returns:
            Recommended resolution
        """
        
        if target_memory_gb is None:
            target_memory_gb = self.budget.available_ram_gb * self.safety_margin
        
        # Binary search for optimal resolution
        res_min, res_max = 1.0, 100.0
        
        for _ in range(20):  # Max 20 iterations
            res_mid = (res_min + res_max) / 2
            
            requirements = self.estimate_meep_memory(res_mid, cell_size, runtime)
            
            if requirements.total_gb < target_memory_gb:
                res_min = res_mid
            else:
                res_max = res_mid
            
            if res_max - res_min < 0.1:
                break
        
        optimal_resolution = res_min
        
        logger.info(f"🎯 Optimal resolution: {optimal_resolution:.1f} pixels/unit")
        logger.info(f"   Estimated memory: {self.estimate_meep_memory(optimal_resolution, cell_size, runtime).total_gb:.1f} GB")
        
        return optimal_resolution
    
    def enforce_memory_budget(self, 
                            resolution: float,
                            cell_size: Tuple[float, float, float],
                            runtime: float) -> Tuple[bool, Dict[str, any]]:
        """
        Enforce memory budget for simulation parameters.
        
        Args:
            resolution: Proposed resolution
            cell_size: Simulation cell size
            runtime: Simulation runtime
        
        Returns:
            Tuple of (can_proceed, validation_result)
        
        Raises:
            MemoryError: If requirements exceed budget and no solution found
        """
        
        requirements = self.estimate_meep_memory(resolution, cell_size, runtime)
        validation = self.validate_memory_requirements(requirements)
        
        if validation['can_fit']:
            return True, validation
        
        if validation['exceeds_budget']:
            error_msg = (
                f"Memory requirements ({requirements.total_gb:.1f} GB) exceed available budget "
                f"({validation['max_safe_memory_gb']:.1f} GB). "
                f"Suggestions: {'; '.join(validation['suggestions'][:3])}"
            )
            
            logger.error(f"❌ {error_msg}")
            raise MemoryError(error_msg)
        
        return False, validation
    
    def get_current_usage(self) -> float:
        """Get current memory usage in GB."""
        memory = psutil.virtual_memory()
        return memory.used / GB
    
    def check_memory_limit(self) -> bool:
        """Check if current memory usage is within limits."""
        current_usage = self.get_current_usage()
        if self.max_memory_gb is not None:
            return current_usage <= self.max_memory_gb
        else:
            # Use safety margin of available memory
            available = self.budget.available_ram_gb
            return current_usage <= (available * self.safety_margin)
    
    def start_monitoring(self) -> None:
        """Start memory monitoring (placeholder for future implementation)."""
        self.logger.debug("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring (placeholder for future implementation)."""
        self.logger.debug("Memory monitoring stopped")
    
    def cleanup(self) -> None:
        """Clean up memory manager resources."""
        self.logger.debug("Memory manager cleanup completed")


def check_memory_safety(resolution: float,
                       cell_size: Tuple[float, float, float],
                       runtime: float,
                       safety_margin: float = DEFAULT_MEMORY_SAFETY_MARGIN) -> bool:
    """
    Quick check if simulation parameters are memory-safe.
    
    Args:
        resolution: Grid resolution
        cell_size: Simulation cell size
        runtime: Simulation runtime
        safety_margin: Memory safety margin (0.0-1.0)
    
    Returns:
        True if memory-safe, False otherwise
    """
    
    manager = MemoryManager(safety_margin)
    requirements = manager.estimate_meep_memory(resolution, cell_size, runtime)
    validation = manager.validate_memory_requirements(requirements)
    
    return validation['can_fit']


def auto_scale_resolution(cell_size: Tuple[float, float, float],
                         runtime: float,
                         max_memory_gb: Optional[float] = None) -> float:
    """
    Automatically scale resolution to fit memory budget.
    
    Args:
        cell_size: Simulation cell size
        runtime: Simulation runtime
        max_memory_gb: Maximum memory budget (auto-detect if None)
    
    Returns:
        Scaled resolution that fits in memory budget
    """
    
    manager = MemoryManager()
    return manager.suggest_optimal_resolution(cell_size, runtime, max_memory_gb)


if __name__ == "__main__":
    # Test memory management system
    print("🧪 Testing Memory Management System")
    
    # Initialize manager
    manager = MemoryManager()
    
    # Test simulation parameters
    resolution = 30.0  # pixels/μm (high resolution)
    cell_size = (10.0, 5.0, 2.0)  # μm
    runtime = 1000.0  # time units
    
    print(f"\n📊 Testing simulation parameters:")
    print(f"   Resolution: {resolution} pixels/μm")
    print(f"   Cell size: {cell_size} μm")
    print(f"   Runtime: {runtime} time units")
    
    # Estimate memory
    requirements = manager.estimate_meep_memory(resolution, cell_size, runtime)
    print(f"\n💾 Memory requirements: {requirements.total_gb:.1f} GB")
    
    # Validate requirements
    validation = manager.validate_memory_requirements(requirements)
    
    if validation['can_fit']:
        print("✅ Simulation fits in memory budget")
    else:
        print("❌ Simulation exceeds memory budget")
        print("\n💡 Suggestions:")
        for suggestion in validation['suggestions']:
            print(f"   • {suggestion}")
    
    # Test auto-scaling
    optimal_res = manager.suggest_optimal_resolution(cell_size, runtime)
    print(f"\n🎯 Optimal resolution: {optimal_res:.1f} pixels/μm")
    
    print("🎉 Memory management testing complete!")
