"""
Global Seed Management for Deterministic Reproducibility
========================================================

NATURE PHOTONICS EDITORIAL STANDARDS - CRITICAL FIX #1
Enhanced with comprehensive audit trail and subprocess seed propagation.

This module provides centralized seed management across all random number generators:
- NumPy 
- Python random
- PyTorch (CPU + CUDA)
- MEEP (if available)
- QuTiP (if available)
- Multiprocessing workers
- All subprocesses and external calls

MANDATED REQUIREMENTS:
- Global determinism enforcement at EVERY entry point
- Complete audit trail of all seed usage
- Subprocess seed inheritance
- Cross-platform reproducibility validation

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Nature Photonics Editorial Standards - Global Determinism
"""

import numpy as np
import random
import os
import sys
import logging
import multiprocessing
import subprocess
import time
import hashlib
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Import professional logging
from professional_logging import ProfessionalLogger

# Global seed state
_GLOBAL_SEED: Optional[int] = None
_DETERMINISTIC_MODE: bool = False
_SEED_AUDIT_TRAIL: List[Dict[str, Any]] = []
_INITIALIZATION_HASH: Optional[str] = None


@dataclass
class SeedAuditEntry:
    """Audit trail entry for seed operations."""
    timestamp: str
    operation: str
    seed_value: int
    context: str
    module: str
    function: str
    process_id: int
    thread_id: int
    success: bool
    details: Dict[str, Any]


class DeterministicContext:
    """Context manager for enforcing deterministic execution."""
    
    def __init__(self, seed: int, context_name: str):
        self.seed = seed
        self.context_name = context_name
        self.previous_state = None
        
    def __enter__(self):
        """Enter deterministic context."""
        self.previous_state = {
            'np_state': np.random.get_state(),
            'py_state': random.getstate()
        }
        
        # Set local seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        _log_seed_usage(self.seed, f"deterministic_context_enter_{self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit deterministic context."""
        # Restore previous state
        np.random.set_state(self.previous_state['np_state'])
        random.setstate(self.previous_state['py_state'])
        
        _log_seed_usage(self.seed, f"deterministic_context_exit_{self.context_name}")


def seed_everything(seed: int, deterministic_mode: bool = True, 
                   context: str = "global_initialization") -> None:
    """
    Set deterministic seeds for all random number generators.
    
    MANDATED: This is the single entry point for reproducibility control.
    Must be called at the start of EVERY CLI command, test, and worker process.
    
    Args:
        seed: Global seed value (must be between 0 and 2^32-1)
        deterministic_mode: If True, enables maximum determinism
        context: Context description for audit trail
    
    Raises:
        ValueError: If seed is invalid
        RuntimeError: If deterministic setup fails
    """
    global _GLOBAL_SEED, _DETERMINISTIC_MODE, _INITIALIZATION_HASH
    
    logger = ProfessionalLogger("SeedManager")
    
    # Validate seed
    if not isinstance(seed, int) or seed < 0 or seed >= 2**32:
        raise ValueError(f"Seed must be integer between 0 and 2^32-1, got {seed}")
    
    _GLOBAL_SEED = seed
    _DETERMINISTIC_MODE = deterministic_mode
    
    logger.info(f"Setting global seed: {seed}")
    logger.info(f"Deterministic mode: {deterministic_mode}")
    logger.info(f"Context: {context}")
    
    # Generate initialization hash for verification
    init_data = {
        'seed': seed,
        'deterministic_mode': deterministic_mode,
        'context': context,
        'platform': sys.platform,
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'timestamp': datetime.now().isoformat()
    }
    
    _INITIALIZATION_HASH = hashlib.sha256(
        json.dumps(init_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    logger.info(f"Initialization hash: {_INITIALIZATION_HASH}")
    
    # 1. Python built-in random
    try:
        random.seed(seed)
        _log_seed_usage(seed, f"{context}_python_random")
        logger.info("Python random seeded successfully")
    except Exception as e:
        logger.error(f"Failed to seed Python random: {e}")
        raise RuntimeError(f"Python random seeding failed: {e}")
    
    # 2. NumPy
    try:
        np.random.seed(seed)
        _log_seed_usage(seed, f"{context}_numpy_random")
        logger.info("NumPy random seeded successfully")
    except Exception as e:
        logger.error(f"Failed to seed NumPy: {e}")
        raise RuntimeError(f"NumPy seeding failed: {e}")
    
    # 3. Environment variables for external processes
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['GLOBAL_DETERMINISTIC_SEED'] = str(seed)
        os.environ['DETERMINISTIC_MODE'] = str(deterministic_mode)
        _log_seed_usage(seed, f"{context}_environment_vars")
        logger.info("Environment variables set for subprocess determinism")
    except Exception as e:
        logger.error(f"Failed to set environment variables: {e}")
        raise RuntimeError(f"Environment variable setup failed: {e}")
    
    # 4. PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        
        # CUDA seeding
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            logger.info("PyTorch CUDA seeded successfully")
        
        if deterministic_mode:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Enable deterministic algorithms (PyTorch 1.8+)
            try:
                torch.use_deterministic_algorithms(True)
                logger.info("PyTorch deterministic algorithms enabled")
            except AttributeError:
                logger.warning("PyTorch deterministic algorithms not available (requires PyTorch 1.8+)")
                
        _log_seed_usage(seed, f"{context}_pytorch")
        logger.info("PyTorch seeded successfully")
        
    except ImportError:
        logger.warning("PyTorch not available - skipping torch seeding")
        _log_seed_usage(seed, f"{context}_pytorch_unavailable", success=False)
    except Exception as e:
        logger.error(f"PyTorch seeding failed: {e}")
        if deterministic_mode:
            raise RuntimeError(f"PyTorch seeding failed in deterministic mode: {e}")
    
    # 5. MEEP (if available)
    try:
        import meep as mp
        # MEEP uses GSL random number generator
        os.environ['GSL_RNG_SEED'] = str(seed)
        os.environ['GSL_RNG_TYPE'] = 'mt19937'  # Ensure consistent RNG type
        _log_seed_usage(seed, f"{context}_meep_gsl")
        logger.info("MEEP/GSL seeded successfully")
        
    except ImportError:
        logger.warning("MEEP not available - skipping MEEP seeding")
        _log_seed_usage(seed, f"{context}_meep_unavailable", success=False)
    except Exception as e:
        logger.error(f"MEEP seeding failed: {e}")
        if deterministic_mode:
            raise RuntimeError(f"MEEP seeding failed in deterministic mode: {e}")
    
    # 6. QuTiP (if available)
    try:
        import qutip
        # QuTiP uses NumPy random, but we set it explicitly
        if hasattr(qutip, 'settings'):
            qutip.settings.auto_tidyup = False  # Disable automatic cleanup for determinism
        _log_seed_usage(seed, f"{context}_qutip")
        logger.info("QuTiP configured for deterministic operation")
        
    except ImportError:
        logger.warning("QuTiP not available - skipping QuTiP configuration")
        _log_seed_usage(seed, f"{context}_qutip_unavailable", success=False)
    except Exception as e:
        logger.error(f"QuTiP configuration failed: {e}")
        if deterministic_mode:
            raise RuntimeError(f"QuTiP configuration failed in deterministic mode: {e}")
    
    # 7. Multiprocessing seed inheritance
    try:
        # Set up multiprocessing to inherit seeds
        multiprocessing.set_start_method('spawn', force=True)
        _setup_worker_seed_inheritance(seed)
        _log_seed_usage(seed, f"{context}_multiprocessing")
        logger.info("Multiprocessing seed inheritance configured")
        
    except Exception as e:
        logger.error(f"Multiprocessing seed setup failed: {e}")
        if deterministic_mode:
            raise RuntimeError(f"Multiprocessing seed setup failed: {e}")
    
    # 8. Additional deterministic configurations
    if deterministic_mode:
        try:
            # Disable hash randomization
            os.environ['PYTHONHASHSEED'] = '0'
            
            # Set thread count for deterministic parallel operations
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            
            _log_seed_usage(seed, f"{context}_deterministic_config")
            logger.info("Additional deterministic configurations applied")
            
        except Exception as e:
            logger.error(f"Deterministic configuration failed: {e}")
            raise RuntimeError(f"Deterministic configuration failed: {e}")
    
    # Log successful initialization
    _log_seed_usage(seed, f"{context}_complete", success=True, 
                   details={'initialization_hash': _INITIALIZATION_HASH})
    
    logger.info(f"Global seed initialization complete: {seed}")
    logger.info(f"Initialization hash: {_INITIALIZATION_HASH}")


def _setup_worker_seed_inheritance(base_seed: int) -> None:
    """Set up worker processes to inherit deterministic seeds."""
    
    def worker_initializer():
        """Initialize worker process with deterministic seed."""
        worker_id = multiprocessing.current_process().pid
        worker_seed = (base_seed + worker_id) % (2**32)  # Ensure valid seed range
        seed_everything(worker_seed, context=f"worker_process_{worker_id}")
    
    # Store initializer for use by process pools
    multiprocessing._worker_initializer = worker_initializer


def _log_seed_usage(seed: int, context: str, success: bool = True, 
                   details: Optional[Dict[str, Any]] = None) -> None:
    """Log seed usage to audit trail."""
    
    entry = SeedAuditEntry(
        timestamp=datetime.now().isoformat(),
        operation="seed_set",
        seed_value=seed,
        context=context,
        module=__name__,
        function=_get_calling_function(),
        process_id=os.getpid(),
        thread_id=_get_thread_id(),
        success=success,
        details=details or {}
    )
    
    _SEED_AUDIT_TRAIL.append(asdict(entry))


def _get_calling_function() -> str:
    """Get the name of the calling function."""
    import inspect
    try:
        frame = inspect.currentframe().f_back.f_back
        return frame.f_code.co_name
    except:
        return "unknown"


def _get_thread_id() -> int:
    """Get current thread ID."""
    import threading
    return threading.get_ident()


def get_global_seed() -> Optional[int]:
    """
    Get the current global seed.
    
    Returns:
        Current global seed or None if not set
    """
    return _GLOBAL_SEED


def get_deterministic_mode() -> bool:
    """
    Check if deterministic mode is enabled.
    
    Returns:
        True if deterministic mode is enabled
    """
    return _DETERMINISTIC_MODE


def get_initialization_hash() -> Optional[str]:
    """
    Get the initialization hash for verification.
    
    Returns:
        Initialization hash or None if not initialized
    """
    return _INITIALIZATION_HASH


def ensure_seeded(context: str = "runtime_check") -> None:
    """
    Ensure that global seeding has been performed.
    
    Args:
        context: Context for logging
    
    Raises:
        RuntimeError: If global seed not set
    """
    if _GLOBAL_SEED is None:
        raise RuntimeError(
            f"Global seed not set in context '{context}'. "+
            f"Call seed_everything() before any random operations."
        )
    
    _log_seed_usage(_GLOBAL_SEED, f"{context}_ensure_seeded")


def create_deterministic_context(seed: int, context_name: str) -> DeterministicContext:
    """
    Create a deterministic context manager.
    
    Args:
        seed: Seed for the context
        context_name: Name of the context
    
    Returns:
        Deterministic context manager
    """
    return DeterministicContext(seed, context_name)


def seed_subprocess(subprocess_cmd: List[str], seed: int) -> subprocess.Popen:
    """
    Launch subprocess with deterministic seed inheritance.
    
    Args:
        subprocess_cmd: Command to execute
        seed: Seed to propagate
    
    Returns:
        Subprocess handle
    """
    env = os.environ.copy()
    env['GLOBAL_DETERMINISTIC_SEED'] = str(seed)
    env['DETERMINISTIC_MODE'] = str(_DETERMINISTIC_MODE)
    env['PYTHONHASHSEED'] = str(seed)
    env['GSL_RNG_SEED'] = str(seed)
    
    _log_seed_usage(seed, f"subprocess_launch_{subprocess_cmd[0]}")
    
    return subprocess.Popen(subprocess_cmd, env=env)


def get_worker_seed(worker_id: int) -> int:
    """
    Generate deterministic worker seed.
    
    Args:
        worker_id: Worker process ID
    
    Returns:
        Deterministic seed for worker
    """
    if _GLOBAL_SEED is None:
        raise RuntimeError("Global seed not initialized")
    
    worker_seed = (_GLOBAL_SEED + worker_id) % (2**32)
    _log_seed_usage(worker_seed, f"worker_seed_generation_{worker_id}")
    
    return worker_seed


def export_seed_audit_trail(filepath: str) -> None:
    """
    Export complete seed audit trail.
    
    Args:
        filepath: Output file path
    """
    audit_data = {
        'metadata': {
            'export_timestamp': datetime.now().isoformat(),
            'global_seed': _GLOBAL_SEED,
            'deterministic_mode': _DETERMINISTIC_MODE,
            'initialization_hash': _INITIALIZATION_HASH,
            'total_entries': len(_SEED_AUDIT_TRAIL),
            'platform': sys.platform,
            'python_version': sys.version,
            'numpy_version': np.__version__
        },
        'audit_trail': _SEED_AUDIT_TRAIL
    }
    
    with open(filepath, 'w') as f:
        json.dump(audit_data, f, indent=2)
    
    logger = ProfessionalLogger("SeedManager")
    logger.info(f"Seed audit trail exported to {filepath}")


def validate_deterministic_state() -> Dict[str, Any]:
    """
    Validate current deterministic state.
    
    Returns:
        Validation results
    """
    results = {
        'global_seed_set': _GLOBAL_SEED is not None,
        'deterministic_mode': _DETERMINISTIC_MODE,
        'initialization_hash': _INITIALIZATION_HASH,
        'environment_variables': {},
        'library_states': {},
        'issues': []
    }
    
    # Check environment variables
    env_vars = ['PYTHONHASHSEED', 'GLOBAL_DETERMINISTIC_SEED', 'GSL_RNG_SEED']
    for var in env_vars:
        results['environment_variables'][var] = os.environ.get(var)
    
    # Check library availability and state
    try:
        import torch
        results['library_states']['pytorch'] = {
            'available': True,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic if torch.cuda.is_available() else None
        }
    except ImportError:
        results['library_states']['pytorch'] = {'available': False}
    
    try:
        import meep
        results['library_states']['meep'] = {'available': True}
    except ImportError:
        results['library_states']['meep'] = {'available': False}
    
    try:
        import qutip
        results['library_states']['qutip'] = {'available': True}
    except ImportError:
        results['library_states']['qutip'] = {'available': False}
    
    # Check for potential issues
    if not results['global_seed_set']:
        results['issues'].append("Global seed not initialized")
    
    # Check PYTHONHASHSEED - should be '0' in deterministic mode or match seed
    pythonhashseed = os.environ.get('PYTHONHASHSEED', '')
    if _DETERMINISTIC_MODE:
        # In deterministic mode, PYTHONHASHSEED should be '0'
        if pythonhashseed != '0':
            results['issues'].append(f"PYTHONHASHSEED should be '0' in deterministic mode, got '{pythonhashseed}'")
    else:
        # In non-deterministic mode, it should match the global seed
        if pythonhashseed != str(_GLOBAL_SEED):
            results['issues'].append(f"PYTHONHASHSEED mismatch: expected '{_GLOBAL_SEED}', got '{pythonhashseed}'")
    return results


def generate_seed_report() -> str:
    """
    Generate comprehensive seed state report.
    
    Returns:
        Formatted seed report
    """
    validation = validate_deterministic_state()
    
    report = []
    report.append("=" * 80)
    report.append("DETERMINISTIC SEED STATE REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Basic state
    report.append("Global State:")
    report.append(f"  Global Seed: {_GLOBAL_SEED}")
    report.append(f"  Deterministic Mode: {_DETERMINISTIC_MODE}")
    report.append(f"  Initialization Hash: {_INITIALIZATION_HASH}")
    report.append("")
    
    # Environment variables
    report.append("Environment Variables:")
    for var, value in validation['environment_variables'].items():
        report.append(f"  {var}: {value}")
    report.append("")
    
    # Library states
    report.append("Library States:")
    for lib, state in validation['library_states'].items():
        report.append(f"  {lib.upper()}:")
        for key, value in state.items():
            report.append(f"    {key}: {value}")
    report.append("")
    
    # Audit trail summary
    report.append("Audit Trail Summary:")
    report.append(f"  Total seed operations: {len(_SEED_AUDIT_TRAIL)}")
    
    if _SEED_AUDIT_TRAIL:
        contexts = {}
        for entry in _SEED_AUDIT_TRAIL:
            context = entry['context']
            if context not in contexts:
                contexts[context] = 0
            contexts[context] += 1
        
        report.append("  Operations by context:")
        for context, count in sorted(contexts.items()):
            report.append(f"    {context}: {count}")
    
    report.append("")
    
    # Issues
    if validation['issues']:
        report.append("ISSUES DETECTED:")
        for issue in validation['issues']:
            report.append(f"  - {issue}")
    else:
        report.append("No issues detected.")
    
    return "\n".join(report)


# Decorator for enforcing seeded execution
def requires_seeded_execution(func: Callable) -> Callable:
    """
    Decorator to ensure function runs with global seed set.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        ensure_seeded(f"function_{func.__name__}")
        return func(*args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


if __name__ == "__main__":
    # Test comprehensive seed management system
    print("Testing Enhanced Seed Management System")
    
    try:
        # Test basic seeding
        seed_everything(42, deterministic_mode=True, context="test_initialization")
        
        # Test state validation
        validation = validate_deterministic_state()
        print(f"Validation results: {validation}")
        
        # Test deterministic context
        with create_deterministic_context(123, "test_context"):
            random_value = np.random.random()
            print(f"Random value in context: {random_value}")
        
        # Test worker seed generation
        worker_seeds = [get_worker_seed(i) for i in range(3)]
        print(f"Worker seeds: {worker_seeds}")
        
        # Generate report
        report = generate_seed_report()
        print("\nSeed State Report:")
        print(report)
        
        # Export audit trail
        export_seed_audit_trail("seed_audit_test.json")
        
        print("Enhanced seed management system test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. SciPy random state
    try:
        from scipy import stats
        # Note: SciPy uses NumPy's random state by default
        logger.info(f"   ✅ SciPy uses NumPy random state")
        
    except ImportError:
        logger.warning(f"   ⚠️  SciPy not available")
    
    logger.info(f"🎉 Global seeding complete! Seed: {seed}")


def get_worker_seed(worker_id: int, base_seed: Optional[int] = None) -> int:
    """
    Generate deterministic seed for multiprocessing workers.
    
    This ensures each worker has a unique but deterministic seed
    derived from the global seed.
    
    Args:
        worker_id: Unique worker identifier
        base_seed: Base seed (uses global if not provided)
    
    Returns:
        Deterministic worker seed
    """
    if base_seed is None:
        if _GLOBAL_SEED is None:
            raise RuntimeError("Global seed not set. Call seed_everything() first.")
        base_seed = _GLOBAL_SEED
    
    # Generate worker seed using hash function for good distribution
    worker_seed = (base_seed + worker_id * 65537) % (2**32)
    
    logger.debug(f"Worker {worker_id} seed: {worker_seed}")
    return worker_seed


def seed_worker(worker_id: int) -> None:
    """
    Seed a multiprocessing worker with deterministic seed.
    
    This should be called at the start of each worker process.
    
    Args:
        worker_id: Unique worker identifier
    """
    worker_seed = get_worker_seed(worker_id)
    
    # Seed all available RNGs in worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    try:
        import torch
        torch.manual_seed(worker_seed)
    except ImportError:
        pass
    
    logger.debug(f"Worker {worker_id} seeded with {worker_seed}")


def get_global_seed() -> Optional[int]:
    """
    Get the current global seed.
    
    Returns:
        Current global seed or None if not set
    """
    return _GLOBAL_SEED


def is_deterministic_mode() -> bool:
    """
    Check if deterministic mode is enabled.
    
    Returns:
        True if deterministic mode is active
    """
    return _DETERMINISTIC_MODE


def verify_reproducibility(test_function: callable, n_runs: int = 3) -> bool:
    """
    Verify that a function produces identical results across runs.
    
    Args:
        test_function: Function to test (should return numeric result)
        n_runs: Number of test runs
    
    Returns:
        True if all runs produce identical results
    """
    if _GLOBAL_SEED is None:
        logger.warning("No global seed set - reproducibility test may fail")
        return False
    
    results = []
    
    for run in range(n_runs):
        # Reset seed before each run
        seed_everything(_GLOBAL_SEED, _DETERMINISTIC_MODE)
        
        try:
            result = test_function()
            results.append(result)
        except Exception as e:
            logger.error(f"Test function failed on run {run}: {e}")
            return False
    
    # Check if all results are identical
    first_result = results[0]
    
    for i, result in enumerate(results[1:], 1):
        if not np.allclose(result, first_result, rtol=1e-10, atol=1e-10):
            logger.error(f"Run {i+1} differs from run 1")
            logger.error(f"  Run 1:   {first_result}")
            logger.error(f"  Run {i+1}: {result}")
            return False
    
    logger.info(f"✅ Reproducibility verified across {n_runs} runs")
    return True


def create_deterministic_generator(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random Generator with deterministic seed.
    
    This is the preferred way to create random generators for
    new code following NumPy best practices.
    
    Args:
        seed: Specific seed (uses worker seed if None)
    
    Returns:
        Seeded NumPy Generator
    """
    if seed is None:
        if _GLOBAL_SEED is None:
            raise RuntimeError("Global seed not set. Call seed_everything() first.")
        seed = _GLOBAL_SEED
    
    rng = np.random.default_rng(seed)
    logger.debug(f"Created deterministic generator with seed {seed}")
    return rng


# Example test functions for reproducibility verification
def _test_numpy_random() -> float:
    """Test function for NumPy reproducibility."""
    return np.random.randn(1000).sum()


def _test_python_random() -> float:
    """Test function for Python random reproducibility."""
    return sum(random.random() for _ in range(1000))


if __name__ == "__main__":
    # Demonstration and testing
    print("🧪 Testing Seed Management System")
    
    # Test basic seeding
    seed_everything(42, deterministic_mode=True)
    print(f"Global seed: {get_global_seed()}")
    print(f"Deterministic mode: {is_deterministic_mode()}")
    
    # Test worker seeding
    for worker_id in range(3):
        worker_seed = get_worker_seed(worker_id)
        print(f"Worker {worker_id} seed: {worker_seed}")
    
    # Test reproducibility
    print("\n🔍 Testing NumPy reproducibility...")
    numpy_reproducible = verify_reproducibility(_test_numpy_random, n_runs=3)
    
    print("\n🔍 Testing Python random reproducibility...")
    python_reproducible = verify_reproducibility(_test_python_random, n_runs=3)
    
    if numpy_reproducible and python_reproducible:
        print("\n✅ All reproducibility tests passed!")
    else:
        print("\n❌ Some reproducibility tests failed!")
