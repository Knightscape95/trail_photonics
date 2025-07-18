"""
Graceful Import Management for Optional Dependencies
===================================================

Critical fix for code review blocking issue #2: Runtime-killing imports
that hard stop if `pymeep` or full "production engines" missing.

This module provides graceful degradation mechanisms:
- Optional imports with fallback behavior
- Environment detection
- Test skipping for missing dependencies
- Clear user messaging without pipeline abortion

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Code Review Critical Fix #2
"""

import warnings
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import sys

# Configure logging for import management
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global availability tracking
OPTIONAL_DEPENDENCIES = {
    'meep': False,
    'qutip': False,
    'torch': False,
    'cuda': False,
    'wandb': False,
    'ray': False,
    'cupy': False,
}

# Import status tracking
IMPORT_ERRORS = {}


def optional_import(module_name: str, 
                   fallback_message: Optional[str] = None,
                   raise_on_missing: bool = False):
    """
    Import a module with graceful failure handling.
    
    Args:
        module_name: Name of the module to import
        fallback_message: Custom message when module unavailable
        raise_on_missing: If True, raise ImportError; if False, return None
    
    Returns:
        Module object if successful, None if failed and raise_on_missing=False
    
    Raises:
        ImportError: If raise_on_missing=True and import fails
    """
    try:
        if '.' in module_name:
            # Handle submodule imports like 'torch.cuda'
            parts = module_name.split('.')
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_name)
        
        OPTIONAL_DEPENDENCIES[module_name.split('.')[0]] = True
        logger.debug(f"✅ Successfully imported {module_name}")
        return module
        
    except ImportError as e:
        IMPORT_ERRORS[module_name] = str(e)
        OPTIONAL_DEPENDENCIES[module_name.split('.')[0]] = False
        
        message = fallback_message or f"Optional dependency '{module_name}' not available"
        
        if raise_on_missing:
            logger.error(f"❌ Required import {module_name} failed: {e}")
            raise ImportError(f"Required dependency '{module_name}' not available: {e}")
        else:
            logger.warning(f"⚠️  {message}: {e}")
            return None


def check_environment() -> Dict[str, bool]:
    """
    Check availability of all optional dependencies.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    logger.info("🔍 Checking environment dependencies...")
    
    # Core scientific libraries
    optional_import('numpy', 'NumPy not available - core functionality limited')
    optional_import('scipy', 'SciPy not available - optimization features limited')
    optional_import('matplotlib', 'Matplotlib not available - plotting disabled')
    
    # MEEP electromagnetic simulation
    meep = optional_import('meep', 
        'MEEP not available - electromagnetic simulations will use fallback methods')
    if meep:
        try:
            version = getattr(meep, '__version__', 'Unknown')
            logger.info(f"   📡 MEEP version: {version}")
        except:
            logger.info(f"   📡 MEEP available (version unknown)")
    
    # PyTorch for ML
    torch = optional_import('torch', 
        'PyTorch not available - 4D DDPM training disabled')
    if torch:
        logger.info(f"   🔥 PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            OPTIONAL_DEPENDENCIES['cuda'] = True
            logger.info(f"   🚀 CUDA available: {torch.cuda.device_count()} devices")
        else:
            logger.info(f"   💻 CUDA not available - using CPU")
    
    # QuTiP for quantum simulations
    qutip = optional_import('qutip', 
        'QuTiP not available - quantum state transfer simulations limited')
    if qutip:
        logger.info(f"   ⚛️  QuTiP version: {qutip.__version__}")
    
    # Weights & Biases for experiment tracking
    wandb = optional_import('wandb', 
        'Weights & Biases not available - experiment tracking disabled')
    if wandb:
        logger.info(f"   📊 WandB available")
    
    # Ray for distributed computing
    ray = optional_import('ray', 
        'Ray not available - distributed computing disabled')
    if ray:
        logger.info(f"   ☄️  Ray available")
    
    # CuPy for GPU arrays
    cupy = optional_import('cupy', 
        'CuPy not available - GPU array operations disabled')
    if cupy:
        logger.info(f"   🔢 CuPy available")
    
    return OPTIONAL_DEPENDENCIES.copy()


def require_dependency(dependency: str):
    """
    Decorator to mark functions that require specific dependencies.
    
    Args:
        dependency: Name of required dependency
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPTIONAL_DEPENDENCIES.get(dependency, False):
                error_msg = IMPORT_ERRORS.get(dependency, "Unknown import error")
                raise ImportError(
                    f"Function '{func.__name__}' requires '{dependency}' but it's not available. "
                    f"Error: {error_msg}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_missing(dependency: str, reason: Optional[str] = None):
    """
    Decorator to skip tests/functions if dependency is missing.
    
    Args:
        dependency: Name of required dependency
        reason: Optional reason for skipping
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OPTIONAL_DEPENDENCIES.get(dependency, False):
                skip_reason = reason or f"Dependency '{dependency}' not available"
                logger.info(f"⏭️  Skipping {func.__name__}: {skip_reason}")
                
                # Try to use pytest.skip if available
                try:
                    import pytest
                    pytest.skip(skip_reason)
                except ImportError:
                    # Fallback: just return None
                    return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_fallback_implementation(dependency: str, 
                              fallback_func: Optional[Callable] = None):
    """
    Decorator to provide fallback implementation when dependency missing.
    
    Args:
        dependency: Name of required dependency
        fallback_func: Function to call when dependency missing
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if OPTIONAL_DEPENDENCIES.get(dependency, False):
                return func(*args, **kwargs)
            else:
                if fallback_func:
                    logger.warning(
                        f"Using fallback implementation for {func.__name__} "
                        f"(dependency '{dependency}' not available)"
                    )
                    return fallback_func(*args, **kwargs)
                else:
                    logger.warning(
                        f"Skipping {func.__name__} - dependency '{dependency}' not available"
                    )
                    return None
        return wrapper
    return decorator


class MockModule:
    """
    Mock module for graceful degradation.
    
    Provides basic interface compatibility when real module unavailable.
    """
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._warned = False
    
    def __getattr__(self, name):
        if not self._warned:
            logger.warning(
                f"Using mock implementation of {self.module_name}.{name} - "
                f"install {self.module_name} for full functionality"
            )
            self._warned = True
        return MockModule(f"{self.module_name}.{name}")
    
    def __call__(self, *args, **kwargs):
        logger.warning(
            f"Mock call to {self.module_name}(*args, **kwargs) - "
            f"install real module for functionality"
        )
        return self


def create_mock_meep():
    """Create mock MEEP module for fallback."""
    
    class MockMEEP:
        def __init__(self):
            self.verbose = 0
        
        def Simulation(self, *args, **kwargs):
            logger.warning("Using mock MEEP simulation - install pymeep for real EM simulation")
            return MockSimulation()
        
        def Medium(self, *args, **kwargs):
            return MockMedium()
        
        def Source(self, *args, **kwargs):
            return MockSource()
        
        def Vector3(self, *args, **kwargs):
            return MockVector3(*args, **kwargs)
        
        def __getattr__(self, name):
            return MockModule(f"meep.{name}")
    
    class MockSimulation:
        def run(self, *args, **kwargs):
            logger.warning("Mock MEEP simulation run - no actual EM calculation performed")
        
        def get_array(self, *args, **kwargs):
            import numpy as np
            return np.zeros((10, 10))  # Dummy array
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    class MockMedium:
        def __init__(self, **kwargs):
            self.epsilon = kwargs.get('epsilon', 1.0)
    
    class MockSource:
        def __init__(self, **kwargs):
            pass
    
    class MockVector3:
        def __init__(self, x=0, y=0, z=0):
            self.x, self.y, self.z = x, y, z
    
    return MockMEEP()


def get_safe_meep():
    """
    Get MEEP module with graceful fallback.
    
    Returns:
        Real MEEP module if available, mock module otherwise
    """
    meep = optional_import('meep')
    if meep is not None:
        return meep
    else:
        logger.warning("Using mock MEEP implementation - install pymeep for real simulations")
        return create_mock_meep()


def create_mock_torch():
    """Create mock PyTorch module for fallback."""
    
    class MockTorch:
        def __init__(self):
            self.cuda = MockCuda()
        
        def tensor(self, data):
            import numpy as np
            return np.array(data)
        
        def randn(self, *shape):
            import numpy as np
            return np.random.randn(*shape)
        
        def manual_seed(self, seed):
            import numpy as np
            np.random.seed(seed)
        
        def __getattr__(self, name):
            return MockModule(f"torch.{name}")
    
    class MockCuda:
        def is_available(self):
            return False
        
        def device_count(self):
            return 0
    
    return MockTorch()


def print_environment_summary():
    """Print summary of environment status."""
    print("\n" + "="*60)
    print("📋 ENVIRONMENT DEPENDENCY SUMMARY")
    print("="*60)
    
    for dep, available in OPTIONAL_DEPENDENCIES.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"   {dep:<15} {status}")
        
        if not available and dep in IMPORT_ERRORS:
            print(f"   {'':<15} Error: {IMPORT_ERRORS[dep]}")
    
    print("="*60)
    
    # Recommendations
    missing_deps = [dep for dep, avail in OPTIONAL_DEPENDENCIES.items() if not avail]
    if missing_deps:
        print("\n💡 INSTALLATION RECOMMENDATIONS:")
        for dep in missing_deps:
            if dep == 'meep':
                print("   conda install -c conda-forge pymeep")
            elif dep == 'torch':
                print("   pip install torch torchvision torchaudio")
            elif dep == 'qutip':
                print("   pip install qutip")
            elif dep == 'wandb':
                print("   pip install wandb")
            elif dep == 'ray':
                print("   pip install ray[tune]")
            elif dep == 'cupy':
                print("   pip install cupy-cuda11x  # or appropriate CUDA version")
    
    print("")


if __name__ == "__main__":
    # Test the import management system
    print("🧪 Testing Graceful Import Management")
    
    # Check all dependencies
    deps = check_environment()
    
    # Print summary
    print_environment_summary()
    
    # Test decorators
    @require_dependency('numpy')
    def test_numpy_required():
        import numpy as np
        return np.array([1, 2, 3])
    
    @skip_if_missing('nonexistent_module')
    def test_skip_missing():
        return "This should be skipped"
    
    try:
        result = test_numpy_required()
        print(f"✅ NumPy test passed: {result}")
    except ImportError as e:
        print(f"❌ NumPy test failed: {e}")
    
    result = test_skip_missing()
    print(f"Skip test result: {result}")
    
    print("🎉 Import management testing complete!")
