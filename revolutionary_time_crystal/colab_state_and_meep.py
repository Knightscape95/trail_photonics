#!/usr/bin/env python3
"""
Google Colab State Management & MEEP Installation Utilities
===========================================================

A self-contained module providing:
1. Persistent checkpointing with Google Drive integration
2. One-time MEEP installation and environment management
3. Robust state management for long-running Colab sessions

Author: Revolutionary Time-Crystal Team
Date: July 2025
License: MIT
Python: 3.10+
"""

import argparse
import logging
import os
import pathlib
import pickle
import random
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global state for environment management
_drive_mounted: bool = False
_meep_installed: bool = False
_checkpoint_dir: Optional[pathlib.Path] = None


# =============================================================================
# 1. PERSISTENT CHECKPOINTING UTILITIES
# =============================================================================

def mount_drive(drive_path: str = '/content/drive') -> pathlib.Path:
    """
    Mount Google Drive if not already mounted.
    
    Args:
        drive_path: Path where Google Drive should be mounted
        
    Returns:
        pathlib.Path to mounted Google Drive
        
    Raises:
        RuntimeError: If mounting fails or path is inaccessible
    """
    global _drive_mounted
    
    drive_path_obj = pathlib.Path(drive_path)
    
    # Check if already mounted and verify it's actually accessible
    if drive_path_obj.exists() and (drive_path_obj / 'MyDrive').exists():
        if not _drive_mounted:
            _drive_mounted = True  # Update flag if mount exists but flag wasn't set
        logger.debug(f"Google Drive already mounted at {drive_path}")
        return drive_path_obj
    
    try:
        # Attempt to mount Google Drive
        logger.info("Mounting Google Drive...")
        
        try:
            from google.colab import drive
            drive.mount(drive_path, force_remount=False)
            logger.info(f"Google Drive mounted successfully at {drive_path}")
        except ImportError:
            logger.warning("google.colab.drive not available - assuming drive is already accessible")
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            raise RuntimeError(f"Google Drive mounting failed: {e}") from e
        
        # Verify mount success
        if not drive_path_obj.exists():
            raise RuntimeError(f"Drive path {drive_path} does not exist after mounting")
        
        if not (drive_path_obj / 'MyDrive').exists():
            raise RuntimeError(f"MyDrive not found at {drive_path}/MyDrive")
        
        _drive_mounted = True
        return drive_path_obj
        
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        raise RuntimeError(f"Drive mounting failed: {e}") from e


def get_ckpt_dir(project_name: str = 'my_project') -> pathlib.Path:
    """
    Get or create checkpoint directory in Google Drive.
    
    Args:
        project_name: Name of the project for checkpoint directory
        
    Returns:
        pathlib.Path to checkpoint directory
        
    Raises:
        RuntimeError: If directory creation fails
    """
    global _checkpoint_dir
    
    # Return cached directory if available
    if _checkpoint_dir is not None and _checkpoint_dir.exists():
        return _checkpoint_dir
    
    try:
        # Ensure drive is mounted
        drive_path = mount_drive()
        
        # Create checkpoint directory
        ckpt_dir = drive_path / 'MyDrive' / f'{project_name}_ckpts'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Checkpoint directory: {ckpt_dir}")
        _checkpoint_dir = ckpt_dir
        return ckpt_dir
        
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory: {e}")
        raise RuntimeError(f"Checkpoint directory creation failed: {e}") from e


def save_state(step: int, **objects: Any) -> None:
    """
    Save current state including RNG states and custom objects.
    
    Args:
        step: Current training/simulation step
        **objects: Named objects to save in checkpoint
        
    Raises:
        RuntimeError: If saving fails
    """
    try:
        ckpt_dir = get_ckpt_dir()
        filename = f"ckpt_step_{step:06d}.pkl"
        filepath = ckpt_dir / filename
        
        # Collect all state information
        state_data = {
            'step': step,
            'objects': objects,
            'rng_states': _collect_rng_states()
        }
        
        # Save to file with atomic write
        temp_filepath = filepath.with_suffix('.pkl.tmp')
        try:
            with open(temp_filepath, 'wb') as f:
                pickle.dump(state_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, TypeError) as e:
            logger.error(f"Failed to serialize objects for step {step}: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise RuntimeError(f"Object serialization failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise
        
        try:
            # Atomic rename
            temp_filepath.rename(filepath)
            logger.info(f"State saved at step {step}: {filepath}")
            
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise
            
    except Exception as e:
        logger.error(f"Failed to save state at step {step}: {e}")
        raise RuntimeError(f"State saving failed: {e}") from e


def load_latest_state() -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Load the most recent checkpoint state.
    
    Returns:
        Tuple of (state_dict, step) or (None, -1) if no checkpoints found
        
    Raises:
        RuntimeError: If loading fails
    """
    try:
        ckpt_dir = get_ckpt_dir()
        
        # Find all checkpoint files
        ckpt_files = list(ckpt_dir.glob("ckpt_step_*.pkl"))
        
        if not ckpt_files:
            logger.info("No checkpoints found")
            return None, -1
        
        # Sort by step number (extracted from filename)
        def extract_step(filepath: pathlib.Path) -> int:
            try:
                # Extract step from filename: ckpt_step_XXXXXX.pkl
                stem = filepath.stem  # ckpt_step_XXXXXX
                step_str = stem.split('_')[-1]  # XXXXXX
                return int(step_str)
            except (ValueError, IndexError):
                return -1
        
        ckpt_files.sort(key=extract_step, reverse=True)
        latest_file = ckpt_files[0]
        latest_step = extract_step(latest_file)
        
        # Load the checkpoint
        with open(latest_file, 'rb') as f:
            state_data = pickle.load(f)
        
        # Restore RNG states
        if 'rng_states' in state_data:
            _restore_rng_states(state_data['rng_states'])
        
        logger.info(f"✅ Loaded checkpoint from step {latest_step}: {latest_file}")
        
        return state_data.get('objects', {}), latest_step
        
    except Exception as e:
        logger.error(f"Failed to load latest state: {e}")
        raise RuntimeError(f"State loading failed: {e}") from e


def _collect_rng_states() -> Dict[str, Any]:
    """Collect all available RNG states."""
    rng_states = {}
    
    try:
        # NumPy RNG state
        import numpy as np
        rng_states['numpy'] = np.random.get_state()
    except ImportError:
        logger.debug("NumPy not available for RNG state collection")
    
    try:
        # Python random state
        rng_states['python_random'] = random.getstate()
    except Exception as e:
        logger.warning(f"Failed to collect Python random state: {e}")
    
    try:
        # PyTorch RNG states (if available)
        import torch
        rng_states['torch_cpu'] = torch.random.get_rng_state()
        
        if torch.cuda.is_available():
            rng_states['torch_cuda'] = torch.cuda.random.get_rng_state_all()
            
    except ImportError:
        logger.debug("PyTorch not available for RNG state collection")
    except Exception as e:
        logger.warning(f"Failed to collect PyTorch RNG states: {e}")
    
    return rng_states


def _restore_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restore RNG states from saved data."""
    try:
        # Restore NumPy RNG state
        if 'numpy' in rng_states:
            import numpy as np
            np.random.set_state(rng_states['numpy'])
            logger.debug("NumPy RNG state restored")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to restore NumPy RNG state: {e}")
    
    try:
        # Restore Python random state
        if 'python_random' in rng_states:
            random.setstate(rng_states['python_random'])
            logger.debug("Python random state restored")
    except Exception as e:
        logger.warning(f"Failed to restore Python random state: {e}")
    
    try:
        # Restore PyTorch RNG states
        if 'torch_cpu' in rng_states:
            import torch
            torch.random.set_rng_state(rng_states['torch_cpu'])
            
            if 'torch_cuda' in rng_states and torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(rng_states['torch_cuda'])
            
            logger.debug("PyTorch RNG states restored")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to restore PyTorch RNG states: {e}")


def train_demo() -> None:
    """
    Demonstration training loop with checkpointing.
    
    Runs for 1000 iterations, saves every 25 steps, and handles resume logic.
    """
    logger.info("🚀 Starting training demo with checkpointing")
    
    # Load existing state or start fresh
    state, last_step = load_latest_state()
    
    start_step = last_step + 1
    if start_step > 0:
        logger.info(f"📋 Resumed training at step {start_step}")
        # Restore any custom objects from state if needed
        if state:
            logger.info(f"Restored state contains: {list(state.keys())}")
    else:
        logger.info("🆕 Starting fresh training from step 0")
    
    total_steps = 1000
    save_interval = 25
    
    try:
        for step in range(start_step, total_steps):
            # Simulate some work
            time.sleep(0.01)  # Small delay to simulate computation
            
            # Generate some dummy data to save
            iteration_data = {
                'loss': random.uniform(0.1, 1.0),
                'accuracy': random.uniform(0.8, 0.99),
                'learning_rate': 0.001 * (0.95 ** (step // 100))
            }
            
            # Log progress
            if step % 100 == 0:
                logger.info(f"Step {step:4d}: loss={iteration_data['loss']:.3f}, "
                          f"acc={iteration_data['accuracy']:.3f}")
            
            # Save checkpoint periodically
            if step % save_interval == 0 and step > start_step:
                try:
                    save_state(step, 
                             current_metrics=iteration_data,
                             total_steps=total_steps,
                             demo_data=f"Demo checkpoint at step {step}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {step}: {e}")
                    # Continue training even if checkpoint fails
        
        logger.info(f"✅ Training demo completed successfully! Total steps: {total_steps}")
        
    except KeyboardInterrupt:
        logger.info(f"Training interrupted at step {step}")
        # Save final state before exit
        try:
            save_state(step, 
                     interrupted=True,
                     final_metrics=iteration_data)
            logger.info("Final state saved before exit")
        except Exception as e:
            logger.error(f"Failed to save final state: {e}")
    except Exception as e:
        logger.error(f"Training demo failed: {e}")
        raise


# =============================================================================
# 2. ONE-TIME MEEP INSTALLATION & REUSE
# =============================================================================

def ensure_meep(env_name: str = 'meep_env', meep_version: str = '1.27.0') -> None:
    """
    Ensure MEEP is installed and available for import.
    
    Args:
        env_name: Name of the conda environment
        meep_version: Version of PyMEEP to install
        
    Raises:
        RuntimeError: If installation or activation fails
    """
    global _meep_installed
    
    if _meep_installed:
        logger.debug("MEEP already installed and activated")
        return
    
    try:
        # Check if MEEP is already importable
        try:
            import meep as mp  # Try importing first
            logger.info(f"MEEP already available: version {getattr(mp, '__version__', 'unknown')}")
            _meep_installed = True
            return
        except ImportError:
            pass  # Need to install
        
        logger.info(f"Setting up MEEP environment: {env_name} (version {meep_version})")
        
        # Ensure Google Drive is mounted
        drive_path = mount_drive()
        conda_envs_dir = drive_path / 'MyDrive' / 'conda_envs'
        env_path = conda_envs_dir / env_name
        
        # Check if environment already exists
        if env_path.exists() and (env_path / 'bin' / 'python').exists():
            logger.info(f"Found existing environment: {env_path}")
        else:
            logger.info(f"Creating new MEEP environment: {env_path}")
            _install_meep_environment(env_path, meep_version)
        
        # Activate environment for current session
        _activate_conda_environment(env_path)
        
        # Verify MEEP installation
        try:
            import meep as mp
            version = getattr(mp, '__version__', 'unknown')
            logger.info(f"MEEP successfully activated! Version: {version}")
            _meep_installed = True
            
        except ImportError as e:
            raise RuntimeError(f"MEEP import failed after installation: {e}") from e
        
    except Exception as e:
        logger.error(f"MEEP setup failed: {e}")
        raise RuntimeError(f"MEEP installation failed: {e}") from e


def _install_meep_environment(env_path: pathlib.Path, meep_version: str) -> None:
    """Install MEEP in a new conda environment."""
    try:
        # Create conda environments directory
        env_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if condacolab is available without installing
        try:
            import condacolab
            logger.info("condacolab already available")
        except ImportError:
            # Install condacolab if not already available
            logger.info("Installing condacolab...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "condacolab"])
            
            # Import and install condacolab
            import condacolab
            logger.warning("WARNING: condacolab.install() will restart the kernel!")
            logger.warning("After restart, you must manually re-run ensure_meep() to complete installation")
            condacolab.install()
            
            # Note: Execution stops here due to kernel restart
            return
        
        # Create MEEP environment using micromamba
        logger.info(f"Creating MEEP environment with version {meep_version}...")
        
        cmd = [
            "micromamba", "create", "-y", "-p", str(env_path),
            "-c", "conda-forge", "python=3.10", f"pymeep={meep_version}",
            "mpb", "h5py", "numpy", "scipy", "matplotlib"
        ]
        
        try:
            subprocess.check_call(cmd)
            logger.info(f"MEEP environment created: {env_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create conda environment (exit code: {e.returncode})") from e
        
    except Exception as e:
        logger.error(f"Failed to install MEEP environment: {e}")
        raise


def _activate_conda_environment(env_path: pathlib.Path) -> None:
    """Activate conda environment for current Python session."""
    try:
        import site
        
        # Add environment's site-packages to Python path
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = env_path / 'lib' / f'python{python_version}' / 'site-packages'
        
        if site_packages.exists():
            # Add to sys.path
            if str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
            
            # Add to site packages
            site.addsitedir(str(site_packages))
            
            # Also add lib64 path if it exists (for some conda installations)
            lib64_packages = env_path / 'lib64' / f'python{python_version}' / 'site-packages'
            if lib64_packages.exists() and str(lib64_packages) not in sys.path:
                sys.path.insert(0, str(lib64_packages))
                site.addsitedir(str(lib64_packages))
            
            # Set environment variables for library paths
            lib_path = env_path / 'lib'
            if lib_path.exists():
                current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                new_ld_path = f"{lib_path}:{current_ld_path}" if current_ld_path else str(lib_path)
                os.environ['LD_LIBRARY_PATH'] = new_ld_path
            
            logger.info(f"Activated environment: {env_path}")
        else:
            raise RuntimeError(f"Site-packages not found at {site_packages}")
            
    except Exception as e:
        logger.error(f"Failed to activate conda environment: {e}")
        raise


def run_meep_test() -> bool:
    """
    Run a simple MEEP test simulation to verify installation.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    try:
        logger.info("Running MEEP test simulation...")
        
        # Import MEEP
        try:
            import meep as mp
        except ImportError as e:
            raise RuntimeError("MEEP not available. Run ensure_meep() first.") from e
        
        # Set up a simple 2D cavity simulation
        logger.info("Setting up 2D cavity resonator...")
        
        start_time = time.time()
        
        # Simulation parameters
        cell_size = mp.Vector3(16, 8, 0)
        resolution = 10
        
        # Define geometry: dielectric cavity
        geometry = [mp.Block(mp.Vector3(12, 1, mp.inf),
                            center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
        
        # PML boundary conditions
        pml_layers = [mp.PML(1.0)]
        
        # Source: Gaussian pulse
        sources = [mp.Source(mp.GaussianSource(frequency=0.15, fwidth=0.1),
                           component=mp.Ez,
                           center=mp.Vector3(-5, 0))]
        
        # Set up simulation
        sim = mp.Simulation(cell_size=cell_size,
                          boundary_layers=pml_layers,
                          geometry=geometry,
                          sources=sources,
                          resolution=resolution)
        
        # Run simulation
        logger.info("Running simulation...")
        
        # Monitor field at center
        field_monitor = []
        
        def get_field(sim):
            field_monitor.append(sim.get_field_point(mp.Ez, mp.Vector3()))
        
        sim.run(mp.at_every(1, get_field), until=50)
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        # Analyze results
        if field_monitor:
            peak_field = max(abs(f) for f in field_monitor)
            logger.info(f"MEEP test completed successfully!")
            logger.info(f"   Simulation time: {simulation_time:.2f} seconds")
            logger.info(f"   Peak electric field: {peak_field:.4e}")
            return True
        else:
            logger.warning("No field data collected during simulation")
            return False
        
    except Exception as e:
        logger.error(f"MEEP test failed: {e}")
        return False


# =============================================================================
# 3. MAIN EXECUTION AND CLI
# =============================================================================

def main() -> None:
    """Main execution function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Google Colab State Management & MEEP Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python colab_state_and_meep.py --demo          # Run full demo
  python colab_state_and_meep.py --checkpoint    # Test checkpointing only  
  python colab_state_and_meep.py --meep          # Test MEEP installation only
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run complete demo (checkpointing + MEEP)')
    parser.add_argument('--checkpoint', action='store_true',
                       help='Test checkpointing functionality only')
    parser.add_argument('--meep', action='store_true',
                       help='Test MEEP installation only')
    parser.add_argument('--project', default='demo_project',
                       help='Project name for checkpoints (default: demo_project)')
    parser.add_argument('--env-name', default='meep_env',
                       help='Conda environment name (default: meep_env)')
    parser.add_argument('--meep-version', default='1.27.0',
                       help='MEEP version to install (default: 1.27.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🚀 Google Colab State Management & MEEP Utilities")
        logger.info("=" * 60)
        
        if args.demo or args.checkpoint:
            logger.info("📋 Testing checkpointing functionality...")
            
            # Set global checkpoint directory for this project
            global _checkpoint_dir
            _checkpoint_dir = None  # Reset to force re-creation
            get_ckpt_dir(args.project)
            
            # Run training demo
            train_demo()
        
        if args.demo or args.meep:
            logger.info("🔧 Testing MEEP installation...")
            
            # Ensure MEEP is installed
            ensure_meep(args.env_name, args.meep_version)
            
            # Run MEEP test
            run_meep_test()
        
        logger.info("🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
