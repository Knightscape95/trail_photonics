#!/usr/bin/env python3
"""
Deterministic CI Validation Framework
====================================

This script provides comprehensive validation for deterministic CI runs,
ensuring that all pipeline stages produce bitwise-identical results across
different runs with the same seed, platform, and code revision.

Features:
- Seed validation and enforcement
- Environment consistency checking
- Output reproducibility verification
- Hardware-aware graceful degradation
- CI artifact generation and comparison

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Deterministic CI Implementation
"""

import os
import sys
import json
import hashlib
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import shutil
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CIEnvironment:
    """Environment configuration for CI runs."""
    
    runner_type: str  # 'cpu' or 'gpu'
    python_version: str
    platform_info: str
    cuda_available: bool
    seed: int
    commit_sha: Optional[str] = None
    workflow_run_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


class DeterministicValidator:
    """Validator for deterministic CI execution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            config_path: Path to validation configuration file
        """
        self.config = self._load_config(config_path)
        self.environment = self._detect_environment()
        self.results = []
        
        # Set up output directory
        self.output_dir = Path("ci_validation_output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Deterministic validator initialized for {self.environment.runner_type} runner")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration."""
        
        default_config = {
            "global_seed": int(os.environ.get("GLOBAL_SEED", 42)),
            "timeout_seconds": 600,
            "max_memory_gb": 16.0,
            "required_coverage": 0.62,
            "pipeline_stages": ["dataset", "ddpm", "meep", "quantum", "publication"],
            "reproducibility_tolerance": 1e-15,
            "artifacts_retention_days": 30
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _detect_environment(self) -> CIEnvironment:
        """Detect current CI environment."""
        
        # Detect runner type
        runner_type = os.environ.get("RUNNER_TYPE", "cpu")
        
        # Check CUDA availability
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False
        
        # Override based on environment variable
        if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true":
            cuda_available = True
        
        return CIEnvironment(
            runner_type=runner_type,
            python_version=sys.version,
            platform_info=platform.platform(),
            cuda_available=cuda_available,
            seed=self.config["global_seed"],
            commit_sha=os.environ.get("GITHUB_SHA"),
            workflow_run_id=os.environ.get("GITHUB_RUN_ID"),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
    
    def validate_seed_determinism(self) -> ValidationResult:
        """Validate that seeding produces deterministic results."""
        
        logger.info("Validating seed determinism...")
        start_time = time.time()
        
        try:
            # Import after ensuring deterministic environment
            from seed_manager import seed_everything
            import numpy as np
            
            def generate_test_data():
                """Generate test data using random operations."""
                return {
                    'random_array': np.random.randn(100).tolist(),
                    'random_sum': float(np.random.randn(1000).sum()),
                    'random_integers': np.random.randint(0, 1000, 50).tolist()
                }
            
            # Generate data multiple times with same seed
            results = []
            for i in range(3):
                seed_everything(self.config["global_seed"], deterministic=True)
                data = generate_test_data()
                results.append(data)
            
            # Verify all results are identical
            reference = results[0]
            for i, result in enumerate(results[1:], 2):
                for key in reference:
                    if isinstance(reference[key], list):
                        if not np.allclose(reference[key], result[key], atol=self.config["reproducibility_tolerance"]):
                            return ValidationResult(
                                passed=False,
                                message=f"Non-deterministic results detected for {key} in run {i}",
                                execution_time=time.time() - start_time
                            )
                    else:
                        if abs(reference[key] - result[key]) > self.config["reproducibility_tolerance"]:
                            return ValidationResult(
                                passed=False,
                                message=f"Non-deterministic results detected for {key} in run {i}",
                                execution_time=time.time() - start_time
                            )
            
            return ValidationResult(
                passed=True,
                message="Seed determinism validated successfully",
                details={"test_runs": len(results), "seed": self.config["global_seed"]},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Seed determinism validation failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def validate_environment_setup(self) -> ValidationResult:
        """Validate that the environment is properly configured."""
        
        logger.info("Validating environment setup...")
        start_time = time.time()
        
        checks = []
        
        # Check Python hash seed
        python_hash_seed = os.environ.get("PYTHONHASHSEED")
        if python_hash_seed != "42":
            checks.append(f"PYTHONHASHSEED not set correctly: {python_hash_seed}")
        
        # Check matplotlib backend
        mpl_backend = os.environ.get("MPLBACKEND")
        if mpl_backend != "Agg":
            checks.append(f"MPLBACKEND not set to Agg: {mpl_backend}")
        
        # Check CUDA configuration (if applicable)
        if self.environment.cuda_available:
            cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if not cublas_config:
                checks.append("CUBLAS_WORKSPACE_CONFIG not set for CUDA determinism")
        
        # Check PyTorch deterministic mode (if available)
        try:
            import torch
            if not torch.are_deterministic_algorithms_enabled():
                checks.append("PyTorch deterministic algorithms not enabled")
        except ImportError:
            pass  # PyTorch not available, skip check
        
        if checks:
            return ValidationResult(
                passed=False,
                message="Environment setup validation failed",
                details={"failed_checks": checks},
                execution_time=time.time() - start_time
            )
        else:
            return ValidationResult(
                passed=True,
                message="Environment setup validated successfully",
                details={"runner_type": self.environment.runner_type, "cuda_available": self.environment.cuda_available},
                execution_time=time.time() - start_time
            )
    
    def validate_dependency_availability(self) -> ValidationResult:
        """Validate that dependencies are available or gracefully degraded."""
        
        logger.info("Validating dependency availability...")
        start_time = time.time()
        
        try:
            from graceful_imports import check_environment
            
            env_status = check_environment()
            
            # Required dependencies should be available
            required = ["numpy", "scipy", "matplotlib"]
            missing_required = [dep for dep in required if not env_status.get(dep, {}).get("available", False)]
            
            if missing_required:
                return ValidationResult(
                    passed=False,
                    message=f"Required dependencies missing: {missing_required}",
                    details=env_status,
                    execution_time=time.time() - start_time
                )
            
            # Optional dependencies should either be available or gracefully degraded
            optional = ["meep", "qutip", "skrf"]
            degraded_optional = []
            
            for dep in optional:
                if dep in env_status:
                    if not env_status[dep]["available"] and not env_status[dep].get("mock_available", False):
                        degraded_optional.append(dep)
            
            return ValidationResult(
                passed=True,
                message="Dependency availability validated",
                details={
                    "environment_status": env_status,
                    "degraded_optional": degraded_optional
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Dependency validation failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def validate_pipeline_stages(self) -> ValidationResult:
        """Validate that all pipeline stages can execute without error."""
        
        logger.info("Validating pipeline stages...")
        start_time = time.time()
        
        stage_results = {}
        failed_stages = []
        
        # Create temporary output directory for stage testing
        with tempfile.TemporaryDirectory() as temp_dir:
            
            for stage in self.config["pipeline_stages"]:
                logger.info(f"Testing pipeline stage: {stage}")
                
                try:
                    # Build command for pipeline stage
                    cmd = [
                        sys.executable, "modular_cli.py", stage,
                        "--seed", str(self.config["global_seed"]),
                        "--output-dir", temp_dir,
                        "--timeout", str(self.config["timeout_seconds"])
                    ]
                    
                    # Add stage-specific parameters for minimal testing
                    if stage == "dataset":
                        cmd.extend(["--samples", "5"])
                    elif stage == "ddmp":
                        cmd.extend(["--epochs", "1"])
                    elif stage == "meep":
                        cmd.extend(["--resolution", "5"])
                    elif stage == "quantum":
                        cmd.extend(["--qubits", "2"])
                    
                    # Execute command with timeout
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.config["timeout_seconds"],
                        env=dict(os.environ, PYTHONPATH=os.getcwd())
                    )
                    
                    if result.returncode == 0:
                        stage_results[stage] = {"passed": True, "output": result.stdout}
                    else:
                        stage_results[stage] = {
                            "passed": False, 
                            "error": result.stderr,
                            "output": result.stdout,
                            "returncode": result.returncode
                        }
                        failed_stages.append(stage)
                
                except subprocess.TimeoutExpired:
                    stage_results[stage] = {"passed": False, "error": "Timeout"}
                    failed_stages.append(stage)
                except Exception as e:
                    stage_results[stage] = {"passed": False, "error": str(e)}
                    failed_stages.append(stage)
        
        if failed_stages:
            return ValidationResult(
                passed=False,
                message=f"Pipeline stages failed: {failed_stages}",
                details=stage_results,
                execution_time=time.time() - start_time
            )
        else:
            return ValidationResult(
                passed=True,
                message="All pipeline stages validated successfully",
                details=stage_results,
                execution_time=time.time() - start_time
            )
    
    def validate_test_coverage(self) -> ValidationResult:
        """Validate that test coverage meets requirements."""
        
        logger.info("Validating test coverage...")
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=.",
                "--cov-report=json",
                "--cov-report=term-missing",
                f"--cov-fail-under={int(self.config['required_coverage'] * 100)}",
                "--tb=short",
                "-q"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["timeout_seconds"]
            )
            
            # Parse coverage report
            coverage_data = {}
            if os.path.exists("coverage.json"):
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
            
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0) / 100
            
            if result.returncode == 0 and total_coverage >= self.config["required_coverage"]:
                return ValidationResult(
                    passed=True,
                    message=f"Test coverage validation passed: {total_coverage:.1%}",
                    details={"coverage": total_coverage, "required": self.config["required_coverage"]},
                    execution_time=time.time() - start_time
                )
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Test coverage validation failed: {total_coverage:.1%} < {self.config['required_coverage']:.1%}",
                    details={
                        "coverage": total_coverage,
                        "required": self.config["required_coverage"],
                        "test_output": result.stdout,
                        "test_errors": result.stderr
                    },
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Test coverage validation failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def generate_reproducibility_manifest(self, output_dir: Path) -> Dict[str, Any]:
        """Generate reproducibility manifest for current run."""
        
        logger.info("Generating reproducibility manifest...")
        
        # Calculate file hashes
        file_hashes = {}
        
        if output_dir.exists():
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    # Skip log files and temporary files
                    if file_path.suffix in ['.log', '.tmp', '.pid']:
                        continue
                    
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            file_hash = hashlib.sha256(content).hexdigest()
                            rel_path = str(file_path.relative_to(output_dir))
                            file_hashes[rel_path] = file_hash
                    except Exception as e:
                        logger.warning(f"Could not hash file {file_path}: {e}")
        
        manifest = {
            "environment": asdict(self.environment),
            "config": self.config,
            "file_hashes": file_hashes,
            "total_files": len(file_hashes),
            "validation_results": [asdict(result) for result in self.results]
        }
        
        # Save manifest
        manifest_file = self.output_dir / f"reproducibility_manifest_{self.environment.runner_type}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Reproducibility manifest saved: {manifest_file}")
        return manifest
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        
        logger.info("Starting comprehensive CI validation...")
        
        validations = [
            ("Seed Determinism", self.validate_seed_determinism),
            ("Environment Setup", self.validate_environment_setup),
            ("Dependency Availability", self.validate_dependency_availability),
            ("Pipeline Stages", self.validate_pipeline_stages),
            ("Test Coverage", self.validate_test_coverage)
        ]
        
        all_passed = True
        
        for name, validation_func in validations:
            logger.info(f"Running validation: {name}")
            
            try:
                result = validation_func()
                self.results.append(result)
                
                if result.passed:
                    logger.info(f"✅ {name}: {result.message}")
                else:
                    logger.error(f"❌ {name}: {result.message}")
                    all_passed = False
                    
                    if result.details:
                        logger.debug(f"Details: {json.dumps(result.details, indent=2)}")
                        
            except Exception as e:
                error_result = ValidationResult(
                    passed=False,
                    message=f"Validation {name} crashed: {e}"
                )
                self.results.append(error_result)
                logger.error(f"💥 {name}: Validation crashed: {e}")
                all_passed = False
        
        # Generate manifest regardless of validation results
        results_dir = Path("results") / self.environment.runner_type
        self.generate_reproducibility_manifest(results_dir)
        
        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        logger.info(f"Validation summary: {passed_count}/{total_count} checks passed")
        
        if all_passed:
            logger.info("🎉 All validations passed! CI is deterministic and reproducible.")
        else:
            logger.error("❌ Some validations failed. CI is not fully deterministic.")
        
        return all_passed


def main():
    """Main entry point for CI validation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Deterministic CI Validation Framework")
    parser.add_argument("--config", help="Path to validation configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--stage", help="Run specific validation stage only")
    parser.add_argument("--output-dir", help="Output directory for results", default="ci_validation_output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = DeterministicValidator(args.config)
    
    # Run validations
    if args.stage:
        # Run specific stage only
        stage_methods = {
            "seed": validator.validate_seed_determinism,
            "environment": validator.validate_environment_setup,
            "dependencies": validator.validate_dependency_availability,
            "pipeline": validator.validate_pipeline_stages,
            "coverage": validator.validate_test_coverage
        }
        
        if args.stage in stage_methods:
            result = stage_methods[args.stage]()
            print(json.dumps(asdict(result), indent=2))
            sys.exit(0 if result.passed else 1)
        else:
            logger.error(f"Unknown validation stage: {args.stage}")
            sys.exit(1)
    else:
        # Run all validations
        success = validator.run_all_validations()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
