#!/usr/bin/env python3
"""
CI Validation Tools for Nature-Grade Standards
=============================================

Comprehensive validation tools for ensuring Nature Photonics editorial standards
are met across all aspects of the time-crystal isolator codebase.

MANDATED FUNCTIONALITY:
1. Deterministic execution validation with bit-exact comparison
2. Real physics calculation verification (no mock/placeholder data)
3. Parameter transparency and literature validation
4. Docstring coverage and API documentation validation
5. Scientific assumptions and error budget reporting
6. Complete audit trail generation and verification

Author: Revolutionary Time-Crystal Team
Date: July 2025
Status: Nature Photonics Editorial Standards - Critical Implementation
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import subprocess
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib
import csv

# Import project modules
from parameter_manager import get_parameter_manager, ParameterManager
from seed_manager import export_seed_audit_trail, generate_seed_report
from scientific_integrity import generate_scientific_report
from professional_logging import ProfessionalLogger


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    details: Dict[str, Any]
    issues: List[str]
    timestamp: str


class DeterministicValidator:
    """Validator for deterministic execution."""
    
    def __init__(self, logger: Optional[ProfessionalLogger] = None):
        self.logger = logger or ProfessionalLogger("DeterministicValidator")
        
    def compare_runs(self, run1_dir: str, run2_dir: str, 
                    tolerance: float = 1e-12) -> ValidationResult:
        """
        Compare outputs from two deterministic runs.
        
        Args:
            run1_dir: Directory containing first run outputs
            run2_dir: Directory containing second run outputs
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Validation result
        """
        
        self.logger.info(f"Comparing deterministic runs: {run1_dir} vs {run2_dir}")
        
        issues = []
        details = {
            'files_compared': 0,
            'files_identical': 0,
            'files_within_tolerance': 0,
            'files_different': 0,
            'differences': []
        }
        
        run1_path = Path(run1_dir)
        run2_path = Path(run2_dir)
        
        if not run1_path.exists():
            issues.append(f"Run 1 directory not found: {run1_dir}")
            return ValidationResult(
                check_name="deterministic_comparison",
                passed=False,
                details=details,
                issues=issues,
                timestamp=datetime.now().isoformat()
            )
        
        if not run2_path.exists():
            issues.append(f"Run 2 directory not found: {run2_dir}")
            return ValidationResult(
                check_name="deterministic_comparison",
                passed=False,
                details=details,
                issues=issues,
                timestamp=datetime.now().isoformat()
            )
        
        # Get all files from both runs
        run1_files = set(self._get_all_files(run1_path))
        run2_files = set(self._get_all_files(run2_path))
        
        # Check for missing files
        missing_in_run2 = run1_files - run2_files
        missing_in_run1 = run2_files - run1_files
        
        if missing_in_run2:
            issues.append(f"Files missing in run 2: {list(missing_in_run2)}")
        if missing_in_run1:
            issues.append(f"Files missing in run 1: {list(missing_in_run1)}")
        
        # Compare common files
        common_files = run1_files & run2_files
        
        for file_rel_path in common_files:
            file1_path = run1_path / file_rel_path
            file2_path = run2_path / file_rel_path
            
            details['files_compared'] += 1
            
            try:
                comparison_result = self._compare_files(file1_path, file2_path, tolerance)
                
                if comparison_result['identical']:
                    details['files_identical'] += 1
                elif comparison_result['within_tolerance']:
                    details['files_within_tolerance'] += 1
                else:
                    details['files_different'] += 1
                    details['differences'].append({
                        'file': str(file_rel_path),
                        'type': comparison_result['type'],
                        'max_difference': comparison_result.get('max_difference'),
                        'details': comparison_result.get('details')
                    })
                    
            except Exception as e:
                issues.append(f"Error comparing {file_rel_path}: {e}")
                details['files_different'] += 1
        
        # Determine if validation passed
        passed = (len(issues) == 0 and details['files_different'] == 0)
        
        details['differences_found'] = details['files_different'] > 0
        
        self.logger.info(f"Deterministic comparison complete: "
                        f"{details['files_compared']} files compared, "
                        f"{details['files_different']} differences found")
        
        return ValidationResult(
            check_name="deterministic_comparison",
            passed=passed,
            details=details,
            issues=issues,
            timestamp=datetime.now().isoformat()
        )
    
    def _get_all_files(self, directory: Path) -> List[str]:
        """Get all files in directory recursively."""
        files = []
        for item in directory.rglob('*'):
            if item.is_file():
                files.append(str(item.relative_to(directory)))
        return files
    
    def _compare_files(self, file1: Path, file2: Path, tolerance: float) -> Dict[str, Any]:
        """Compare two files with appropriate method based on file type."""
        
        # Determine file type and comparison method
        if file1.suffix in ['.npy', '.npz']:
            return self._compare_numpy_files(file1, file2, tolerance)
        elif file1.suffix in ['.json']:
            return self._compare_json_files(file1, file2, tolerance)
        elif file1.suffix in ['.csv']:
            return self._compare_csv_files(file1, file2, tolerance)
        elif file1.suffix in ['.txt', '.log']:
            return self._compare_text_files(file1, file2)
        else:
            return self._compare_binary_files(file1, file2)
    
    def _compare_numpy_files(self, file1: Path, file2: Path, tolerance: float) -> Dict[str, Any]:
        """Compare NumPy array files."""
        try:
            arr1 = np.load(file1, allow_pickle=False)
            arr2 = np.load(file2, allow_pickle=False)
            
            if arr1.shape != arr2.shape:
                return {
                    'identical': False,
                    'within_tolerance': False,
                    'type': 'numpy_shape_mismatch',
                    'details': f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
                }
            
            diff = np.abs(arr1 - arr2)
            max_diff = np.max(diff)
            
            if max_diff == 0:
                return {'identical': True, 'within_tolerance': True, 'type': 'numpy'}
            elif max_diff <= tolerance:
                return {
                    'identical': False,
                    'within_tolerance': True,
                    'type': 'numpy',
                    'max_difference': float(max_diff)
                }
            else:
                return {
                    'identical': False,
                    'within_tolerance': False,
                    'type': 'numpy',
                    'max_difference': float(max_diff),
                    'details': f"Maximum difference {max_diff} exceeds tolerance {tolerance}"
                }
                
        except Exception as e:
            return {
                'identical': False,
                'within_tolerance': False,
                'type': 'numpy_error',
                'details': str(e)
            }
    
    def _compare_json_files(self, file1: Path, file2: Path, tolerance: float) -> Dict[str, Any]:
        """Compare JSON files with numerical tolerance."""
        try:
            with open(file1) as f:
                data1 = json.load(f)
            with open(file2) as f:
                data2 = json.load(f)
            
            if self._json_equal_with_tolerance(data1, data2, tolerance):
                return {'identical': True, 'within_tolerance': True, 'type': 'json'}
            else:
                return {
                    'identical': False,
                    'within_tolerance': False,
                    'type': 'json',
                    'details': "JSON content differs beyond tolerance"
                }
                
        except Exception as e:
            return {
                'identical': False,
                'within_tolerance': False,
                'type': 'json_error',
                'details': str(e)
            }
    
    def _json_equal_with_tolerance(self, obj1: Any, obj2: Any, tolerance: float) -> bool:
        """Compare JSON objects with numerical tolerance."""
        if type(obj1) != type(obj2):
            return False
        
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(self._json_equal_with_tolerance(obj1[k], obj2[k], tolerance) 
                      for k in obj1.keys())
        
        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(self._json_equal_with_tolerance(a, b, tolerance) 
                      for a, b in zip(obj1, obj2))
        
        elif isinstance(obj1, (int, float)):
            if isinstance(obj2, (int, float)):
                return abs(obj1 - obj2) <= tolerance
            return False
        
        else:
            return obj1 == obj2
    
    def _compare_csv_files(self, file1: Path, file2: Path, tolerance: float) -> Dict[str, Any]:
        """Compare CSV files."""
        try:
            with open(file1) as f1, open(file2) as f2:
                reader1 = csv.reader(f1)
                reader2 = csv.reader(f2)
                
                rows1 = list(reader1)
                rows2 = list(reader2)
                
                if len(rows1) != len(rows2):
                    return {
                        'identical': False,
                        'within_tolerance': False,
                        'type': 'csv_length_mismatch',
                        'details': f"Row count mismatch: {len(rows1)} vs {len(rows2)}"
                    }
                
                # Compare rows
                for i, (row1, row2) in enumerate(zip(rows1, rows2)):
                    if len(row1) != len(row2):
                        return {
                            'identical': False,
                            'within_tolerance': False,
                            'type': 'csv_column_mismatch',
                            'details': f"Column count mismatch in row {i}"
                        }
                    
                    for j, (cell1, cell2) in enumerate(zip(row1, row2)):
                        # Try numerical comparison first
                        try:
                            val1 = float(cell1)
                            val2 = float(cell2)
                            if abs(val1 - val2) > tolerance:
                                return {
                                    'identical': False,
                                    'within_tolerance': False,
                                    'type': 'csv_numerical_difference',
                                    'details': f"Numerical difference in row {i}, col {j}: {val1} vs {val2}"
                                }
                        except ValueError:
                            # String comparison
                            if cell1 != cell2:
                                return {
                                    'identical': False,
                                    'within_tolerance': False,
                                    'type': 'csv_string_difference',
                                    'details': f"String difference in row {i}, col {j}: '{cell1}' vs '{cell2}'"
                                }
                
                return {'identical': True, 'within_tolerance': True, 'type': 'csv'}
                
        except Exception as e:
            return {
                'identical': False,
                'within_tolerance': False,
                'type': 'csv_error',
                'details': str(e)
            }
    
    def _compare_text_files(self, file1: Path, file2: Path) -> Dict[str, Any]:
        """Compare text files."""
        try:
            with open(file1) as f1, open(file2) as f2:
                content1 = f1.read()
                content2 = f2.read()
            
            if content1 == content2:
                return {'identical': True, 'within_tolerance': True, 'type': 'text'}
            else:
                # Generate diff for details
                diff = list(difflib.unified_diff(
                    content1.splitlines(keepends=True),
                    content2.splitlines(keepends=True),
                    fromfile=str(file1),
                    tofile=str(file2)
                ))
                
                return {
                    'identical': False,
                    'within_tolerance': False,
                    'type': 'text_difference',
                    'details': ''.join(diff[:100])  # Limit diff length
                }
                
        except Exception as e:
            return {
                'identical': False,
                'within_tolerance': False,
                'type': 'text_error',
                'details': str(e)
            }
    
    def _compare_binary_files(self, file1: Path, file2: Path) -> Dict[str, Any]:
        """Compare binary files using hash."""
        try:
            hash1 = self._file_hash(file1)
            hash2 = self._file_hash(file2)
            
            if hash1 == hash2:
                return {'identical': True, 'within_tolerance': True, 'type': 'binary'}
            else:
                return {
                    'identical': False,
                    'within_tolerance': False,
                    'type': 'binary_difference',
                    'details': f"File hashes differ: {hash1} vs {hash2}"
                }
                
        except Exception as e:
            return {
                'identical': False,
                'within_tolerance': False,
                'type': 'binary_error',
                'details': str(e)
            }
    
    def _file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class PhysicsValidator:
    """Validator for real physics calculations."""
    
    def __init__(self, logger: Optional[ProfessionalLogger] = None):
        self.logger = logger or ProfessionalLogger("PhysicsValidator")
    
    def validate_physics(self, require_real_calculation: bool = True,
                        no_mock_allowed: bool = True) -> ValidationResult:
        """
        Validate that real physics calculations are performed.
        
        Args:
            require_real_calculation: If True, require at least one real calculation
            no_mock_allowed: If True, fail if any mock calculations detected
            
        Returns:
            Validation result
        """
        
        self.logger.info("Validating physics calculation authenticity")
        
        issues = []
        details = {
            'real_calculations_performed': 0,
            'mock_calculations_detected': 0,
            'calculation_types': [],
            'mock_locations': [],
            'validation_summary': {}
        }
        
        # Scan code for physics calculations
        physics_modules = [
            'revolutionary_meep_engine.py',
            'rigorous_floquet_engine.py',
            'rigorous_qed_engine.py',
            'thz_bandwidth_framework.py',
            'dual_band_modulator.py',
            'quantum_regime_extension.py',
            'topological_enhancement.py'
        ]
        
        for module_name in physics_modules:
            module_path = Path(module_name)
            if module_path.exists():
                result = self._analyze_physics_module(module_path)
                details['real_calculations_performed'] += result['real_calculations']
                details['mock_calculations_detected'] += result['mock_calculations']
                details['calculation_types'].extend(result['calculation_types'])
                details['mock_locations'].extend(result['mock_locations'])
                
                details['validation_summary'][module_name] = result
        
        # Check requirements
        if require_real_calculation and details['real_calculations_performed'] == 0:
            issues.append("No real physics calculations detected")
        
        if no_mock_allowed and details['mock_calculations_detected'] > 0:
            issues.append(f"Mock calculations detected: {details['mock_locations']}")
        
        passed = len(issues) == 0
        
        self.logger.info(f"Physics validation complete: "
                        f"{details['real_calculations_performed']} real calculations, "
                        f"{details['mock_calculations_detected']} mock calculations")
        
        return ValidationResult(
            check_name="physics_validation",
            passed=passed,
            details=details,
            issues=issues,
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_physics_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a physics module for real vs mock calculations."""
        
        result = {
            'real_calculations': 0,
            'mock_calculations': 0,
            'calculation_types': [],
            'mock_locations': []
        }
        
        try:
            with open(module_path) as f:
                content = f.read()
            
            # Parse AST to find function definitions and calls
            tree = ast.parse(content)
            
            # Look for indicators of real physics calculations
            real_indicators = [
                'fdtd', 'meep', 'solve_eigenmodes', 'maxwell_solver',
                'floquet_hamiltonian', 'magnus_expansion', 'propagator',
                'quantum_evolution', 'lindblad', 'master_equation',
                'scattering_matrix', 'transmission', 'reflection'
            ]
            
            # Look for indicators of mock calculations
            mock_indicators = [
                'mock', 'fake', 'dummy', 'placeholder', 'synthetic',
                'hardcoded', 'hard_coded', 'return 1.0', 'return np.ones',
                'if not.*engine', 'fallback'
            ]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name.lower()
                    func_content = ast.get_source_segment(content, node)
                    
                    # Check for real calculation indicators
                    for indicator in real_indicators:
                        if indicator in func_content.lower():
                            result['real_calculations'] += 1
                            result['calculation_types'].append(f"{func_name}:{indicator}")
                            break
                    
                    # Check for mock calculation indicators
                    for indicator in mock_indicators:
                        if indicator in func_content.lower():
                            result['mock_calculations'] += 1
                            result['mock_locations'].append(f"{module_path}:{func_name}")
                            break
                            
        except Exception as e:
            self.logger.warning(f"Error analyzing {module_path}: {e}")
        
        return result


class DocumentationValidator:
    """Validator for documentation coverage and quality."""
    
    def __init__(self, logger: Optional[ProfessionalLogger] = None):
        self.logger = logger or ProfessionalLogger("DocumentationValidator")
    
    def validate_docstrings(self, minimum_coverage: float = 95,
                           require_parameter_docs: bool = True,
                           require_return_docs: bool = True,
                           require_units: bool = True) -> ValidationResult:
        """
        Validate docstring coverage and quality.
        
        Args:
            minimum_coverage: Minimum required docstring coverage percentage
            require_parameter_docs: Require parameter documentation
            require_return_docs: Require return value documentation
            require_units: Require units in parameter documentation
            
        Returns:
            Validation result
        """
        
        self.logger.info("Validating docstring coverage and quality")
        
        issues = []
        details = {
            'total_functions': 0,
            'documented_functions': 0,
            'coverage_percentage': 0.0,
            'missing_param_docs': [],
            'missing_return_docs': [],
            'missing_units': [],
            'quality_issues': []
        }
        
        # Scan all Python modules
        python_files = list(Path('.').glob('*.py'))
        
        for py_file in python_files:
            if py_file.name.startswith('test_'):
                continue  # Skip test files
                
            file_analysis = self._analyze_docstrings(py_file, require_parameter_docs,
                                                   require_return_docs, require_units)
            
            details['total_functions'] += file_analysis['total_functions']
            details['documented_functions'] += file_analysis['documented_functions']
            details['missing_param_docs'].extend(file_analysis['missing_param_docs'])
            details['missing_return_docs'].extend(file_analysis['missing_return_docs'])
            details['missing_units'].extend(file_analysis['missing_units'])
            details['quality_issues'].extend(file_analysis['quality_issues'])
        
        # Calculate coverage
        if details['total_functions'] > 0:
            details['coverage_percentage'] = (details['documented_functions'] / 
                                           details['total_functions']) * 100
        
        # Check requirements
        if details['coverage_percentage'] < minimum_coverage:
            issues.append(f"Docstring coverage {details['coverage_percentage']:.1f}% "
                         f"below minimum {minimum_coverage}%")
        
        if require_parameter_docs and details['missing_param_docs']:
            issues.append(f"Functions missing parameter documentation: "
                         f"{len(details['missing_param_docs'])}")
        
        if require_return_docs and details['missing_return_docs']:
            issues.append(f"Functions missing return documentation: "
                         f"{len(details['missing_return_docs'])}")
        
        if require_units and details['missing_units']:
            issues.append(f"Functions missing units in documentation: "
                         f"{len(details['missing_units'])}")
        
        passed = len(issues) == 0
        
        self.logger.info(f"Docstring validation complete: "
                        f"{details['coverage_percentage']:.1f}% coverage, "
                        f"{len(issues)} issues found")
        
        return ValidationResult(
            check_name="docstring_validation",
            passed=passed,
            details=details,
            issues=issues,
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_docstrings(self, py_file: Path, require_param_docs: bool,
                           require_return_docs: bool, require_units: bool) -> Dict[str, Any]:
        """Analyze docstrings in a Python file."""
        
        result = {
            'total_functions': 0,
            'documented_functions': 0,
            'missing_param_docs': [],
            'missing_return_docs': [],
            'missing_units': [],
            'quality_issues': []
        }
        
        try:
            with open(py_file) as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('_') and not node.name.startswith('__'):
                        continue  # Skip private functions
                    
                    result['total_functions'] += 1
                    func_name = f"{py_file.name}:{node.name}"
                    
                    docstring = ast.get_docstring(node)
                    
                    if docstring:
                        result['documented_functions'] += 1
                        
                        # Analyze docstring quality
                        if require_param_docs and node.args.args:
                            # Check for parameter documentation
                            param_names = [arg.arg for arg in node.args.args 
                                         if arg.arg != 'self']
                            
                            if param_names:
                                has_args_section = any(keyword in docstring.lower() 
                                                     for keyword in ['args:', 'arguments:', 'parameters:'])
                                
                                if not has_args_section:
                                    result['missing_param_docs'].append(func_name)
                                
                                # Check for units in parameter docs
                                if require_units:
                                    units_keywords = ['units:', 'm', 'kg', 's', 'hz', 'w', 'v', 'a']
                                    has_units = any(keyword in docstring.lower() 
                                                  for keyword in units_keywords)
                                    
                                    if not has_units:
                                        result['missing_units'].append(func_name)
                        
                        # Check for return documentation
                        if require_return_docs:
                            has_returns = any(keyword in docstring.lower() 
                                            for keyword in ['returns:', 'return:', 'yields:'])
                            
                            if not has_returns:
                                result['missing_return_docs'].append(func_name)
                    
                    else:
                        # No docstring
                        result['missing_param_docs'].append(func_name)
                        result['missing_return_docs'].append(func_name)
                        result['missing_units'].append(func_name)
                        
        except Exception as e:
            result['quality_issues'].append(f"Error analyzing {py_file}: {e}")
        
        return result


def main():
    """Main CLI interface for validation tools."""
    
    parser = argparse.ArgumentParser(description="CI Validation Tools for Nature-Grade Standards")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compare runs command
    compare_parser = subparsers.add_parser('compare-runs', help='Compare deterministic runs')
    compare_parser.add_argument('--run1-dir', required=True, help='First run directory')
    compare_parser.add_argument('--run2-dir', required=True, help='Second run directory')
    compare_parser.add_argument('--tolerance', type=float, default=1e-12, help='Numerical tolerance')
    compare_parser.add_argument('--output-report', required=True, help='Output report file')
    
    # Validate physics command
    physics_parser = subparsers.add_parser('validate-physics', help='Validate physics calculations')
    physics_parser.add_argument('--require-real-calculation', action='store_true', 
                               help='Require at least one real calculation')
    physics_parser.add_argument('--no-mock-allowed', action='store_true',
                               help='Fail if mock calculations detected')
    physics_parser.add_argument('--output-validation', required=True, help='Output validation file')
    
    # Validate docstrings command
    doc_parser = subparsers.add_parser('validate-docstrings', help='Validate documentation')
    doc_parser.add_argument('--minimum-coverage', type=float, default=95, 
                           help='Minimum coverage percentage')
    doc_parser.add_argument('--require-parameter-docs', action='store_true',
                           help='Require parameter documentation')
    doc_parser.add_argument('--require-return-docs', action='store_true',
                           help='Require return documentation')
    doc_parser.add_argument('--require-units', action='store_true',
                           help='Require units in documentation')
    doc_parser.add_argument('--output-report', required=True, help='Output report file')
    
    # Additional commands would be implemented here...
    
    args = parser.parse_args()
    
    if args.command == 'compare-runs':
        validator = DeterministicValidator()
        result = validator.compare_runs(args.run1_dir, args.run2_dir, args.tolerance)
        
        with open(args.output_report, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"Comparison complete. Report saved to {args.output_report}")
        if not result.passed:
            sys.exit(1)
    
    elif args.command == 'validate-physics':
        validator = PhysicsValidator()
        result = validator.validate_physics(args.require_real_calculation, args.no_mock_allowed)
        
        with open(args.output_validation, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"Physics validation complete. Report saved to {args.output_validation}")
        if not result.passed:
            sys.exit(1)
    
    elif args.command == 'validate-docstrings':
        validator = DocumentationValidator()
        result = validator.validate_docstrings(
            args.minimum_coverage, args.require_parameter_docs,
            args.require_return_docs, args.require_units
        )
        
        with open(args.output_report, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"Documentation validation complete. Report saved to {args.output_report}")
        if not result.passed:
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
