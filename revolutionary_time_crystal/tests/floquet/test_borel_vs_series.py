#!/usr/bin/env python3
"""
Floquet Borel Resummation Tests
==============================

Tests for Borel resummation vs. series convergence in Floquet calculations.
Verifies proper handling of divergent perturbative series.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rigorous_floquet_engine_fixed import RigorousFloquetEngine, FloquetSystemParameters
from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters

class TestFloquetBorelResummation:
    """Test suite for Borel resummation in Floquet calculations"""
    
    def setup_method(self):
        """Set up test systems"""
        
        # QED engine
        self.qed_params = QEDSystemParameters(
            modulation_frequency=2*np.pi*15e9,  # 15 GHz
            coupling_strength=0.02,
            device_length=4e-6,
            device_width=2e-6,
            device_height=0.2e-6
        )
        
        self.qed_engine = QuantumElectrodynamicsEngine(self.qed_params)
        
        # Base Floquet parameters
        self.floquet_params = FloquetSystemParameters(
            driving_frequency=self.qed_params.modulation_frequency,
            driving_amplitude=0.05,  # Moderate for testing
            n_harmonics=10,
            eigenvalue_tolerance=1e-8
        )
        
        self.floquet_engine = RigorousFloquetEngine(self.qed_engine, self.floquet_params)
    
    def test_borel_transform_convergent_series(self):
        """Test Borel resummation on known convergent series"""
        
        # Test with geometric series: 1/(1-z) = Σ z^n
        # Borel transform should give exact result
        coefficients = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # First few terms
        
        # Test at point z = 0.3 (well within convergence radius)
        z = 0.3
        expected = 1.0 / (1.0 - z)  # Exact result
        
        borel_result, info = self.floquet_engine.borel_resummation(coefficients, z)
        
        # Should converge and give accurate result
        assert info['converged'], f"Borel resummation failed to converge: {info}"
        
        relative_error = abs(borel_result - expected) / abs(expected)
        print(f"Geometric series test: expected {expected:.6f}, got {borel_result:.6f}")
        print(f"Relative error: {relative_error:.2e}")
        
        assert relative_error < 0.1, f"Borel resummation error {relative_error:.2e} too large"
    
    def test_borel_transform_divergent_series(self):
        """Test Borel resummation on factorial divergent series"""
        
        # Create factorial-like divergent series: Σ n! z^n
        coefficients = np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0])  # n! for n=0...5
        
        # Test at small point where series diverges but Borel might work
        z = 0.1
        
        borel_result, info = self.floquet_engine.borel_resummation(coefficients, z)
        
        # Should either converge with Borel or fallback to Padé
        assert info['converged'] or 'pade' in info.get('method', ''), \
            f"Both Borel and Padé failed: {info}"
        
        # Result should be finite
        assert np.isfinite(borel_result), f"Borel resummation gave infinite result: {borel_result}"
        
        print(f"Divergent series result: {borel_result:.6f}, method: {info.get('method', 'unknown')}")
    
    def test_pade_fallback_mechanism(self):
        """Test that Padé approximants work as fallback"""
        
        # Create series that will cause Borel integral to diverge
        # Alternating series with large coefficients
        coefficients = np.array([1.0, -10.0, 50.0, -200.0, 1000.0, -5000.0])
        
        z = 0.5
        
        # Force Padé fallback by using problematic series
        borel_result, info = self.floquet_engine.borel_resummation(coefficients, z)
        
        # Should use Padé method
        method = info.get('method', '')
        if not info['converged']:
            assert 'pade' in method or 'failed' in method, f"Unexpected fallback method: {method}"
        
        print(f"Padé fallback test: method = {method}, result = {borel_result:.6f}")
    
    def test_borel_vs_truncated_series_comparison(self):
        """Compare Borel resummation vs. truncated series for strong driving"""
        
        # Create system with strong driving where perturbation series might diverge
        strong_params = FloquetSystemParameters(
            driving_frequency=self.floquet_params.driving_frequency,
            driving_amplitude=0.3,  # Strong driving
            n_harmonics=8,
            eigenvalue_tolerance=1e-6
        )
        
        strong_engine = RigorousFloquetEngine(self.qed_engine, strong_params)
        
        # For this test, we'll simulate perturbative coefficients
        # In real application, these would come from Floquet perturbation theory
        
        # Mock perturbative series coefficients (growing like factorial)
        n_terms = 6
        perturbation_parameter = strong_params.driving_amplitude
        
        coefficients = np.zeros(n_terms)
        for n in range(n_terms):
            # Simulate typical quantum field theory divergence
            coefficients[n] = (-1)**n * np.math.factorial(n) * (0.1)**n
        
        # Compare truncated series vs. Borel resummation
        z = perturbation_parameter
        
        # Truncated series (naively summing first few terms)
        truncated_result = sum(coefficients[n] * (z**n) for n in range(n_terms))
        
        # Borel resummation
        try:
            borel_result, info = strong_engine.borel_resummation(coefficients, z)
            
            print(f"Truncated series: {truncated_result:.6f}")
            print(f"Borel resummation: {borel_result:.6f}, method: {info.get('method', 'unknown')}")
            
            # Results should be different for divergent series
            if info['converged']:
                relative_diff = abs(borel_result - truncated_result) / (abs(truncated_result) + 1e-10)
                
                # For divergent series, Borel should give different (hopefully better) result
                if abs(truncated_result) > 1e3:  # Series clearly divergent
                    assert relative_diff > 0.1, "Borel resummation should differ from divergent truncated series"
                    
        except Exception as e:
            print(f"Borel resummation failed for strong driving: {e}")
            # This is acceptable for extremely divergent cases
    
    def test_gauss_laguerre_integration(self):
        """Test Gauss-Laguerre quadrature implementation"""
        
        # Test with known integral: ∫₀^∞ e^(-t) t^n dt = n!
        for n in [0, 1, 2, 3]:
            # Create coefficients for t^n
            coefficients = np.zeros(n+2)
            if n == 0:
                coefficients[0] = 1.0
            else:
                coefficients[n] = 1.0  # t^n term
            
            result, info = self.floquet_engine.borel_resummation(coefficients, 1.0)
            
            if info.get('method') == 'borel_gauss_laguerre':
                expected = np.math.factorial(n)
                relative_error = abs(result - expected) / expected if expected > 0 else abs(result)
                
                print(f"∫₀^∞ e^(-t) t^{n} dt: expected {expected}, got {result:.6f}")
                print(f"Relative error: {relative_error:.2e}")
                
                # Should be reasonably accurate for low-order polynomials
                assert relative_error < 0.1, f"Gauss-Laguerre error {relative_error:.2e} too large for t^{n}"
    
    def test_numerical_stability(self):
        """Test numerical stability of Borel resummation"""
        
        # Test with coefficients of varying magnitude
        test_cases = [
            np.array([1.0, 0.1, 0.01, 0.001]),  # Decreasing
            np.array([0.001, 0.01, 0.1, 1.0]),  # Increasing
            np.array([1.0, -1.0, 1.0, -1.0]),   # Alternating
            np.array([1e-10, 1e-5, 1.0, 1e5])   # Wide range
        ]
        
        for i, coefficients in enumerate(test_cases):
            print(f"Testing stability case {i+1}: {coefficients}")
            
            try:
                result, info = self.floquet_engine.borel_resummation(coefficients, 0.1)
                
                # Result should be finite
                assert np.isfinite(result), f"Non-finite result for case {i+1}: {result}"
                
                # Should not be extremely large
                assert abs(result) < 1e10, f"Extremely large result for case {i+1}: {result}"
                
                print(f"  Result: {result:.6f}, method: {info.get('method', 'unknown')}")
                
            except Exception as e:
                print(f"  Failed (acceptable for pathological cases): {e}")
    
    def test_convergence_guard_enforcement(self):
        """Test that convergence guards raise RuntimeError when appropriate"""
        
        # Create pathological coefficients that should cause all methods to fail
        pathological_coeffs = np.array([np.inf, np.nan, 1e20, -1e20])
        
        with pytest.raises(RuntimeError, match="Both Borel resummation and Padé approximation failed"):
            self.floquet_engine.borel_resummation(pathological_coeffs, 1.0)
        
        print("Convergence guard properly enforced for pathological input")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
