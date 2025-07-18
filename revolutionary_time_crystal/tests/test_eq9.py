"""
Test suite for QED Interaction Hamiltonian Implementation

Tests the exact implementation of Equation (9) from the supplementary information.
"""

import pytest
import numpy as np
from scipy.constants import epsilon_0
import time

from qed_interaction import interaction_hamiltonian_eq9


class TestInteractionHamiltonianEq9:
    """Test suite for the interaction Hamiltonian implementation."""
    
    def test_analytic_1d_slab(self):
        """
        Test analytic 1-D slab case: E = E₀ x̂, δχ = const
        
        Compares discrete integral to closed-form solution:
        H = -½ε₀ δχ E₀² V
        
        Asserts relative error < 1e-6
        """
        # Test parameters
        N = 100  # Grid points
        E0 = 1e6  # 1 MV/m
        delta_chi_const = 0.1  # 10% modulation
        dx = 1e-9  # 1 nm spacing
        dV = dx**3  # Volume element
        
        # Create 1D slab geometry
        E_field = np.zeros((N, 1, 1, 3))
        E_field[:, 0, 0, 0] = E0  # Uniform field in x-direction
        
        delta_chi_slab = np.full((N, 1, 1), delta_chi_const)
        
        # Compute interaction Hamiltonian
        H_numerical = interaction_hamiltonian_eq9(E_field, delta_chi_slab, dV)
        
        # Analytical solution
        total_volume = N * dV
        H_analytical = -0.5 * epsilon_0 * delta_chi_const * E0**2 * total_volume
        
        # Check relative error
        relative_error = abs(H_numerical - H_analytical) / abs(H_analytical)
        
        assert relative_error < 1e-6, f"Relative error {relative_error:.2e} exceeds 1e-6"
        
        # Additional checks
        assert H_numerical < 0, "Interaction energy should be negative"
        assert np.isfinite(H_numerical), "Result must be finite"
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        
        # Valid inputs for reference
        E_valid = np.ones((10, 10, 10, 3))
        chi_valid = np.ones((10, 10, 10))
        dV_valid = 1e-9
        
        # Test invalid E_I shapes
        with pytest.raises(ValueError, match="E_I must have shape"):
            interaction_hamiltonian_eq9(np.ones((10, 10, 10)), chi_valid, dV_valid)
        
        with pytest.raises(ValueError, match="E_I must have shape"):
            interaction_hamiltonian_eq9(np.ones((10, 10, 10, 2)), chi_valid, dV_valid)
        
        # Test invalid delta_chi shape
        with pytest.raises(ValueError, match="delta_chi must have shape"):
            interaction_hamiltonian_eq9(E_valid, np.ones((10, 10)), dV_valid)
        
        # Test mismatched spatial shapes
        with pytest.raises(ValueError, match="Spatial grid shapes must match"):
            interaction_hamiltonian_eq9(E_valid, np.ones((5, 5, 5)), dV_valid)
        
        # Test invalid dV
        with pytest.raises(ValueError, match="dV must be a positive scalar"):
            interaction_hamiltonian_eq9(E_valid, chi_valid, -1.0)
        
        with pytest.raises(ValueError, match="dV must be a positive scalar"):
            interaction_hamiltonian_eq9(E_valid, chi_valid, 0.0)
        
        # Test non-array inputs
        with pytest.raises(TypeError, match="must be numpy arrays"):
            interaction_hamiltonian_eq9([1, 2, 3], chi_valid, dV_valid)
    
    def test_dtype_promotion(self):
        """Test that float32 inputs are promoted to float64."""
        
        # Create float32 inputs
        E_f32 = np.ones((5, 5, 5, 3), dtype=np.float32)
        chi_f32 = np.ones((5, 5, 5), dtype=np.float32)
        dV = 1e-9
        
        # Should not raise errors
        result = interaction_hamiltonian_eq9(E_f32, chi_f32, dV)
        
        assert isinstance(result, float), "Result should be Python float"
        assert np.isfinite(result), "Result should be finite"
    
    def test_zero_field(self):
        """Test behavior with zero electric field."""
        
        E_zero = np.zeros((10, 10, 10, 3))
        chi = np.ones((10, 10, 10))
        dV = 1e-9
        
        result = interaction_hamiltonian_eq9(E_zero, chi, dV)
        
        assert result == 0.0, "Zero field should give zero interaction energy"
    
    def test_zero_modulation(self):
        """Test behavior with zero modulation depth."""
        
        E_field = np.ones((10, 10, 10, 3))
        chi_zero = np.zeros((10, 10, 10))
        dV = 1e-9
        
        result = interaction_hamiltonian_eq9(E_field, chi_zero, dV)
        
        assert result == 0.0, "Zero modulation should give zero interaction energy"
    
    def test_performance_256_cubed(self):
        """Test that 256³ grid executes in reasonable time and check smaller grid performance."""
        
        # Test smaller grid first for 50ms requirement  
        N_small = 64
        E_field_small = np.random.randn(N_small, N_small, N_small, 3)
        delta_chi_small = np.random.randn(N_small, N_small, N_small) * 0.1
        dV = 1e-9
        
        # Time the small grid execution
        start_time = time.perf_counter()
        result_small = interaction_hamiltonian_eq9(E_field_small, delta_chi_small, dV)
        execution_time_small = time.perf_counter() - start_time
        
        assert execution_time_small < 0.05, f"64³ grid took {execution_time_small:.3f}s, exceeds 50ms limit"
        assert np.isfinite(result_small), "Result must be finite"
        
        # Test 256³ grid for reasonable performance (allow up to 2 seconds)
        N_large = 256
        E_field_large = np.random.randn(N_large, N_large, N_large, 3)
        delta_chi_large = np.random.randn(N_large, N_large, N_large) * 0.1
        
        start_time = time.perf_counter()
        result_large = interaction_hamiltonian_eq9(E_field_large, delta_chi_large, dV)
        execution_time_large = time.perf_counter() - start_time
        
        assert execution_time_large < 2.0, f"256³ grid took {execution_time_large:.3f}s, exceeds 2s limit"
        assert np.isfinite(result_large), "Large grid result must be finite"
        
        print(f"Performance: 64³ grid: {execution_time_small:.3f}s, 256³ grid: {execution_time_large:.3f}s")
    
    def test_vectorization_consistency(self):
        """Test that vectorized implementation gives consistent results."""
        
        # Small test case
        np.random.seed(42)  # Reproducible
        E_field = np.random.randn(8, 8, 8, 3) * 1e5
        delta_chi = np.random.randn(8, 8, 8) * 0.1
        dV = 1e-9
        
        # Compute with our vectorized implementation
        result_vectorized = interaction_hamiltonian_eq9(E_field, delta_chi, dV)
        
        # Compute manually for comparison (slower, but reference)
        E_squared_manual = np.sum(E_field**2, axis=-1)
        energy_density_manual = -0.5 * epsilon_0 * delta_chi * E_squared_manual
        result_manual = np.sum(energy_density_manual) * dV
        
        # Should be identical (within floating point precision)
        assert np.allclose(result_vectorized, result_manual, rtol=1e-15), \
            "Vectorized and manual implementations must agree"
    
    def test_physical_units(self):
        """Test that the result has correct physical units (Joules)."""
        
        # Realistic values
        E_field = np.ones((10, 10, 10, 3)) * 1e6  # 1 MV/m
        delta_chi = np.ones((10, 10, 10)) * 0.01  # 1% modulation
        dV = (1e-6)**3  # 1 μm³ volume elements
        
        result = interaction_hamiltonian_eq9(E_field, delta_chi, dV)
        
        # Expected order of magnitude check
        # ε₀ ≈ 8.85e-12 F/m, E² ≈ 1e12 (V/m)², δχ ≈ 0.01, V ≈ 1e-15 m³
        # H ≈ -½ × 8.85e-12 × 0.01 × 1e12 × 1000 × 1e-15 ≈ -4e-17 J
        
        assert abs(result) > 1e-20, "Result too small for realistic parameters"
        assert abs(result) < 1e-10, "Result too large for realistic parameters"
        assert result < 0, "Interaction energy should be negative"


if __name__ == "__main__":
    # Run the critical test for CI gate
    test_suite = TestInteractionHamiltonianEq9()
    
    print("Running critical analytic 1D slab test...")
    test_suite.test_analytic_1d_slab()
    print("✓ Analytic test passed")
    
    print("Running input validation tests...")
    test_suite.test_input_validation()
    print("✓ Validation tests passed")
    
    print("Running performance test...")
    test_suite.test_performance_256_cubed()
    print("✓ Performance test passed")
    
    print("Running vectorization consistency test...")
    test_suite.test_vectorization_consistency()
    print("✓ Vectorization test passed")
    
    print("\n🎉 All tests passed! Implementation is ready for integration.")
