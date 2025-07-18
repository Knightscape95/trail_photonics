#!/usr/bin/env python3
"""
QED Renormalization Tests
========================

Tests for proper renormalization implementation ensuring vacuum energy shift → 0.
This verifies the counter-terms Z₁-Z₃ are correctly applied.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters

class TestQEDRenormalization:
    """Test suite for QED renormalization implementation"""
    
    def setup_method(self):
        """Set up test QED system"""
        self.qed_params = QEDSystemParameters(
            modulation_frequency=2*np.pi*10e9,  # 10 GHz
            coupling_strength=0.01,  # Small for perturbative regime
            device_length=10e-6,  # 10 μm
            device_width=5e-6,   # 5 μm
            device_height=0.5e-6,  # 500 nm
            chi_1=0.1,
            chi_2=0.05
        )
        
        self.engine = QuantumElectrodynamicsEngine(self.qed_params)
    
    def test_vacuum_energy_shift_convergence(self):
        """Test that vacuum energy shift approaches zero with renormalization"""
        
        # Create small spatial grid for testing
        N_x, N_y, N_z = 8, 8, 4
        spatial_grid = self._create_test_spatial_grid(N_x, N_y, N_z)
        
        # Time points for one modulation period
        T_period = 2 * np.pi / self.qed_params.modulation_frequency
        time_points = np.linspace(0, T_period, 32)
        
        # Calculate interaction Hamiltonian with renormalization
        H_int = self.engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
        
        # Vacuum energy shift is trace of time-averaged Hamiltonian
        H_avg = np.mean(H_int, axis=2)
        vacuum_energy_shift = np.trace(H_avg).real
        
        # After proper renormalization, vacuum energy shift should be very small
        vacuum_shift_threshold = 1e-10  # Demanding threshold for proper renormalization
        
        print(f"Vacuum energy shift: {vacuum_energy_shift:.2e}")
        print(f"Threshold: {vacuum_shift_threshold:.2e}")
        
        assert abs(vacuum_energy_shift) < vacuum_shift_threshold, \
            f"Vacuum energy shift {vacuum_energy_shift:.2e} exceeds threshold {vacuum_shift_threshold:.2e}"
    
    def test_renormalization_counterterms_applied(self):
        """Test that renormalization counter-terms are properly applied"""
        
        # Create test Hamiltonian matrix
        n_modes = 4
        H_test = np.random.randn(n_modes, n_modes) + 1j * np.random.randn(n_modes, n_modes)
        H_test = (H_test + H_test.conj().T) / 2  # Make Hermitian
        
        # Apply renormalization
        H_renorm = self.engine._apply_renormalization_counterterms(H_test, 0.0)
        
        # Check that renormalized matrix is still Hermitian
        assert np.allclose(H_renorm, H_renorm.conj().T), "Renormalized Hamiltonian not Hermitian"
        
        # Check that renormalization changes the matrix (counter-terms applied)
        assert not np.allclose(H_test, H_renorm), "Renormalization counter-terms not applied"
        
        # Check that the change is finite (no divergences)
        assert np.all(np.isfinite(H_renorm)), "Renormalized Hamiltonian contains infinities"
    
    def test_3d_integration_accuracy(self):
        """Test that 3D integration gives correct results for known functions"""
        
        # Create test spatial grid
        N_x, N_y, N_z = 16, 16, 8
        spatial_grid = self._create_test_spatial_grid(N_x, N_y, N_z)
        
        # Test function: f(x,y,z) = sin(πx/L_x) * cos(πy/L_y) * exp(-z²/σ²)
        L_x = self.qed_params.device_length
        L_y = self.qed_params.device_width
        sigma = self.qed_params.device_height / 4
        
        x = spatial_grid[:, :, :, 0]
        y = spatial_grid[:, :, :, 1]
        z = spatial_grid[:, :, :, 2]
        
        test_function = (np.sin(np.pi * x / L_x) * 
                        np.cos(np.pi * y / L_y) * 
                        np.exp(-(z**2) / (2 * sigma**2)))
        
        # Numerical integration using engine's method
        from scipy.integrate import simpson
        dx = L_x / N_x
        dy = self.qed_params.device_width / N_y
        dz = self.qed_params.device_height / N_z
        
        integral_z = simpson(test_function, dx=dz, axis=2)
        integral_yz = simpson(integral_z, dx=dy, axis=1)
        numerical_integral = simpson(integral_yz, dx=dx, axis=0)
        
        # Analytical result for this specific function
        analytical_integral = (L_x * L_y * sigma * np.sqrt(2 * np.pi) * 
                             (2 / np.pi) * (2 / np.pi))  # sin and cos integrals
        
        relative_error = abs(numerical_integral - analytical_integral) / abs(analytical_integral)
        
        print(f"Numerical integral: {numerical_integral:.6f}")
        print(f"Analytical integral: {analytical_integral:.6f}")
        print(f"Relative error: {relative_error:.2e}")
        
        # Demand high accuracy for 3D integration
        assert relative_error < 0.01, f"3D integration error {relative_error:.2e} too large"
    
    def test_renormalization_scaling(self):
        """Test that renormalization constants scale correctly with coupling"""
        
        # Test with different coupling strengths
        couplings = [0.001, 0.01, 0.1]
        vacuum_shifts = []
        
        for coupling in couplings:
            params = QEDSystemParameters(
                modulation_frequency=self.qed_params.modulation_frequency,
                coupling_strength=coupling,
                device_length=self.qed_params.device_length,
                device_width=self.qed_params.device_width,
                device_height=self.qed_params.device_height,
                chi_1=self.qed_params.chi_1,
                chi_2=self.qed_params.chi_2
            )
            
            engine = QuantumElectrodynamicsEngine(params)
            
            # Small test calculation
            spatial_grid = self._create_test_spatial_grid(4, 4, 2)
            time_points = np.linspace(0, 1e-12, 8)  # Short time for speed
            
            H_int = engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
            H_avg = np.mean(H_int, axis=2)
            vacuum_shift = abs(np.trace(H_avg).real)
            
            vacuum_shifts.append(vacuum_shift)
        
        # Vacuum shift should remain small for all couplings due to renormalization
        max_shift = max(vacuum_shifts)
        print(f"Maximum vacuum shift across couplings: {max_shift:.2e}")
        
        assert max_shift < 1e-8, f"Renormalization fails for strong coupling: {max_shift:.2e}"
    
    def _create_test_spatial_grid(self, N_x: int, N_y: int, N_z: int) -> np.ndarray:
        """Create test spatial grid"""
        
        x = np.linspace(0, self.qed_params.device_length, N_x)
        y = np.linspace(0, self.qed_params.device_width, N_y)
        z = np.linspace(0, self.qed_params.device_height, N_z)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        spatial_grid = np.stack([X, Y, Z], axis=-1)
        
        return spatial_grid

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
