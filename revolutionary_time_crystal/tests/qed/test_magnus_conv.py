#!/usr/bin/env python3
"""
Magnus Expansion Convergence Tests
=================================

Tests for Magnus expansion convergence with higher-order terms Ω₄...Ω₆.
Verifies proper convergence criteria and failure handling.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters

class TestMagnusConvergence:
    """Test suite for Magnus expansion convergence"""
    
    def setup_method(self):
        """Set up test QED system"""
        self.qed_params = QEDSystemParameters(
            modulation_frequency=2*np.pi*5e9,  # 5 GHz - moderate frequency
            coupling_strength=0.001,  # Very weak coupling for convergence
            device_length=5e-6,
            device_width=3e-6,
            device_height=0.3e-6,
            chi_1=0.01,  # Small modulation
            chi_2=0.005
        )
        
        self.engine = QuantumElectrodynamicsEngine(self.qed_params)
    
    def test_magnus_convergence_weak_coupling(self):
        """Test Magnus expansion converges for weak coupling"""
        
        # Create minimal test system
        spatial_grid = self._create_minimal_spatial_grid()
        
        # Short time period for convergence
        T_period = 2 * np.pi / self.qed_params.modulation_frequency
        time_points = np.linspace(0, T_period, 16)
        
        # Calculate interaction Hamiltonian
        H_int = self.engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
        
        # Test time evolution with Magnus expansion
        U_magnus, convergence_info = self.engine.time_evolution_operator(H_int, time_points)
        
        # Should converge for weak coupling
        assert convergence_info['converged'], f"Magnus expansion failed to converge: {convergence_info['reason']}"
        
        # Check unitarity of evolution operator
        U_dagger_U = U_magnus.conj().T @ U_magnus
        identity_error = np.linalg.norm(U_dagger_U - np.eye(U_magnus.shape[0]))
        
        print(f"Unitarity error: {identity_error:.2e}")
        assert identity_error < 1e-10, f"Evolution operator not unitary: error = {identity_error:.2e}"
    
    def test_magnus_convergence_failure_strong_coupling(self):
        """Test Magnus expansion properly detects divergence for strong coupling"""
        
        # Create system with strong coupling that should cause divergence
        strong_params = QEDSystemParameters(
            modulation_frequency=2*np.pi*20e9,  # High frequency
            coupling_strength=0.5,  # Strong coupling
            device_length=self.qed_params.device_length,
            device_width=self.qed_params.device_width,
            device_height=self.qed_params.device_height,
            chi_1=0.5,  # Large modulation
            chi_2=0.3
        )
        
        strong_engine = QuantumElectrodynamicsEngine(strong_params)
        
        spatial_grid = self._create_minimal_spatial_grid()
        T_period = 2 * np.pi / strong_params.modulation_frequency
        time_points = np.linspace(0, T_period, 16)
        
        H_int = strong_engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
        
        # Should detect convergence failure and use Borel resummation
        with pytest.warns(UserWarning, match="Magnus expansion may not converge"):
            U_magnus, convergence_info = strong_engine.time_evolution_operator(H_int, time_points)
        
        # Should mark as not converged
        assert not convergence_info['converged'], "Should detect convergence failure for strong coupling"
        
        # But should still produce finite result via Borel resummation
        assert np.all(np.isfinite(U_magnus)), "Evolution operator should be finite even with Borel resummation"
    
    def test_higher_order_magnus_terms(self):
        """Test calculation of higher-order Magnus terms Ω₄...Ω₆"""
        
        spatial_grid = self._create_minimal_spatial_grid()
        T_period = 2 * np.pi / self.qed_params.modulation_frequency
        time_points = np.linspace(0, T_period, 12)  # Fewer points for efficiency
        
        H_int = self.engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
        
        # Calculate higher-order Magnus terms
        magnus_terms, converged = self.engine._calculate_higher_order_magnus(H_int, time_points, max_order=6)
        
        # Should have at least Ω₁, Ω₂, Ω₃
        assert len(magnus_terms) >= 3, f"Expected at least 3 Magnus terms, got {len(magnus_terms)}"
        
        # Terms should generally decrease in magnitude (convergence criterion)
        norms = [np.linalg.norm(omega) for omega in magnus_terms]
        print(f"Magnus term norms: {[f'{norm:.2e}' for norm in norms]}")
        
        # At least the first few terms should show decreasing pattern
        if len(norms) >= 3:
            # Allow some flexibility, but generally should decrease
            decreasing_count = sum(norms[i] > norms[i+1] for i in range(len(norms)-1))
            total_pairs = len(norms) - 1
            
            # At least half should be decreasing
            assert decreasing_count >= total_pairs // 2, f"Magnus terms not showing convergence pattern"
    
    def test_magnus_term_scaling(self):
        """Test that Magnus terms scale correctly with system parameters"""
        
        # Test different time windows
        T_short = 1e-12  # Very short time
        T_long = 2 * np.pi / self.qed_params.modulation_frequency  # Full period
        
        spatial_grid = self._create_minimal_spatial_grid()
        
        for T, label in [(T_short, "short"), (T_long, "long")]:
            time_points = np.linspace(0, T, 12)
            H_int = self.engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
            
            # Calculate first few Magnus terms
            Omega_1 = self.engine._calculate_omega_1(H_int, time_points)
            Omega_2 = self.engine._calculate_omega_2(H_int, time_points)
            Omega_3 = self.engine._calculate_omega_3(H_int, time_points)
            
            norm_1 = np.linalg.norm(Omega_1)
            norm_2 = np.linalg.norm(Omega_2)
            norm_3 = np.linalg.norm(Omega_3)
            
            print(f"{label} time: ||Ω₁|| = {norm_1:.2e}, ||Ω₂|| = {norm_2:.2e}, ||Ω₃|| = {norm_3:.2e}")
            
            # For short time, all terms should be small
            if T == T_short:
                assert norm_1 < 1e-6, f"Ω₁ too large for short time: {norm_1:.2e}"
                assert norm_2 < 1e-9, f"Ω₂ too large for short time: {norm_2:.2e}"
                assert norm_3 < 1e-12, f"Ω₃ too large for short time: {norm_3:.2e}"
    
    def test_commutator_calculation(self):
        """Test double and triple commutator calculations"""
        
        # Create test matrices
        n = 4
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        B = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        C = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        
        # Test double commutator [A, [B, C]]
        double_comm = self.engine._double_commutator(A, B, C)
        
        # Manual calculation
        inner = B @ C - C @ B
        expected_double = A @ inner - inner @ A
        
        assert np.allclose(double_comm, expected_double), "Double commutator calculation incorrect"
        
        # Test anti-symmetry property: [A, [B, C]] = -[A, [C, B]]
        double_comm_reversed = self.engine._double_commutator(A, C, B)
        assert np.allclose(double_comm, -double_comm_reversed), "Double commutator anti-symmetry violated"
    
    def test_magnus_norm_criterion(self):
        """Test Magnus convergence norm criterion ||∫H(t)dt|| < π"""
        
        spatial_grid = self._create_minimal_spatial_grid()
        T_period = 2 * np.pi / self.qed_params.modulation_frequency
        time_points = np.linspace(0, T_period, 16)
        
        H_int = self.engine.interaction_hamiltonian_matrix(spatial_grid, time_points)
        
        # Calculate Ω₁ = ∫H(t)dt
        Omega_1 = self.engine._calculate_omega_1(H_int, time_points)
        norm_Omega_1 = np.linalg.norm(Omega_1)
        
        print(f"||Ω₁|| = {norm_Omega_1:.2e} (convergence requires < π = {np.pi:.2e})")
        
        # For weak coupling, should satisfy convergence criterion
        assert norm_Omega_1 < np.pi, f"Magnus norm criterion violated: ||Ω₁|| = {norm_Omega_1:.2e} ≥ π"
    
    def _create_minimal_spatial_grid(self) -> np.ndarray:
        """Create minimal spatial grid for testing"""
        
        # Very small grid for computational efficiency
        N_x, N_y, N_z = 4, 4, 2
        
        x = np.linspace(0, self.qed_params.device_length, N_x)
        y = np.linspace(0, self.qed_params.device_width, N_y)
        z = np.linspace(0, self.qed_params.device_height, N_z)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        spatial_grid = np.stack([X, Y, Z], axis=-1)
        
        return spatial_grid

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
