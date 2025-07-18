#!/usr/bin/env python3
"""
Floquet Engine Adaptive Harmonics Tests
=======================================

Tests for adaptive harmonic truncation in Floquet calculations.
Verifies convergence and proper harmonic scaling.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rigorous_floquet_engine_fixed import RigorousFloquetEngine, FloquetSystemParameters
from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters

class TestFloquetAdaptiveHarmonics:
    """Test suite for adaptive harmonic calculations"""
    
    def setup_method(self):
        """Set up test Floquet system"""
        
        # Create QED engine first
        self.qed_params = QEDSystemParameters(
            modulation_frequency=2*np.pi*10e9,  # 10 GHz
            susceptibility_amplitude=0.01,
            device_length=5e-6,
            device_width=3e-6,
            device_height=0.3e-6
        )
        
        self.qed_engine = QuantumElectrodynamicsEngine(self.qed_params)
        
        # Floquet parameters
        self.floquet_params = FloquetSystemParameters(
            driving_frequency=self.qed_params.modulation_frequency,
            driving_amplitude=0.1,  # Moderate driving
            n_harmonics=5,  # Starting point for adaptive calculation
            eigenvalue_tolerance=1e-8,
            norm_condition_threshold=np.pi
        )
        
        self.floquet_engine = RigorousFloquetEngine(self.qed_engine, self.floquet_params)
    
    def test_adaptive_harmonic_convergence(self):
        """Test that adaptive harmonic calculation converges"""
        
        # Run adaptive calculation
        result = self.floquet_engine.calculate_floquet_states_adaptive()
        
        # Should find convergence
        assert 'converged_harmonics' in result, "Adaptive calculation missing convergence info"
        assert result['converged_harmonics'] >= self.floquet_params.n_harmonics, \
            "Converged harmonics should be at least initial value"
        
        # Should have reasonable number of harmonics
        assert result['converged_harmonics'] <= 50, \
            f"Excessive harmonics {result['converged_harmonics']} suggests convergence failure"
        
        # Quasi-energies should be real and finite
        quasi_energies = result['quasi_energies']
        assert np.all(np.isreal(quasi_energies)), "Quasi-energies should be real"
        assert np.all(np.isfinite(quasi_energies)), "Quasi-energies should be finite"
        
        print(f"Converged with {result['converged_harmonics']} harmonics")
        print(f"Quasi-energy range: [{np.min(quasi_energies.real):.2e}, {np.max(quasi_energies.real):.2e}]")
    
    def test_harmonic_convergence_history(self):
        """Test that eigenvalues show convergence pattern"""
        
        result = self.floquet_engine.calculate_floquet_states_adaptive()
        eigenvalue_history = result['eigenvalue_history']
        
        # Should have multiple steps
        assert len(eigenvalue_history) >= 2, "Should have convergence history"
        
        # Calculate convergence rates
        convergence_rates = []
        
        for i in range(1, len(eigenvalue_history)):
            prev_eigs = eigenvalue_history[i-1]
            curr_eigs = eigenvalue_history[i]
            
            # Compare overlapping eigenvalues
            min_len = min(len(prev_eigs), len(curr_eigs))
            if min_len > 0:
                rel_change = np.max(np.abs(curr_eigs[:min_len] - prev_eigs[:min_len]) / 
                                  (np.abs(prev_eigs[:min_len]) + 1e-15))
                convergence_rates.append(rel_change)
        
        print(f"Convergence rates: {[f'{rate:.2e}' for rate in convergence_rates]}")
        
        # Should show decreasing convergence rates (convergence)
        if len(convergence_rates) >= 2:
            # At least the final step should show small change
            assert convergence_rates[-1] < 1e-6, \
                f"Final convergence rate {convergence_rates[-1]:.2e} too large"
    
    def test_harmonic_scaling_weak_driving(self):
        """Test harmonic requirements scale with driving strength"""
        
        driving_amplitudes = [0.01, 0.05, 0.1]  # Weak to moderate
        required_harmonics = []
        
        for amplitude in driving_amplitudes:
            # Create system with different driving amplitude
            params = FloquetSystemParameters(
                driving_frequency=self.floquet_params.driving_frequency,
                driving_amplitude=amplitude,
                n_harmonics=5,
                eigenvalue_tolerance=1e-8,
                norm_condition_threshold=np.pi
            )
            
            engine = RigorousFloquetEngine(self.qed_engine, params)
            
            try:
                result = engine.calculate_floquet_states_adaptive()
                required_harmonics.append(result['converged_harmonics'])
                
                print(f"Amplitude {amplitude:.2f}: {result['converged_harmonics']} harmonics")
                
            except Exception as e:
                print(f"Failed for amplitude {amplitude}: {e}")
                required_harmonics.append(float('inf'))
        
        # Generally expect more harmonics for stronger driving
        valid_harmonics = [h for h in required_harmonics if np.isfinite(h)]
        
        if len(valid_harmonics) >= 2:
            # Should show some scaling trend (not necessarily monotonic due to nonlinear effects)
            max_harmonics = max(valid_harmonics)
            min_harmonics = min(valid_harmonics)
            
            assert max_harmonics <= 3 * min_harmonics, \
                f"Harmonic scaling too extreme: {min_harmonics} to {max_harmonics}"
    
    def test_convergence_failure_detection(self):
        """Test that excessive driving strength is detected"""
        
        # Create system with very strong driving that should fail
        strong_params = FloquetSystemParameters(
            driving_frequency=self.floquet_params.driving_frequency,
            driving_amplitude=1.0,  # Very strong driving
            n_harmonics=5,
            eigenvalue_tolerance=1e-8,
            norm_condition_threshold=np.pi
        )
        
        strong_engine = RigorousFloquetEngine(self.qed_engine, strong_params)
        
        # Should either converge with many harmonics or fail gracefully
        try:
            result = strong_engine.calculate_floquet_states_adaptive()
            
            # If it converges, should require many harmonics
            assert result['converged_harmonics'] > 20, \
                f"Strong driving should require many harmonics, got {result['converged_harmonics']}"
                
        except RuntimeError as e:
            # Failure is acceptable for very strong driving
            assert "failed to converge" in str(e), f"Unexpected error: {e}"
            print("Strong driving properly detected as non-convergent")
    
    def test_floquet_hamiltonian_properties(self):
        """Test mathematical properties of Floquet Hamiltonian"""
        
        result = self.floquet_engine.calculate_floquet_states_adaptive()
        H_F = result['floquet_hamiltonian']
        
        # Should be Hermitian
        assert np.allclose(H_F, H_F.conj().T), "Floquet Hamiltonian not Hermitian"
        
        # Should have finite norm
        H_norm = np.linalg.norm(H_F)
        assert np.isfinite(H_norm), "Floquet Hamiltonian has infinite norm"
        assert H_norm > 0, "Floquet Hamiltonian is zero matrix"
        
        # Eigenvalues should be real (since Hermitian)
        eigenvals = np.linalg.eigvals(H_F)
        assert np.allclose(eigenvals.imag, 0), "Floquet eigenvalues not real"
        
        print(f"Floquet Hamiltonian: {H_F.shape}, norm = {H_norm:.2e}")
    
    def test_quasi_energy_properties(self):
        """Test physical properties of quasi-energies"""
        
        result = self.floquet_engine.calculate_floquet_states_adaptive()
        quasi_energies = result['quasi_energies']
        
        # Should be in reasonable range
        driving_energy = HBAR * self.floquet_params.driving_frequency
        max_expected = 10 * driving_energy  # Rough upper bound
        
        assert np.all(np.abs(quasi_energies) < max_expected), \
            f"Quasi-energies too large: max = {np.max(np.abs(quasi_energies)):.2e}"
        
        # Should have proper periodicity structure (modulo ℏω)
        omega = self.floquet_params.driving_frequency
        period_check = quasi_energies % (HBAR * omega)
        
        # All should be in [0, ℏω) interval
        assert np.all(period_check >= 0), "Quasi-energies outside fundamental zone"
        assert np.all(period_check < HBAR * omega), "Quasi-energies outside fundamental zone"
        
        print(f"Quasi-energy range: [{np.min(quasi_energies):.2e}, {np.max(quasi_energies):.2e}] J")

# Add HBAR constant for tests
HBAR = 1.054571817e-34  # J⋅s

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
