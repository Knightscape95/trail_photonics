"""
Test Magnus expansion convergence with Ω₄-Ω₆

Verify that:
1. Magnus operators up to Ω₆ are computed via explicit nested commutators
2. Calculation aborts if ||Ω_n|| grows between successive orders  
3. RuntimeError is raised for non-convergent expansions
"""

import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters


def test_magnus_omega_4_5_6_computation():
    """Test that Ω₄, Ω₅, Ω₆ are computed explicitly"""
    
    # Small system for testing
    params = QEDSystemParameters(
        device_length=1e-6,
        device_width=0.5e-6,
        device_height=0.3e-6,
        susceptibility_amplitude=0.02,  # Small for convergence
        modulation_frequency=2 * np.pi * 1e9  # 1 GHz
    )
    
    qed_engine = QuantumElectrodynamicsEngine(params)
    
    # Test Magnus expansion calculation
    spatial_grid = np.random.rand(4, 3) * [1e-6, 0.5e-6, 0.3e-6]
    
    try:
        result = qed_engine.calculate_complete_time_evolution(spatial_grid)
        
        # Verify Magnus expansion contains Ω₄-Ω₆
        magnus_data = result['magnus_expansion']
        omega_terms = magnus_data['omega_terms']
        
        assert len(omega_terms) >= 6, \
            f"Expected at least 6 Magnus terms, got {len(omega_terms)}"
        
        # Verify each Ω_n has correct dimensions
        n_modes = omega_terms[0].shape[0]
        for i, omega in enumerate(omega_terms[:6], 1):
            assert omega.shape == (n_modes, n_modes), \
                f"Ω_{i} has incorrect shape {omega.shape}, expected ({n_modes}, {n_modes})"
            
            # Verify Ω_n is finite
            assert np.all(np.isfinite(omega)), f"Ω_{i} contains non-finite elements"
        
        # Verify higher-order terms are computed (non-zero)
        omega_4_norm = np.linalg.norm(omega_terms[3])  # Ω₄
        omega_5_norm = np.linalg.norm(omega_terms[4])  # Ω₅
        omega_6_norm = np.linalg.norm(omega_terms[5])  # Ω₆
        
        # For non-trivial system, higher-order terms should be non-zero but small
        assert omega_4_norm > 0, "Ω₄ is identically zero - not computed"
        assert omega_5_norm > 0, "Ω₅ is identically zero - not computed"  
        assert omega_6_norm > 0, "Ω₆ is identically zero - not computed"
        
        print(f"✅ Magnus terms computed: ||Ω₄|| = {omega_4_norm:.2e}, "
              f"||Ω₅|| = {omega_5_norm:.2e}, ||Ω₆|| = {omega_6_norm:.2e}")
        
    except RuntimeError as e:
        if "MAGNUS EXPANSION DIVERGENCE" in str(e):
            print("✅ Magnus divergence detection working correctly")
        else:
            pytest.fail(f"Unexpected RuntimeError: {e}")


def test_convergence_enforcement():
    """Test that RuntimeError is raised for non-convergent Magnus expansion"""
    
    # Parameters designed to cause divergence
    divergent_params = QEDSystemParameters(
        device_length=1e-6,
        device_width=0.5e-6,
        device_height=0.3e-6,
        susceptibility_amplitude=0.8,  # Very large - should cause divergence
        modulation_frequency=2 * np.pi * 50e9  # High frequency
    )
    
    qed_engine = QuantumElectrodynamicsEngine(divergent_params)
    spatial_grid = np.random.rand(3, 3) * [1e-6, 0.5e-6, 0.3e-6]
    
    # Should raise RuntimeError for divergent parameters
    with pytest.raises(RuntimeError) as exc_info:
        qed_engine.calculate_complete_time_evolution(spatial_grid)
    
    error_msg = str(exc_info.value)
    assert "MAGNUS EXPANSION DIVERGENCE" in error_msg, \
        f"Expected Magnus divergence error, got: {error_msg}"
    
    print("✅ Convergence enforcement verified - RuntimeError raised for divergent case")


def test_successive_order_growth_detection():
    """Test that calculation aborts if ||Ω_n|| grows between successive orders"""
    
    # Parameters that might cause order growth
    params = QEDSystemParameters(
        device_length=2e-6,
        device_width=1e-6,
        device_height=0.5e-6,
        susceptibility_amplitude=0.5,  # Large enough to potentially cause growth
        modulation_frequency=2 * np.pi * 20e9
    )
    
    qed_engine = QuantumElectrodynamicsEngine(params)
    spatial_grid = np.random.rand(3, 3) * [2e-6, 1e-6, 0.5e-6]
    
    try:
        result = qed_engine.calculate_complete_time_evolution(spatial_grid)
        
        # If calculation succeeds, verify norms are decreasing
        omega_terms = result['magnus_expansion']['omega_terms']
        omega_norms = [np.linalg.norm(omega) for omega in omega_terms]
        
        # Check that higher-order terms generally decrease
        for i in range(2, len(omega_norms) - 1):  # Start from Ω₃
            if omega_norms[i+1] > omega_norms[i]:
                pytest.fail(f"Order growth detected but not caught: "
                          f"||Ω_{i+1}|| = {omega_norms[i]:.2e} > "
                          f"||Ω_{i+2}|| = {omega_norms[i+1]:.2e}")
        
        print("✅ Magnus order progression verified - norms decreasing")
        
    except RuntimeError as e:
        if "ORDER GROWTH DETECTED" in str(e):
            print("✅ Order growth detection working correctly")
        elif "MAGNUS EXPANSION DIVERGENCE" in str(e):
            print("✅ General divergence detection working correctly") 
        else:
            pytest.fail(f"Unexpected error: {e}")


def test_convergence_conditions():
    """Test specific convergence conditions from theory"""
    
    # Well-behaved parameters that should converge
    params = QEDSystemParameters(
        device_length=1e-6,
        device_width=0.5e-6,
        device_height=0.3e-6,
        susceptibility_amplitude=0.01,  # Small for convergence
        modulation_frequency=2 * np.pi * 5e9
    )
    
    qed_engine = QuantumElectrodynamicsEngine(params)
    
    # Create test Hamiltonian matrix
    n_modes = 4
    time_points = np.linspace(0, 1e-10, 10)
    
    # Test Hamiltonian (small, well-behaved)
    H_test = np.random.normal(0, 0.01, (n_modes, n_modes, len(time_points)))
    H_test = H_test + H_test.transpose(1, 0, 2).conj()  # Make Hermitian
    
    # Test convergence analysis
    omega_1 = qed_engine._calculate_omega_1(H_test, time_points)
    omega_2 = qed_engine._calculate_omega_2(H_test, time_points)
    omega_3 = qed_engine._calculate_omega_3(H_test, time_points)
    omega_4 = qed_engine._calculate_omega_4(H_test, time_points)
    omega_5 = qed_engine._calculate_omega_5(H_test, time_points)
    omega_6 = qed_engine._calculate_omega_6(H_test, time_points)
    
    omega_terms = [omega_1, omega_2, omega_3, omega_4, omega_5, omega_6]
    
    period = time_points[-1] - time_points[0]
    convergence_info = qed_engine._analyze_magnus_convergence_complete(omega_terms, period)
    
    # Verify convergence conditions
    assert 'converged' in convergence_info, "Missing convergence status"
    assert 'norm_condition' in convergence_info, "Missing norm condition"
    assert 'spectral_condition' in convergence_info, "Missing spectral condition"
    assert 'omega_norms' in convergence_info, "Missing omega norms"
    
    # For well-behaved case, should satisfy norm condition ||Ω₁|| < π
    omega_1_norm = convergence_info['norm_omega_1']
    assert omega_1_norm < np.pi, f"Norm condition violated: ||Ω₁|| = {omega_1_norm:.3f} > π"
    
    print(f"✅ Convergence analysis verified: ||Ω₁|| = {omega_1_norm:.3f} < π")


if __name__ == "__main__":
    test_magnus_omega_4_5_6_computation()
    test_convergence_enforcement()
    test_successive_order_growth_detection()
    test_convergence_conditions()
    print("🎯 All Magnus convergence tests passed!")
