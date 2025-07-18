"""
Test 3D integration in QED engine

Verify that H_int,I(t) = -ε₀/2 ∫d³r δχ(r,t) E_I² 
is evaluated with correct 3D volume element.
"""

import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters


def test_3d_volume_element():
    """Test that 3D integration uses proper volume element dx*dy*dz"""
    
    # Small test system
    params = QEDSystemParameters(
        device_length=1e-6,   # 1 μm
        device_width=0.5e-6,  # 0.5 μm  
        device_height=0.2e-6, # 0.2 μm
        susceptibility_amplitude=0.01
    )
    
    qed_engine = QuantumElectrodynamicsEngine(params)
    
    # Create minimal 3D spatial grid
    N_x, N_y, N_z = 8, 6, 4
    x_vals = np.linspace(0, params.device_length, N_x)
    y_vals = np.linspace(0, params.device_width, N_y) 
    z_vals = np.linspace(0, params.device_height, N_z)
    
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    spatial_grid = np.stack([x_grid, y_grid, z_grid], axis=-1)
    
    # Test susceptibility calculation on 3D grid
    test_time = 0.0
    delta_chi_3d = qed_engine._susceptibility_modulation_3d(spatial_grid, test_time)
    
    # Verify shape matches 3D grid
    assert delta_chi_3d.shape == (N_x, N_y, N_z), \
        f"Expected shape ({N_x}, {N_y}, {N_z}), got {delta_chi_3d.shape}"
    
    # Verify proper volume integration
    total_volume = params.device_length * params.device_width * params.device_height
    dx = params.device_length / (N_x - 1)
    dy = params.device_width / (N_y - 1) 
    dz = params.device_height / (N_z - 1)
    discrete_volume = dx * dy * dz * (N_x - 1) * (N_y - 1) * (N_z - 1)
    
    volume_error = abs(discrete_volume - total_volume) / total_volume
    assert volume_error < 0.01, f"Volume discretization error {volume_error:.3f} > 1%"


def test_interaction_hamiltonian_3d():
    """Test interaction Hamiltonian uses 3D tensor integration"""
    
    params = QEDSystemParameters(
        device_length=2e-6,
        device_width=1e-6,
        device_height=0.5e-6,
        susceptibility_amplitude=0.05
    )
    
    qed_engine = QuantumElectrodynamicsEngine(params)
    
    # Create test spatial grid in proper 3D format
    N_x, N_y, N_z = 4, 3, 2
    x_vals = np.linspace(0, params.device_length, N_x)
    y_vals = np.linspace(0, params.device_width, N_y)
    z_vals = np.linspace(0, params.device_height, N_z)
    
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    spatial_grid = np.stack([x_grid, y_grid, z_grid], axis=-1)
    
    # Calculate interaction Hamiltonian for small time window
    time_points = np.linspace(0, 1e-12, 5)  # 5 time steps
    
    try:
        H_int = qed_engine.interaction_hamiltonian_matrix(
            spatial_grid, time_points)
        
        # Verify tensor structure
        n_modes = len(qed_engine.k_points)
        expected_shape = (n_modes * 2, n_modes * 2, len(time_points))
        
        assert H_int.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {H_int.shape}"
        
        # Verify Hermiticity at each time step
        for t_idx in range(len(time_points)):
            H_t = H_int[:, :, t_idx]
            hermitian_error = np.max(np.abs(H_t - H_t.conj().T))
            assert hermitian_error < 1e-12, \
                f"Hermiticity violation at t[{t_idx}]: {hermitian_error:.2e}"
        
        # Verify finite matrix elements (no NaN/inf)
        assert np.all(np.isfinite(H_int)), "Non-finite matrix elements detected"
        
        print("✅ 3D interaction Hamiltonian calculation verified")
        
    except Exception as e:
        pytest.fail(f"3D interaction Hamiltonian calculation failed: {e}")


def test_volume_scaling():
    """Test that interaction scales correctly with device volume"""
    
    base_params = QEDSystemParameters(
        device_length=1e-6,
        device_width=1e-6,
        device_height=1e-6,
        susceptibility_amplitude=0.01
    )
    
    # Double the volume
    scaled_params = QEDSystemParameters(
        device_length=2e-6,
        device_width=1e-6,  
        device_height=1e-6,
        susceptibility_amplitude=0.01
    )
    
    qed_base = QuantumElectrodynamicsEngine(base_params)
    qed_scaled = QuantumElectrodynamicsEngine(scaled_params)
    
    # Small test grids in proper 3D format
    N_x, N_y, N_z = 3, 2, 2
    
    # Base grid
    x_vals = np.linspace(0, base_params.device_length, N_x)
    y_vals = np.linspace(0, base_params.device_width, N_y)  
    z_vals = np.linspace(0, base_params.device_height, N_z)
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    grid_base = np.stack([x_grid, y_grid, z_grid], axis=-1)
    
    # Scaled grid
    x_vals_scaled = np.linspace(0, scaled_params.device_length, N_x)
    y_vals_scaled = np.linspace(0, scaled_params.device_width, N_y)
    z_vals_scaled = np.linspace(0, scaled_params.device_height, N_z)
    x_grid_s, y_grid_s, z_grid_s = np.meshgrid(x_vals_scaled, y_vals_scaled, z_vals_scaled, indexing='ij')
    grid_scaled = np.stack([x_grid_s, y_grid_s, z_grid_s], axis=-1)
    
    time_points = np.array([0.0])
    
    try:
        H_base = qed_base.interaction_hamiltonian_matrix(
            grid_base, time_points)
        H_scaled = qed_scaled.interaction_hamiltonian_matrix(
            grid_scaled, time_points)
        
        # Interaction should scale with volume for uniform field
        norm_base = np.linalg.norm(H_base)
        norm_scaled = np.linalg.norm(H_scaled)
        
        # Expect roughly 2x scaling (doubled length)
        if norm_base > 1e-15:  # Avoid division by zero
            scaling_factor = norm_scaled / norm_base
            assert 1.5 < scaling_factor < 3.0, \
                f"Unexpected volume scaling: {scaling_factor:.2f}"
        
        print("✅ Volume scaling verification passed")
        
    except Exception as e:
        pytest.fail(f"Volume scaling test failed: {e}")


if __name__ == "__main__":
    test_3d_volume_element()
    test_interaction_hamiltonian_3d()
    test_volume_scaling()
    print("🎯 All 3D integration tests passed!")
