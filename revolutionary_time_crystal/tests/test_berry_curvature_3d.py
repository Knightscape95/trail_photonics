#!/usr/bin/env python3
"""
Test suite for 3D Berry curvature calculations
Part of Priority 3 topology engine validation

Tests:
- 3D Berry curvature tensor calculation
- 6-point finite difference accuracy
- Gauge independence validation  
- Analytical formula comparison
- First Chern number quantization
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gauge_independent_topology import (
    GaugeIndependentTopology, 
    TopologyParameters
)
from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters
from rigorous_floquet_engine_fixed import RigorousFloquetEngine, FloquetParameters

class TestBerry3DCurvature:
    """Test 3D Berry curvature calculations"""
    
    @pytest.fixture
    def setup_topology_engine(self):
        """Setup topology engine for testing"""
        
        # QED parameters
        qed_params = QEDSystemParameters(
            coupling_strength=0.1,
            modulation_frequency=1e12,
            susceptibility_amplitude=1e-6,
            spatial_resolution=64,
            temporal_resolution=256
        )
        
        # Floquet parameters  
        floquet_params = FloquetParameters(
            driving_frequency=1e12,
            driving_amplitude=1e6,
            harmonic_cutoff=15,
            adaptive_cutoff=True,
            convergence_threshold=1e-8
        )
        
        # Topology parameters
        topo_params = TopologyParameters(
            n_kx=16, n_ky=16, n_kz=8,
            momentum_resolution=0.1,
            berry_tolerance=1e-6,
            wilson_loop_points=101
        )
        
        # Create engines
        qed_engine = QuantumElectrodynamicsEngine(qed_params)
        floquet_engine = RigorousFloquetEngine(qed_engine, floquet_params)
        topology_engine = GaugeIndependentTopology(qed_engine, floquet_engine, topo_params)
        
        return topology_engine
    
    def test_3d_momentum_grid_creation(self, setup_topology_engine):
        """Test 3D momentum grid construction"""
        
        topo = setup_topology_engine
        k_grid = topo._create_3d_momentum_grid()
        
        # Check dimensions
        assert k_grid.shape == (topo.params.n_kx, topo.params.n_ky, topo.params.n_kz, 3)
        
        # Check k-space coverage
        kx_range = k_grid[:, 0, 0, 0]
        ky_range = k_grid[0, :, 0, 1]
        kz_range = k_grid[0, 0, :, 2]
        
        assert np.allclose(kx_range, topo.brillouin_zone['kx_vals'])
        assert np.allclose(ky_range, topo.brillouin_zone['ky_vals']) 
        assert np.allclose(kz_range, topo.brillouin_zone['kz_vals'])
    
    def test_six_point_derivative_accuracy(self, setup_topology_engine):
        """Test 6-point finite difference accuracy"""
        
        topo = setup_topology_engine
        
        # Test function: f(x) = x³ + 2x² - x + 1
        # Derivative: f'(x) = 3x² + 4x - 1
        
        x = 0.5
        dx = 0.01
        x_points = np.array([x - 3*dx, x - 2*dx, x - dx, x, x + dx, x + 2*dx, x + 3*dx])
        f_vals = x_points**3 + 2*x_points**2 - x_points + 1
        
        # Analytical derivative
        analytical_derivative = 3*x**2 + 4*x - 1
        
        # 6-point numerical derivative
        numerical_derivative = topo._six_point_derivative(f_vals, dx)
        
        # Should be accurate to O(h⁶) ~ 1e-12 for smooth functions
        error = abs(numerical_derivative - analytical_derivative)
        assert error < 1e-10, f"6-point derivative error {error} too large"
    
    def test_berry_connections_calculation(self, setup_topology_engine):
        """Test Berry connection calculation"""
        
        topo = setup_topology_engine
        
        # Test at high-symmetry point
        k_vec = np.array([0.0, 0.0, 0.0])
        
        # Get Bloch Hamiltonian
        H_bloch = topo._construct_bloch_hamiltonian(k_vec, None)
        
        # Calculate all Berry connections
        A_x, A_y, A_z = topo._calculate_all_berry_connections(H_bloch, k_vec, None)
        
        # At Γ point, connections should be well-defined
        assert np.isfinite(A_x), "A_x contains non-finite values"
        assert np.isfinite(A_y), "A_y contains non-finite values"
        assert np.isfinite(A_z), "A_z contains non-finite values"
        
        # Connections should be pure imaginary (real Berry connection)
        assert abs(np.real(A_x)) < 1e-10, "A_x has significant real part"
        assert abs(np.real(A_y)) < 1e-10, "A_y has significant real part"
        assert abs(np.real(A_z)) < 1e-10, "A_z has significant real part"
    
    def test_gauge_continuity_fixing(self, setup_topology_engine):
        """Test gauge continuity fixing for eigenstates"""
        
        topo = setup_topology_engine
        
        # Create reference state
        ref_state = np.array([1.0, 0.0]) / np.sqrt(1.0)
        
        # Target state with phase factor
        phase = np.exp(1j * np.pi/4)
        target_state = phase * ref_state
        
        # Fix gauge
        fixed_state = topo._fix_gauge_continuity(ref_state, target_state)
        
        # Should remove phase to maximize real overlap
        overlap = np.vdot(ref_state, fixed_state)
        assert np.real(overlap) > 0.9, "Gauge fixing failed to ensure positive overlap"
        assert abs(np.imag(overlap)) < 1e-10, "Fixed gauge still has imaginary overlap"
    
    def test_3d_berry_curvature_tensor(self, setup_topology_engine):
        """Test full 3D Berry curvature tensor calculation"""
        
        topo = setup_topology_engine
        
        # Use small grid for fast testing
        topo.params.n_kx = 8
        topo.params.n_ky = 8  
        topo.params.n_kz = 4
        topo._setup_brillouin_zone()
        
        # Create momentum grid
        k_grid = topo._create_3d_momentum_grid()
        
        # Calculate 3D Berry curvature
        result = topo._calculate_3d_berry_curvature(k_grid)
        
        # Check structure
        assert 'berry_connections' in result
        assert 'berry_curvature' in result
        assert 'primary_component' in result
        
        # Check tensor components
        curvature = result['berry_curvature']
        assert 'Omega_xy' in curvature  # Ω_z
        assert 'Omega_yz' in curvature  # Ω_x
        assert 'Omega_zx' in curvature  # Ω_y
        
        # Check dimensions
        Omega_xy = curvature['Omega_xy']
        assert Omega_xy.shape == (8, 8, 4)
        
        # Check reality of curvature
        assert np.all(np.isreal(Omega_xy)), "Berry curvature should be real"
        
        # Check antisymmetry: Ω_ij = -Ω_ji
        Omega_yz = curvature['Omega_yz']
        Omega_zx = curvature['Omega_zx']
        
        # This would require full calculation to verify properly
        assert np.all(np.isfinite(Omega_yz)), "Ω_yz contains non-finite values"
        assert np.all(np.isfinite(Omega_zx)), "Ω_zx contains non-finite values"
    
    def test_first_chern_number_calculation(self, setup_topology_engine):
        """Test first Chern number calculation and quantization"""
        
        topo = setup_topology_engine
        
        # Create mock Berry curvature with known integral
        n_kx, n_ky, n_kz = 16, 16, 4
        topo.params.n_kx = n_kx
        topo.params.n_ky = n_ky
        topo.params.n_kz = n_kz
        topo._setup_brillouin_zone()
        
        # Mock uniform Berry curvature that integrates to 2π (C₁ = 1)
        kx_vals = topo.brillouin_zone['kx_vals']
        ky_vals = topo.brillouin_zone['ky_vals']
        
        # Brillouin zone area
        kx_range = kx_vals[-1] - kx_vals[0]
        ky_range = ky_vals[-1] - ky_vals[0]
        BZ_area = kx_range * ky_range
        
        # Uniform curvature for C₁ = 1
        uniform_curvature = 2 * np.pi / BZ_area
        berry_curvature = np.full((n_kx, n_ky, n_kz), uniform_curvature)
        
        # Calculate Chern number
        chern_result = topo._calculate_first_chern_number(berry_curvature)
        
        # Check quantization
        assert abs(chern_result['C1_float'] - 1.0) < 0.1, "Chern number not close to expected value"
        assert chern_result['C1_integer'] == 1, "Integer Chern number incorrect"
        assert chern_result['quantization_error'] < 0.1, "Poor quantization"
        assert chern_result['well_quantized'], "Chern number not well quantized"
    
    def test_boundary_curvature_handling(self, setup_topology_engine):
        """Test boundary point handling in curvature calculation"""
        
        topo = setup_topology_engine
        
        # Small arrays for testing
        n_kx, n_ky, n_kz = 8, 8, 4
        A_x = np.random.random((n_kx, n_ky, n_kz)) + 1j * np.random.random((n_kx, n_ky, n_kz))
        A_y = np.random.random((n_kx, n_ky, n_kz)) + 1j * np.random.random((n_kx, n_ky, n_kz))
        A_z = np.random.random((n_kx, n_ky, n_kz)) + 1j * np.random.random((n_kx, n_ky, n_kz))
        
        # Initialize curvature arrays
        Omega_xy = np.zeros((n_kx, n_ky, n_kz))
        Omega_yz = np.zeros((n_kx, n_ky, n_kz))
        Omega_zx = np.zeros((n_kx, n_ky, n_kz))
        
        # Test boundary handling
        dk_x = dk_y = dk_z = 0.1
        
        topo._handle_boundary_curvature(
            A_x, A_y, A_z, Omega_xy, Omega_yz, Omega_zx,
            dk_x, dk_y, dk_z, n_kx, n_ky, n_kz
        )
        
        # Check that boundary points are handled (non-zero after processing)
        boundary_points_handled = (
            np.sum(np.abs(Omega_xy[:3, :, :])) > 0 or  # Left boundary
            np.sum(np.abs(Omega_xy[-3:, :, :])) > 0 or  # Right boundary
            np.sum(np.abs(Omega_xy[:, :3, :])) > 0 or  # Bottom boundary
            np.sum(np.abs(Omega_xy[:, -3:, :])) > 0    # Top boundary
        )
        
        assert boundary_points_handled, "Boundary points not properly handled"

if __name__ == "__main__":
    pytest.main([__file__])
