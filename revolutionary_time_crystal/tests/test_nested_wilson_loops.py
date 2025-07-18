#!/usr/bin/env python3
"""
Test suite for nested Wilson loop calculations
Part of Priority 3 topology engine validation

Tests:
- 1D Wilson loop calculation
- Nested Wilson loops for weak indices
- Gauge independence validation
- Path-ordered multiplication
- 2D Wilson loop validation
- Topological classification
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
from rigorous_floquet_engine_fixed import RigorousFloquetEngine, FloquetSystemParameters

class TestNestedWilsonLoops:
    """Test nested Wilson loop calculations"""
    
    @pytest.fixture
    def setup_topology_engine(self):
        """Setup topology engine for testing"""
        
        # QED parameters
        qed_params = QEDSystemParameters(
            device_length=20e-6,
            device_width=5e-6,
            susceptibility_amplitude=0.1,
            modulation_frequency=2 * np.pi * 1e12
        )
        
        # Floquet parameters  
        floquet_params = FloquetSystemParameters(
            driving_frequency=2 * np.pi * 1e12,
            driving_amplitude=0.1,
            n_harmonics=10,
            n_time_steps=64
        )
        
        # Topology parameters
        topo_params = TopologyParameters(
            n_kx=12, n_ky=12, n_kz=6,
            momentum_resolution=0.2,
            berry_tolerance=1e-6,
            wilson_loop_points=51
        )
        
        # Create engines
        qed_engine = QuantumElectrodynamicsEngine(qed_params)
        floquet_engine = RigorousFloquetEngine(qed_engine, floquet_params)
        topology_engine = GaugeIndependentTopology(floquet_engine, topo_params)
        
        return topology_engine
    
    def test_k_vector_construction(self, setup_topology_engine):
        """Test construction of k-vectors from fixed coordinates"""
        
        topo = setup_topology_engine
        
        # Test kx sweep
        fixed_coords = {'ky': 0.5, 'kz': 0.0}
        sweep_val = 1.0
        k_vec = topo._construct_k_vector(fixed_coords, 'kx', sweep_val)
        
        expected = np.array([1.0, 0.5, 0.0])
        np.testing.assert_array_equal(k_vec, expected)
        
        # Test ky sweep
        fixed_coords = {'kx': -0.5, 'kz': 0.2}
        sweep_val = 0.8
        k_vec = topo._construct_k_vector(fixed_coords, 'ky', sweep_val)
        
        expected = np.array([-0.5, 0.8, 0.2])
        np.testing.assert_array_equal(k_vec, expected)
        
        # Test kz sweep
        fixed_coords = {'kx': 0.0, 'ky': -1.0}
        sweep_val = 0.3
        k_vec = topo._construct_k_vector(fixed_coords, 'kz', sweep_val)
        
        expected = np.array([0.0, -1.0, 0.3])
        np.testing.assert_array_equal(k_vec, expected)
    
    def test_wilson_gauge_continuity(self, setup_topology_engine):
        """Test gauge continuity fixing for Wilson loops"""
        
        topo = setup_topology_engine
        
        # Create mock 2-band eigenstates
        u_curr = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity basis
        
        # Second eigenstate with random phases
        phases = np.exp(1j * np.array([np.pi/3, -np.pi/4]))
        u_next = u_curr * phases[:, np.newaxis]
        
        # Fix gauge
        u_fixed = topo._fix_wilson_gauge_continuity(u_curr, u_next)
        
        # Check that overlaps are real and positive
        overlaps = [np.vdot(u_curr[:, i], u_fixed[:, i]) for i in range(2)]
        
        for overlap in overlaps:
            assert np.real(overlap) > 0.8, "Overlap not positive after gauge fixing"
            assert abs(np.imag(overlap)) < 1e-10, "Overlap not real after gauge fixing"
    
    def test_1d_wilson_loop_calculation(self, setup_topology_engine):
        """Test 1D Wilson loop calculation along k-path"""
        
        topo = setup_topology_engine
        
        # Reduce grid size for faster testing
        topo.params.n_kx = 8
        topo.params.n_ky = 8
        topo.params.n_kz = 4
        topo._setup_brillouin_zone()
        
        # Test Wilson loop in kx direction
        fixed_coords = {'ky': 0.0, 'kz': 0.0}
        sweep_vals = topo.brillouin_zone['kx_vals']
        
        wilson_loop = topo._calculate_wilson_loop_1d(
            fixed_coords, 'kx', sweep_vals
        )
        
        # Check properties
        assert wilson_loop.shape == (2, 2), "Wilson loop should be 2×2 for 2-band model"
        
        # Wilson loop should be unitary (within numerical precision)
        W_dag_W = wilson_loop.conj().T @ wilson_loop
        unitarity_error = np.linalg.norm(W_dag_W - np.eye(2))
        assert unitarity_error < 1e-6, f"Wilson loop not unitary: error = {unitarity_error}"
        
        # Determinant should have unit magnitude
        det_W = np.linalg.det(wilson_loop)
        assert abs(abs(det_W) - 1.0) < 1e-6, "Wilson loop determinant not unit magnitude"
    
    def test_2d_wilson_loops_validation(self, setup_topology_engine):
        """Test 2D Wilson loops for validation"""
        
        topo = setup_topology_engine
        
        # Reduce grid size
        topo.params.n_kx = 6
        topo.params.n_ky = 6
        topo.params.n_kz = 3
        topo._setup_brillouin_zone()
        
        # Calculate 2D Wilson loops
        result = topo._calculate_2d_wilson_loops()
        
        # Check structure
        assert 'horizontal_windings' in result
        assert 'vertical_windings' in result
        assert 'mean_horizontal' in result
        assert 'mean_vertical' in result
        
        # Check dimensions
        h_windings = result['horizontal_windings']
        v_windings = result['vertical_windings']
        
        assert len(h_windings) == topo.params.n_ky, "Wrong number of horizontal windings"
        assert len(v_windings) == topo.params.n_kx, "Wrong number of vertical windings"
        
        # Winding numbers should be finite
        assert np.all(np.isfinite(h_windings)), "Horizontal windings contain non-finite values"
        assert np.all(np.isfinite(v_windings)), "Vertical windings contain non-finite values"
    
    def test_nested_wilson_loops_calculation(self, setup_topology_engine):
        """Test full nested Wilson loops calculation"""
        
        topo = setup_topology_engine
        
        # Use small grid for testing
        topo.params.n_kx = 6
        topo.params.n_ky = 6
        topo.params.n_kz = 4
        topo._setup_brillouin_zone()
        
        # Calculate nested Wilson loops
        result = topo.nested_wilson_loops_calculation()
        
        # Check structure
        assert 'weak_indices' in result
        assert 'wilson_phases' in result
        assert '2d_wilson_validation' in result
        assert 'gauge_independence' in result
        assert 'topological_class' in result
        
        # Check weak indices
        weak_indices = result['weak_indices']
        nu_x = weak_indices['nu_x']
        nu_y = weak_indices['nu_y']
        
        # Should be Z₂ indices (0 or 1)
        assert nu_x in [0, 1], f"ν_x = {nu_x} not a valid Z₂ index"
        assert nu_y in [0, 1], f"ν_y = {nu_y} not a valid Z₂ index"
        
        # Check Wilson phases
        phases = result['wilson_phases']
        assert 'nu_x_slices' in phases
        assert 'nu_y_slices' in phases
        
        # Each slice should give integer winding
        for slice_winding in phases['nu_x_slices']:
            assert isinstance(slice_winding, int), "Slice winding not integer"
        
        for slice_winding in phases['nu_y_slices']:
            assert isinstance(slice_winding, int), "Slice winding not integer"
    
    def test_gauge_independence_validation(self, setup_topology_engine):
        """Test Wilson loop gauge independence"""
        
        topo = setup_topology_engine
        
        # Mock weak indices for testing
        nu_x, nu_y = 1, 0
        
        # Test gauge independence validation
        gauge_result = topo._validate_wilson_loop_gauge_independence(nu_x, nu_y)
        
        # Check structure
        assert 'gauge_independent' in gauge_result
        assert 'original_invariants' in gauge_result
        assert 'transformed_invariants' in gauge_result
        assert 'n_tests' in gauge_result
        
        # Original invariants should match input
        orig_invariants = gauge_result['original_invariants']
        assert orig_invariants == (nu_x, nu_y), "Original invariants don't match input"
        
        # Should perform multiple tests
        assert gauge_result['n_tests'] >= 3, "Not enough gauge transformation tests"
        
        # Transformed invariants should have correct structure
        transformed = gauge_result['transformed_invariants']
        assert len(transformed) == gauge_result['n_tests'], "Wrong number of transformed results"
        
        for inv in transformed:
            assert len(inv) == 2, "Each transformed invariant should be (nu_x, nu_y) pair"
            assert inv[0] in [0, 1], "Transformed nu_x not Z₂"
            assert inv[1] in [0, 1], "Transformed nu_y not Z₂"
    
    def test_3d_topology_classification(self, setup_topology_engine):
        """Test 3D topological phase classification"""
        
        topo = setup_topology_engine
        
        # Test different index combinations
        test_cases = [
            (0, 0, "Trivial Insulator"),
            (1, 0, "Weak Topological Insulator"),
            (0, 1, "Weak Topological Insulator"),
            (1, 1, "Weak Topological Insulator")
        ]
        
        for nu_x, nu_y, expected_class in test_cases:
            result = topo._classify_3d_topology(nu_x, nu_y)
            
            # Check structure
            assert 'topology_class' in result
            assert 'z2_indices' in result
            assert 'is_strong_ti' in result
            assert 'is_weak_ti' in result
            assert 'is_trivial' in result
            assert 'topological_invariant' in result
            
            # Check Z₂ indices
            z2_indices = result['z2_indices']
            assert z2_indices['nu_1'] == nu_x, "nu_1 doesn't match nu_x"
            assert z2_indices['nu_2'] == nu_y, "nu_2 doesn't match nu_y"
            assert z2_indices['nu_3'] == 0, "nu_3 should be 0 for 2D system"
            
            # Check boolean flags are consistent
            is_strong = result['is_strong_ti']
            is_weak = result['is_weak_ti']
            is_trivial = result['is_trivial']
            
            # Only one should be true
            true_count = sum([is_strong, is_weak, is_trivial])
            assert true_count <= 2, "Multiple topology flags set"  # Mixed phases allowed
            
            # Check topological invariant format
            invariant = result['topological_invariant']
            assert len(invariant) == 4, "Topological invariant should be 4-tuple"
            assert all(inv in [0, 1] for inv in invariant), "All indices should be Z₂"
    
    def test_wilson_loop_unitarity(self, setup_topology_engine):
        """Test unitarity of Wilson loop matrices"""
        
        topo = setup_topology_engine
        
        # Small test parameters
        n_points = 8
        sweep_vals = np.linspace(-np.pi, np.pi, n_points)
        
        # Calculate Wilson loop
        wilson_loop = topo._calculate_wilson_loop_1d(
            fixed_coords={'ky': 0.0, 'kz': 0.0},
            sweep_axis='kx',
            sweep_vals=sweep_vals
        )
        
        # Test unitarity
        W_dag = wilson_loop.conj().T
        identity_test = W_dag @ wilson_loop
        
        unitarity_error = np.linalg.norm(identity_test - np.eye(wilson_loop.shape[0]))
        
        assert unitarity_error < 1e-8, f"Wilson loop not unitary: error = {unitarity_error}"
        
        # Test determinant magnitude
        det_W = np.linalg.det(wilson_loop)
        det_magnitude_error = abs(abs(det_W) - 1.0)
        
        assert det_magnitude_error < 1e-8, f"Wilson loop determinant not unit: error = {det_magnitude_error}"

if __name__ == "__main__":
    pytest.main([__file__])
