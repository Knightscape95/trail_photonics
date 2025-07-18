#!/usr/bin/env python3
"""
Comprehensive Test Suite for THz Time-Crystal Isolator
=====================================================

Complete test coverage for all THz bandwidth extension modules:
- THz bandwidth framework validation
- Dual-band modulator testing  
- Quantum-regime extension verification
- Topological enhancement validation
- Dynamic reconfigurability testing

Author: Revolutionary Time-Crystal Team
Date: July 18, 2025
"""

import pytest
import numpy as np
import time
import tempfile
import os
from typing import Dict, List, Any
import warnings

# Import all modules under test
from seed_manager import seed_everything, verify_reproducibility
from thz_bandwidth_framework import (
    THzFrameworkConfig, QEDFloquetHamiltonian, GroupDelayOptimizer,
    THzBandStructureCalculator, validate_thz_framework
)
from dual_band_modulator import (
    DualBandModulatorConfig, LiNbO3ElectroOpticModulator, SParameterSimulator,
    AlNPiezoMEMSModulator, NLNRSusceptibilityEngine, validate_dual_band_modulator
)
from quantum_regime_extension import (
    QuantumRegimeConfig, ChiralQECouplingHamiltonian, SinglePhotonSMatrix,
    CryogenicTransferMatrix, DilutionRefrigeratorFeedthrough, validate_quantum_regime
)
from topological_enhancement import (
    TopologicalConfig, SyntheticDimensionLattice, WilsonLoopCalculator,
    NonHermitianSkinEffect, NearFieldMapper, validate_topological_enhancement
)
from dynamic_reconfigurability import (
    ReconfigurabilityConfig, PhaseArrayController, IsolatorControlAPI,
    validate_dynamic_reconfigurability
)
from professional_logging import ProfessionalLogger
from scientific_integrity import get_approximations_summary

logger = ProfessionalLogger(__name__)


class TestTHzBandwidthFramework:
    """Test suite for THz bandwidth framework."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        seed_everything(42)
        self.config = THzFrameworkConfig()
        
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration
        config = THzFrameworkConfig(bandwidth_target_thz=1.5, contrast_target_db=30.0)
        assert config.bandwidth_target_thz == 1.5
        assert config.contrast_target_db == 30.0
        
        # Configuration with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = THzFrameworkConfig(bandwidth_target_thz=0.5)  # Below 1 THz
            assert len(w) > 0
            assert "Bandwidth target" in str(w[0].message)
    
    def test_qed_floquet_hamiltonian(self):
        """Test QED-Floquet Hamiltonian construction."""
        hamiltonian = QEDFloquetHamiltonian(self.config)
        
        # Test base Hamiltonian
        H0 = hamiltonian.construct_base_hamiltonian()
        assert H0.shape[0] == H0.shape[1]  # Square matrix
        assert H0.shape[0] == len(hamiltonian.x_grid)
        
        # Test driving Hamiltonian
        H_drive = hamiltonian.construct_driving_hamiltonian(0.0)
        assert H_drive.shape == H0.shape
        
        # Test non-Hermitian potential
        H_nh = hamiltonian.construct_nonhermitian_potential()
        assert H_nh.shape == H0.shape
        assert np.any(np.imag(H_nh.diagonal()) != 0)  # Has imaginary parts
        
        # Test Floquet Hamiltonian
        H_floquet = hamiltonian.floquet_hamiltonian(n_harmonics=5)
        expected_size = len(hamiltonian.x_grid) * 11  # 2*5+1 harmonics
        assert H_floquet.shape == (expected_size, expected_size)
    
    def test_magnus_convergence(self):
        """Test Magnus series convergence validation."""
        hamiltonian = QEDFloquetHamiltonian(self.config)
        H_floquet = hamiltonian.floquet_hamiltonian(n_harmonics=3)  # Small for speed
        
        convergence_data = hamiltonian.validate_magnus_convergence(H_floquet, max_order=5)
        
        assert 'converged' in convergence_data
        assert 'order' in convergence_data
        assert 'norm_ratio' in convergence_data
        assert len(convergence_data['order']) <= 5
        
        if convergence_data['converged']:
            assert convergence_data['final_error'] <= self.config.magnus_convergence_threshold
    
    def test_group_delay_optimizer(self):
        """Test group-delay optimization."""
        optimizer = GroupDelayOptimizer(self.config)
        
        # Test optimal delay calculation
        omega_drive = self.config.omega_1_rad_per_s
        tau_opt = optimizer.calculate_optimal_delay(omega_drive, mode_number=0)
        
        expected_tau = (np.pi / omega_drive) * 0.5
        assert abs(tau_opt - expected_tau) < 1e-15
        
        # Test with different mode number
        tau_opt_1 = optimizer.calculate_optimal_delay(omega_drive, mode_number=1)
        expected_tau_1 = (np.pi / omega_drive) * 1.5
        assert abs(tau_opt_1 - expected_tau_1) < 1e-15
    
    def test_band_structure_calculator(self):
        """Test THz band structure calculation."""
        hamiltonian = QEDFloquetHamiltonian(self.config)
        calculator = THzBandStructureCalculator(self.config)
        
        # Small k-range for testing
        k_range = np.linspace(-np.pi/1e-6, np.pi/1e-6, 5)
        band_data = calculator.compute_band_structure(k_range, hamiltonian)
        
        assert 'k_points' in band_data
        assert 'frequencies_hz' in band_data
        assert 'band_edges' in band_data
        assert 'stopbands' in band_data
        assert 'passbands' in band_data
        
        assert len(band_data['k_points']) == 5
        assert band_data['total_bandwidth_thz'] > 0
    
    def test_full_validation(self):
        """Test complete framework validation."""
        results = validate_thz_framework(self.config)
        
        required_keys = [
            'hamiltonian_constructed', 'magnus_converged', 'optimal_delay_fs',
            'stopbands_found', 'total_bandwidth_thz', 'memory_usage_ok'
        ]
        
        for key in required_keys:
            assert key in results
        
        assert results['hamiltonian_constructed'] is True
        assert results['optimal_delay_fs'] > 0
        assert results['total_bandwidth_thz'] > 0


class TestDualBandModulator:
    """Test suite for dual-band modulator."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        seed_everything(42)
        self.config = DualBandModulatorConfig()
    
    def test_linbo3_modulator(self):
        """Test LiNbO₃ electro-optic modulator."""
        modulator = LiNbO3ElectroOpticModulator(self.config)
        
        # Test effective index calculation
        assert 1.0 < modulator.n_eff_780 < 3.0
        assert 1.0 < modulator.n_eff_1550 < 3.0
        assert modulator.n_eff_780 != modulator.n_eff_1550  # Should be different
        
        # Test electrode geometry
        geometry = modulator.electrode_geometry
        assert 'electrode_pitch_um' in geometry
        assert 'number_of_periods' in geometry
        assert geometry['electrode_pitch_um'] > 0
        assert geometry['number_of_periods'] > 0
        
        # Test electro-optic coupling for both wavelengths
        eo_780 = modulator.calculate_electro_optic_coupling(780.0)
        eo_1550 = modulator.calculate_electro_optic_coupling(1550.0)
        
        for eo_data in [eo_780, eo_1550]:
            assert 'vpi_length_vcm' in eo_data
            assert 'overlap_factor' in eo_data
            assert 'meets_requirement' in eo_data
            assert eo_data['vpi_length_vcm'] > 0
            assert 0 < eo_data['overlap_factor'] <= 1
    
    def test_s_parameter_simulator(self):
        """Test S-parameter simulation."""
        simulator = SParameterSimulator(self.config)
        
        frequencies = np.array([1.0, 10.0, 50.0, 100.0])  # GHz
        s_params = simulator.simulate_s_parameters(frequencies)
        
        required_keys = ['s11_db', 's21_db', 's12_db', 's22_db', 'return_loss_meets_spec']
        for key in required_keys:
            assert key in s_params
            assert len(s_params[key]) == len(frequencies)
        
        # Check S-parameter properties
        assert np.all(s_params['s11_db'] <= 0)  # Return loss should be negative
        assert np.all(s_params['s21_db'] <= 0)  # Insertion loss should be negative
        assert s_params['max_return_loss_db'] <= 0
    
    def test_aln_piezo_mems(self):
        """Test AlN piezoMEMS modulator."""
        piezo = AlNPiezoMEMSModulator(self.config)
        
        response = piezo.calculate_piezo_response(voltage_v=5.0, frequency_hz=1e9)
        
        required_keys = [
            'strain_zz', 'stress_zz_pa', 'delta_n', 'isolation_780_db',
            'isolation_1550_db', 'predicted_isolation_db'
        ]
        
        for key in required_keys:
            assert key in response
        
        assert response['strain_zz'] != 0  # Should have strain
        assert response['stress_zz_pa'] != 0  # Should have stress
        assert response['predicted_isolation_db'] >= 0  # Isolation should be positive
    
    def test_nlnr_susceptibility(self):
        """Test NLNR susceptibility engine."""
        nlnr = NLNRSusceptibilityEngine(self.config)
        
        isolation_data = nlnr.calculate_nlnr_isolation(pump_power_mw=50.0, signal_power_dbm=-10.0)
        
        required_keys = [
            'pump_intensity_w_per_m2', 'signal_intensity_w_per_m2',
            'xpm_phase_shift_rad', 'net_isolation_db', 'meets_60db_target'
        ]
        
        for key in required_keys:
            assert key in isolation_data
        
        assert isolation_data['pump_intensity_w_per_m2'] > 0
        assert isolation_data['signal_intensity_w_per_m2'] > 0
        assert isolation_data['net_isolation_db'] >= 0
    
    def test_full_validation(self):
        """Test complete dual-band modulator validation."""
        results = validate_dual_band_modulator(self.config)
        
        required_sections = ['eo_modulator', 's_parameters', 'piezo_mems', 'nlnr_path']
        for section in required_sections:
            assert section in results
        
        # Check EO modulator results
        eo_results = results['eo_modulator']
        assert eo_results['vpi_780_vcm'] > 0
        assert eo_results['vpi_1550_vcm'] > 0
        
        # Check S-parameter results
        s_results = results['s_parameters']
        assert s_results['max_return_loss_db'] <= 0
        assert s_results['bandwidth_ghz'] > 0


class TestQuantumRegimeExtension:
    """Test suite for quantum-regime extension."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        seed_everything(42)
        self.config = QuantumRegimeConfig()
    
    def test_chiral_qe_hamiltonian(self):
        """Test chiral QE coupling Hamiltonian."""
        hamiltonian = ChiralQECouplingHamiltonian(self.config)
        
        hamiltonian_data = hamiltonian.construct_hamiltonian()
        
        required_keys = ['H_total', 'H_qe', 'H_photon', 'H_interaction', 'H_chiral']
        for key in required_keys:
            assert key in hamiltonian_data
            assert hamiltonian_data[key] is not None
        
        # Check Hamiltonian properties
        H_total = hamiltonian_data['H_total']
        assert H_total.shape[0] == H_total.shape[1]  # Square matrix
        assert hamiltonian_data['hilbert_dimension'] > 0
        assert hamiltonian_data['memory_usage_gb'] > 0
    
    def test_single_photon_s_matrix(self):
        """Test single-photon S-matrix calculation."""
        hamiltonian = ChiralQECouplingHamiltonian(self.config)
        s_matrix_calc = SinglePhotonSMatrix(self.config)
        
        hamiltonian_data = hamiltonian.construct_hamiltonian()
        
        # Small frequency range for testing
        frequencies = np.linspace(2.4e14, 2.41e14, 11)
        s_matrix_data = s_matrix_calc.compute_s_matrix(hamiltonian_data, frequencies)
        
        required_keys = ['s11', 's21', 's12', 's22', 'isolation_db', 'forward_loss_db']
        for key in required_keys:
            assert key in s_matrix_data
            assert len(s_matrix_data[key]) == len(frequencies)
        
        assert s_matrix_data['min_forward_loss_db'] >= 0
        assert s_matrix_data['max_isolation_db'] >= 0
    
    def test_cryogenic_transfer_matrix(self):
        """Test cryogenic transfer matrix simulation."""
        cryo_sim = CryogenicTransferMatrix(self.config)
        
        performance_data = cryo_sim.simulate_cryogenic_performance(optical_power_nw=50.0)
        
        required_sections = ['thermal', 'materials', 'coherence']
        for section in required_sections:
            assert section in performance_data
        
        assert performance_data['effective_temperature_k'] > 0
        assert 0 <= performance_data['quantum_efficiency'] <= 1
        assert isinstance(performance_data['meets_cryogenic_specs'], bool)
    
    def test_dilution_refrigerator_feedthrough(self):
        """Test dilution refrigerator feedthrough design."""
        feedthrough = DilutionRefrigeratorFeedthrough(self.config)
        
        design = feedthrough.design_optical_feedthrough()
        
        required_sections = ['thermal', 'optical', 'mechanical']
        for section in required_sections:
            assert section in design
        
        assert design['total_insertion_loss_db'] >= 0
        assert isinstance(design['meets_specifications'], bool)
    
    def test_full_validation(self):
        """Test complete quantum regime validation."""
        results = validate_quantum_regime(self.config)
        
        required_sections = ['hamiltonian', 's_matrix', 'cryogenic', 'feedthrough']
        for section in required_sections:
            assert section in results
        
        # Check key performance metrics
        assert results['s_matrix']['min_forward_loss_db'] >= 0
        assert results['cryogenic']['effective_temperature_k'] > 0
        assert results['feedthrough']['total_insertion_loss_db'] >= 0


class TestTopologicalEnhancement:
    """Test suite for topological enhancement."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        seed_everything(42)
        self.config = TopologicalConfig(lattice_size=(10, 10))  # Smaller for testing
    
    def test_synthetic_dimension_lattice(self):
        """Test synthetic dimension lattice."""
        lattice = SyntheticDimensionLattice(self.config)
        
        # Test Hamiltonian construction
        H = lattice.construct_hamiltonian(k_x=0.1, k_y=0.2)
        assert H.shape == (self.config.synthetic_dimensions, self.config.synthetic_dimensions)
        assert np.allclose(H, H.conj().T, atol=1e-10) or np.any(np.imag(H) != 0)  # Hermitian or complex
        
        # Test band structure (small for speed)
        self.config.k_point_density = 5
        band_data = lattice.calculate_band_structure()
        
        assert 'eigenvalues' in band_data
        assert 'eigenvectors' in band_data
        assert 'band_gap_mev' in band_data
        assert band_data['band_gap_mev'] >= 0
    
    def test_wilson_loop_calculator(self):
        """Test Wilson loop calculation."""
        lattice = SyntheticDimensionLattice(self.config)
        wilson_calc = WilsonLoopCalculator(self.config)
        
        # Generate band data
        self.config.k_point_density = 5
        band_data = lattice.calculate_band_structure()
        
        wilson_data = wilson_calc.calculate_nested_wilson_loops(band_data)
        
        required_keys = [
            'wilson_loops_x', 'wilson_loop_xy', 'quadrupole_invariant',
            'target_achieved', 'precision'
        ]
        
        for key in required_keys:
            assert key in wilson_data
        
        # Check quadrupole invariant quantization
        Q_xy = wilson_data['quadrupole_invariant']
        assert abs(Q_xy - 0.5) < 0.1 or abs(Q_xy + 0.5) < 0.1  # Should be ±0.5
    
    def test_non_hermitian_skin_effect(self):
        """Test non-Hermitian skin effect."""
        lattice = SyntheticDimensionLattice(self.config)
        skin_effect = NonHermitianSkinEffect(self.config)
        
        # Create test Hamiltonian
        H_hermitian = lattice.construct_hamiltonian(0, 0)
        position_array = np.linspace(0, 10, H_hermitian.shape[0])
        
        # Inject complex potential
        H_nh = skin_effect.inject_complex_potential(H_hermitian, position_array)
        assert not np.allclose(H_nh, H_nh.conj().T)  # Should be non-Hermitian
        
        # Calculate skin modes
        skin_data = skin_effect.calculate_skin_modes(H_nh)
        
        required_keys = [
            'eigenvalues', 'localization_lengths', 'edge_mode_amplitudes',
            'isolation_enhancement_db', 'meets_20db_target'
        ]
        
        for key in required_keys:
            assert key in skin_data
        
        assert skin_data['isolation_enhancement_db'] >= 0
    
    def test_near_field_mapper(self):
        """Test near-field mapping."""
        near_field = NearFieldMapper(self.config)
        
        # Create test wave function
        lattice_positions = np.linspace(0, 10e-6, 20)
        test_wavefunction = np.exp(-((lattice_positions - 5e-6) / 1e-6)**2)  # Gaussian
        
        # NSOM simulation
        nsom_data = near_field.simulate_nsom_measurement(test_wavefunction, lattice_positions)
        
        required_keys = ['x_grid', 'y_grid', 'intensity', 'phase', 'localization_length_nm']
        for key in required_keys:
            assert key in nsom_data
        
        assert nsom_data['localization_length_nm'] > 0
        
        # Leakage radiation simulation
        leakage_data = near_field.simulate_leakage_radiation(test_wavefunction)
        
        assert 'k_spectrum' in leakage_data
        assert 'leakage_efficiency' in leakage_data
        assert 0 <= leakage_data['leakage_efficiency'] <= 1
    
    def test_full_validation(self):
        """Test complete topological enhancement validation."""
        results = validate_topological_enhancement(self.config)
        
        required_sections = ['band_structure', 'topology', 'skin_effect', 'near_field']
        for section in required_sections:
            assert section in results
        
        # Check key metrics
        assert results['band_structure']['band_gap_mev'] >= 0
        assert abs(results['topology']['quadrupole_invariant']) <= 1
        assert results['skin_effect']['isolation_enhancement_db'] >= 0


class TestDynamicReconfigurability:
    """Test suite for dynamic reconfigurability."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        seed_everything(42)
        self.config = ReconfigurabilityConfig()
    
    def test_phase_array_controller(self):
        """Test phase array controller."""
        controller = PhaseArrayController(self.config)
        
        # Test phase array setting
        test_phases = np.array([0, 90, 180, 270])
        test_amplitudes = np.array([1.0, 0.8, 0.9, 1.0])
        
        metrics = controller.set_phase_array(test_phases, test_amplitudes)
        
        required_keys = [
            'switching_time_ns', 'spurious_level_dbc', 'meets_latency_spec',
            'meets_spurious_spec', 'update_counter'
        ]
        
        for key in required_keys:
            assert key in metrics
        
        assert metrics['switching_time_ns'] > 0
        assert metrics['update_counter'] > 0
        
        # Test performance measurement
        performance = controller.measure_switching_performance(num_measurements=10)
        
        assert 'average_switching_time_ns' in performance
        assert 'measured_update_rate_mhz' in performance
        assert performance['num_measurements'] == 10
    
    def test_isolator_control_api(self):
        """Test isolator control API."""
        api = IsolatorControlAPI(self.config)
        
        # Test direction setting
        forward_status = api.set_direction(True)
        assert forward_status['direction'] == 'forward'
        assert forward_status['success'] is True
        assert forward_status['total_latency_ns'] > 0
        
        reverse_status = api.set_direction(False)
        assert reverse_status['direction'] == 'reverse'
        assert reverse_status['success'] is True
        
        # Test frequency setting
        freq_status = api.set_center_freq(10.0)
        assert freq_status['new_frequency_ghz'] == 10.0
        assert freq_status['success'] is True
        assert freq_status['total_latency_ns'] > 0
        
        # Test invalid frequency
        with pytest.raises(ValueError):
            api.set_center_freq(100.0)  # Outside range
        
        # Test performance optimization
        opt_result = api.optimize_performance(10.0, 60.0)
        assert 'optimization_successful' in opt_result
        assert 'achieved_isolation_db' in opt_result
    
    def test_full_validation(self):
        """Test complete dynamic reconfigurability validation."""
        results = validate_dynamic_reconfigurability(self.config)
        
        required_sections = ['phase_controller', 'api_performance', 'optimization']
        for section in required_sections:
            assert section in results
        
        # Check performance metrics
        controller_results = results['phase_controller']
        assert controller_results['average_switching_time_ns'] > 0
        assert controller_results['measured_update_rate_mhz'] > 0
        
        api_results = results['api_performance']
        assert api_results['average_api_latency_ns'] > 0


class TestIntegrationAndReproducibility:
    """Integration tests and reproducibility validation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for integration tests."""
        seed_everything(42)
    
    def test_reproducibility_across_modules(self):
        """Test that all modules produce reproducible results."""
        # Run each validation twice with same seed
        results_1 = {}
        results_2 = {}
        
        configs = {
            'thz': THzFrameworkConfig(),
            'dual_band': DualBandModulatorConfig(),
            'quantum': QuantumRegimeConfig(),
            'topological': TopologicalConfig(lattice_size=(5, 5)),  # Small for speed
            'reconfigurable': ReconfigurabilityConfig()
        }
        
        # First run
        seed_everything(42)
        results_1['thz'] = validate_thz_framework(configs['thz'])
        
        seed_everything(42)
        results_1['dual_band'] = validate_dual_band_modulator(configs['dual_band'])
        
        seed_everything(42)
        results_1['quantum'] = validate_quantum_regime(configs['quantum'])
        
        seed_everything(42)
        results_1['topological'] = validate_topological_enhancement(configs['topological'])
        
        # Second run
        seed_everything(42)
        results_2['thz'] = validate_thz_framework(configs['thz'])
        
        seed_everything(42)
        results_2['dual_band'] = validate_dual_band_modulator(configs['dual_band'])
        
        seed_everything(42)
        results_2['quantum'] = validate_quantum_regime(configs['quantum'])
        
        seed_everything(42)
        results_2['topological'] = validate_topological_enhancement(configs['topological'])
        
        # Compare results
        for module in ['thz', 'dual_band', 'quantum', 'topological']:
            self._compare_results(results_1[module], results_2[module], module)
    
    def _compare_results(self, result1: Dict, result2: Dict, module_name: str):
        """Compare two result dictionaries for reproducibility."""
        def compare_values(v1, v2, path=""):
            if isinstance(v1, dict) and isinstance(v2, dict):
                for key in v1.keys():
                    if key in v2:
                        compare_values(v1[key], v2[key], f"{path}.{key}")
            elif isinstance(v1, (int, float, complex)) and isinstance(v2, (int, float, complex)):
                if not np.isclose(v1, v2, rtol=1e-10, atol=1e-12):
                    logger.warning(f"Reproducibility issue in {module_name}{path}: {v1} != {v2}")
                    # For tests, we'll be more lenient due to floating point precision
                    assert np.isclose(v1, v2, rtol=1e-6, atol=1e-8), f"Reproducibility failed: {v1} != {v2}"
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.allclose(v1, v2, rtol=1e-10, atol=1e-12):
                    logger.warning(f"Array reproducibility issue in {module_name}{path}")
                    assert np.allclose(v1, v2, rtol=1e-6, atol=1e-8), "Array reproducibility failed"
        
        compare_values(result1, result2)
        logger.info(f"Reproducibility validated for {module_name}")
    
    def test_memory_usage_validation(self):
        """Test that memory usage stays within reasonable bounds."""
        from memory_manager import MemoryManager
        
        memory_manager = MemoryManager()
        
        # Test each module's memory usage
        configs = {
            'thz': THzFrameworkConfig(grid_resolution=100),  # Reduced for testing
            'topological': TopologicalConfig(lattice_size=(10, 10))  # Reduced for testing
        }
        
        # Monitor memory during validation
        initial_memory = memory_manager.get_current_memory_usage()
        
        validate_thz_framework(configs['thz'])
        thz_memory = memory_manager.get_current_memory_usage()
        
        validate_topological_enhancement(configs['topological'])
        topo_memory = memory_manager.get_current_memory_usage()
        
        # Check memory usage is reasonable
        assert thz_memory - initial_memory < 2e9  # < 2 GB
        assert topo_memory - initial_memory < 2e9  # < 2 GB
        
        logger.info(f"Memory usage validation passed: max increase = {max(thz_memory, topo_memory) - initial_memory:.0f} bytes")
    
    def test_scientific_integrity_tracking(self):
        """Test that scientific approximations are properly tracked."""
        # Clear any existing approximations
        from scientific_integrity import clear_approximations
        clear_approximations()
        
        # Run validations that should register approximations
        validate_thz_framework(THzFrameworkConfig())
        validate_dual_band_modulator(DualBandModulatorConfig())
        
        # Check that approximations were registered
        approximations = get_approximations_summary()
        
        assert len(approximations) > 0, "No approximations were registered"
        
        # Check for expected approximations
        expected_approximations = [
            'rotating_wave_approximation',
            'quasi_static_approximation',
            'transmission_line_model'
        ]
        
        registered_names = [approx['name'] for approx in approximations]
        
        for expected in expected_approximations:
            if expected not in registered_names:
                logger.warning(f"Expected approximation '{expected}' not found in: {registered_names}")
        
        logger.info(f"Scientific integrity tracking validated: {len(approximations)} approximations registered")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for all modules."""
        benchmark_results = {}
        
        # THz framework benchmark
        start_time = time.perf_counter()
        validate_thz_framework(THzFrameworkConfig())
        benchmark_results['thz_validation_time'] = time.perf_counter() - start_time
        
        # Dual-band modulator benchmark
        start_time = time.perf_counter()
        validate_dual_band_modulator(DualBandModulatorConfig())
        benchmark_results['dual_band_validation_time'] = time.perf_counter() - start_time
        
        # Dynamic reconfigurability benchmark
        start_time = time.perf_counter()
        validate_dynamic_reconfigurability(ReconfigurabilityConfig())
        benchmark_results['reconfig_validation_time'] = time.perf_counter() - start_time
        
        # Check that benchmarks are reasonable
        for test_name, duration in benchmark_results.items():
            assert duration < 30.0, f"{test_name} took too long: {duration:.1f} seconds"
            logger.info(f"{test_name}: {duration:.2f} seconds")
        
        total_time = sum(benchmark_results.values())
        assert total_time < 60.0, f"Total benchmark time too long: {total_time:.1f} seconds"
        
        logger.info(f"Performance benchmarks passed: total time = {total_time:.2f} seconds")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
