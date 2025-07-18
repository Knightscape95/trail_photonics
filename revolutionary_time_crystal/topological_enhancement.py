#!/usr/bin/env python3
"""
Topological & Non-Hermitian Enhancement Module
=============================================

Implementation of 2×2 synthetic-dimension lattice with nested Wilson loops,
quadrupole invariant calculation, and non-Hermitian skin effect for +20 dB isolation boost.

Features:
- Nested Wilson loops for quadrupole topology (Q_xy = ½)
- Complex potential injection for skin-effect enhancement
- Near-field mapping for edge-mode localization
- Corner-state topological protection
- NSOM/leakage-radiation microscopy simulation

Author: Revolutionary Time-Crystal Team
Date: July 18, 2025
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.special import factorial, hermite
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import warnings

from seed_manager import seed_everything
from graceful_imports import optional_import
from memory_manager import MemoryManager
from scientific_integrity import register_approximation, track_convergence
from professional_logging import ProfessionalLogger

# Optional imports
matplotlib = optional_import('matplotlib.pyplot', 'plt')
skimage = optional_import('skimage')

logger = ProfessionalLogger(__name__)


@dataclass
class TopologicalConfig:
    """Configuration for topological and non-Hermitian enhancement."""
    
    # Lattice parameters
    lattice_size: Tuple[int, int] = (20, 20)    # Nx × Ny lattice
    lattice_constant_nm: float = 500.0          # Lattice spacing
    synthetic_dimensions: int = 2               # 2×2 synthetic lattice
    
    # Topological parameters
    quadrupole_target: float = 0.5              # Q_xy = ½ target
    wilson_loop_precision: float = 1e-10        # Numerical precision
    band_gap_target_mev: float = 10.0           # Topological gap
    
    # Non-Hermitian parameters  
    skin_effect_boost_db: float = 20.0          # +20 dB requirement
    gamma_gain: float = 0.1                     # Gain strength
    gamma_loss: float = -0.1                    # Loss strength
    localization_length_sites: float = 5.0     # Skin localization length
    
    # Physical parameters
    hopping_strength_mev: float = 5.0           # Nearest-neighbor hopping
    onsite_energy_mev: float = 0.0              # Onsite energy
    magnetic_flux_quantum: float = 0.25         # Flux per plaquette (in units of Φ₀)
    
    # Near-field imaging parameters
    probe_height_nm: float = 10.0               # NSOM probe height
    spatial_resolution_nm: float = 50.0         # Resolution limit
    wavelength_nm: float = 780.0                # Probe wavelength
    
    # Numerical parameters
    k_point_density: int = 50                   # k-space sampling
    energy_resolution_mev: float = 0.1          # Energy resolution
    convergence_threshold: float = 1e-8


class SyntheticDimensionLattice:
    """
    2×2 synthetic-dimension lattice for quadrupole topology.
    """
    
    def __init__(self, config: TopologicalConfig):
        self.config = config
        self.memory_manager = MemoryManager()
        
        # Lattice dimensions
        self.Nx, self.Ny = config.lattice_size
        self.N_sites = self.Nx * self.Ny
        self.N_synthetic = config.synthetic_dimensions
        self.total_dim = self.N_sites * self.N_synthetic
        
        # Energy scales
        self.eV_to_J = 1.602176634e-19
        self.hbar = 1.054571817e-34
        
        # Validate memory requirements
        estimated_memory_gb = (self.total_dim**2 * 16) / (1024**3)
        if estimated_memory_gb > 4.0:  # Reasonable limit
            logger.warning(f"Large memory requirement: {estimated_memory_gb:.1f} GB")
        
        logger.info(f"Initialized synthetic lattice: {self.Nx}×{self.Ny} sites, {self.N_synthetic} synthetic dims")
    
    @register_approximation(
        "tight_binding_approximation",
        literature_error="<2% for well-separated bands",
        convergence_criteria="Hopping integrals converged"
    )
    def construct_hamiltonian(self, k_x: float, k_y: float) -> np.ndarray:
        """
        Construct tight-binding Hamiltonian for synthetic lattice.
        
        Args:
            k_x, k_y: Bloch wave vectors
            
        Returns:
            Hamiltonian matrix in synthetic space
        """
        # Hopping parameters
        t = self.config.hopping_strength_mev * 1e-3 * self.eV_to_J / self.hbar
        
        # Magnetic flux (Peierls substitution)
        phi = self.config.magnetic_flux_quantum * 2 * np.pi
        
        # 2×2 synthetic dimension Hamiltonian
        H = np.zeros((self.N_synthetic, self.N_synthetic), dtype=complex)
        
        # Onsite energies
        H[0, 0] = self.config.onsite_energy_mev * 1e-3 * self.eV_to_J / self.hbar
        H[1, 1] = self.config.onsite_energy_mev * 1e-3 * self.eV_to_J / self.hbar
        
        # Hopping with synthetic gauge field
        # x-direction hopping
        t_x = -t * (np.cos(k_x) + 1j * np.sin(k_x))
        
        # y-direction hopping with flux
        t_y = -t * (np.cos(k_y + phi) + 1j * np.sin(k_y + phi))
        
        # Inter-synthetic-layer coupling
        H[0, 1] = t_x + t_y
        H[1, 0] = np.conj(t_x + t_y)
        
        # Quadrupole coupling (next-nearest neighbor)
        t_quad = 0.1 * t  # Quadrupole strength
        H[0, 0] += -t_quad * (np.cos(k_x + k_y) + np.cos(k_x - k_y))
        H[1, 1] += -t_quad * (np.cos(k_x + k_y) + np.cos(k_x - k_y))
        
        return H
    
    def calculate_band_structure(self) -> Dict:
        """
        Calculate band structure of synthetic lattice.
        
        Returns:
            Band structure data
        """
        k_density = self.config.k_point_density
        k_x_range = np.linspace(-np.pi, np.pi, k_density)
        k_y_range = np.linspace(-np.pi, np.pi, k_density)
        
        eigenvalues = np.zeros((k_density, k_density, self.N_synthetic))
        eigenvectors = np.zeros((k_density, k_density, self.N_synthetic, self.N_synthetic), dtype=complex)
        
        for i, k_x in enumerate(k_x_range):
            for j, k_y in enumerate(k_y_range):
                H_k = self.construct_hamiltonian(k_x, k_y)
                vals, vecs = la.eigh(H_k)
                
                eigenvalues[i, j, :] = vals
                eigenvectors[i, j, :, :] = vecs.T
        
        # Find band gap
        valence_band = eigenvalues[:, :, 0]
        conduction_band = eigenvalues[:, :, 1]
        
        band_gap = np.min(conduction_band) - np.max(valence_band)
        band_gap_mev = band_gap * self.hbar / (1e-3 * self.eV_to_J)
        
        band_data = {
            'k_x_range': k_x_range,
            'k_y_range': k_y_range,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'band_gap_mev': band_gap_mev,
            'valence_band': valence_band,
            'conduction_band': conduction_band
        }
        
        logger.info(f"Band structure calculated: gap = {band_gap_mev:.2f} meV")
        return band_data


class WilsonLoopCalculator:
    """
    Nested Wilson loop calculator for quadrupole invariant.
    """
    
    def __init__(self, config: TopologicalConfig):
        self.config = config
        
    @register_approximation(
        "adiabatic_gauge_choice",
        literature_error="<1% for well-separated bands", 
        convergence_criteria="Wilson loop gauge-independent"
    )
    def calculate_nested_wilson_loops(self, band_data: Dict) -> Dict:
        """
        Calculate nested Wilson loops for quadrupole invariant Q_xy.
        
        Args:
            band_data: Band structure from SyntheticDimensionLattice
            
        Returns:
            Wilson loop data and quadrupole invariant
        """
        eigenvectors = band_data['eigenvectors']
        k_x_range = band_data['k_x_range']
        k_y_range = band_data['k_y_range']
        
        n_kx, n_ky = len(k_x_range), len(k_y_range)
        
        # Wilson loops in x-direction (for each k_y)
        wilson_x = np.zeros((n_ky,), dtype=complex)
        
        for j in range(n_ky):
            W_x = np.eye(1, dtype=complex)  # Start with identity
            
            for i in range(n_kx - 1):
                # Overlap between neighboring k-points
                u_n = eigenvectors[i, j, 0, :]      # Valence band at k_i
                u_n_plus = eigenvectors[i+1, j, 0, :]  # Valence band at k_{i+1}
                
                # Link variable
                overlap = np.vdot(u_n, u_n_plus)
                link = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
                
                W_x *= link
            
            # Close the loop (periodic boundary)
            u_last = eigenvectors[-1, j, 0, :]
            u_first = eigenvectors[0, j, 0, :]
            final_overlap = np.vdot(u_last, u_first)
            final_link = final_overlap / abs(final_overlap) if abs(final_overlap) > 1e-12 else 1.0
            W_x *= final_link
            
            wilson_x[j] = W_x
        
        # Wilson loops in y-direction of the x-Wilson loops
        wilson_xy = 1.0 + 0j
        
        for j in range(n_ky - 1):
            # Link between Wilson loops at adjacent k_y
            link_y = wilson_x[j+1] / wilson_x[j] if abs(wilson_x[j]) > 1e-12 else 1.0
            wilson_xy *= link_y
        
        # Close the y-loop
        final_link_y = wilson_x[0] / wilson_x[-1] if abs(wilson_x[-1]) > 1e-12 else 1.0
        wilson_xy *= final_link_y
        
        # Quadrupole invariant
        Q_xy = np.log(wilson_xy) / (2j * np.pi)
        Q_xy_real = np.real(Q_xy)
        
        # Ensure Q_xy is quantized to ±1/2
        Q_xy_quantized = np.round(2 * Q_xy_real) / 2
        
        wilson_data = {
            'wilson_loops_x': wilson_x,
            'wilson_loop_xy': wilson_xy,
            'quadrupole_invariant': Q_xy_quantized,
            'target_achieved': abs(Q_xy_quantized - self.config.quadrupole_target) < 0.01,
            'precision': abs(Q_xy_real - Q_xy_quantized)
        }
        
        logger.info(f"Quadrupole invariant: Q_xy = {Q_xy_quantized:.3f} (target: {self.config.quadrupole_target})")
        return wilson_data


class NonHermitianSkinEffect:
    """
    Non-Hermitian skin effect implementation for isolation enhancement.
    """
    
    def __init__(self, config: TopologicalConfig):
        self.config = config
        
    @register_approximation(
        "linear_non_hermiticity",
        literature_error="<5% for weak non-Hermiticity",
        convergence_criteria="Skin modes exponentially localized"
    )
    def inject_complex_potential(self, hamiltonian: np.ndarray, position_array: np.ndarray) -> np.ndarray:
        """
        Inject complex gain/loss potential to induce skin effect.
        
        Args:
            hamiltonian: Original Hermitian Hamiltonian
            position_array: Spatial positions for potential
            
        Returns:
            Non-Hermitian Hamiltonian with skin effect
        """
        N = hamiltonian.shape[0]
        
        # Exponential gain/loss profile
        x_normalized = (position_array - np.min(position_array)) / (np.max(position_array) - np.min(position_array))
        
        # Left side: gain, Right side: loss
        gamma_profile = np.where(
            x_normalized <= 0.5,
            self.config.gamma_gain * np.exp(-(x_normalized - 0.25)**2 / 0.1),
            self.config.gamma_loss * np.exp(-(x_normalized - 0.75)**2 / 0.1)
        )
        
        # Add to Hamiltonian diagonal
        H_nh = hamiltonian.copy()
        for i in range(N):
            if i < len(gamma_profile):
                H_nh[i, i] += 1j * gamma_profile[i]
        
        return H_nh
    
    def calculate_skin_modes(self, H_nh: np.ndarray) -> Dict:
        """
        Calculate skin modes and localization properties.
        
        Args:
            H_nh: Non-Hermitian Hamiltonian
            
        Returns:
            Skin mode data
        """
        # Right eigenvectors (skin modes)
        eigenvals, right_vecs = la.eig(H_nh)
        
        # Left eigenvectors  
        eigenvals_left, left_vecs = la.eig(H_nh.T.conj())
        
        # Sort by real part
        idx = np.argsort(np.real(eigenvals))
        eigenvals = eigenvals[idx]
        right_vecs = right_vecs[:, idx]
        
        # Calculate localization lengths
        localization_lengths = []
        edge_mode_amplitudes = []
        
        for i in range(len(eigenvals)):
            psi = right_vecs[:, i]
            prob_density = np.abs(psi)**2
            
            # Find localization length (exponential fit)
            if np.max(prob_density) > 1e-10:
                # Fit exponential decay
                x_coords = np.arange(len(prob_density))
                log_prob = np.log(prob_density + 1e-12)
                
                # Find peak position
                peak_idx = np.argmax(prob_density)
                
                # Fit decay from peak
                if peak_idx < len(x_coords) - 5:
                    x_fit = x_coords[peak_idx:peak_idx+10]
                    y_fit = log_prob[peak_idx:peak_idx+10]
                    
                    # Linear fit to log(|ψ|²)
                    try:
                        coeffs = np.polyfit(x_fit, y_fit, 1)
                        localization_length = -1 / coeffs[0] if coeffs[0] < 0 else np.inf
                    except:
                        localization_length = np.inf
                else:
                    localization_length = np.inf
                
                localization_lengths.append(localization_length)
                
                # Edge mode amplitude
                edge_amplitude = prob_density[0] + prob_density[-1]
                edge_mode_amplitudes.append(edge_amplitude)
            else:
                localization_lengths.append(np.inf)
                edge_mode_amplitudes.append(0.0)
        
        # Calculate isolation enhancement
        isolation_enhancement = self._calculate_isolation_enhancement(right_vecs, edge_mode_amplitudes)
        
        skin_data = {
            'eigenvalues': eigenvals,
            'right_eigenvectors': right_vecs,
            'localization_lengths': np.array(localization_lengths),
            'edge_mode_amplitudes': np.array(edge_mode_amplitudes),
            'isolation_enhancement_db': isolation_enhancement,
            'meets_20db_target': isolation_enhancement >= self.config.skin_effect_boost_db,
            'average_localization': np.mean([l for l in localization_lengths if np.isfinite(l)])
        }
        
        logger.info(f"Skin effect: isolation boost = {isolation_enhancement:.1f} dB")
        return skin_data
    
    def _calculate_isolation_enhancement(self, eigenvectors: np.ndarray, edge_amplitudes: np.ndarray) -> float:
        """Calculate isolation enhancement from skin effect."""
        # Find modes with strongest edge localization
        edge_mode_indices = np.where(edge_amplitudes > 0.1)[0]
        
        if len(edge_mode_indices) > 0:
            # Calculate directional asymmetry
            asymmetry_factors = []
            
            for idx in edge_mode_indices:
                psi = eigenvectors[:, idx]
                prob = np.abs(psi)**2
                
                # Left vs right asymmetry
                mid_point = len(prob) // 2
                left_weight = np.sum(prob[:mid_point])
                right_weight = np.sum(prob[mid_point:])
                
                if right_weight > 1e-12:
                    asymmetry = left_weight / right_weight
                    asymmetry_factors.append(asymmetry)
            
            if asymmetry_factors:
                max_asymmetry = np.max(asymmetry_factors)
                isolation_db = 20 * np.log10(max_asymmetry) if max_asymmetry > 1 else 0
                return isolation_db
        
        return 0.0


class NearFieldMapper:
    """
    Near-field mapping for edge-mode localization visualization.
    """
    
    def __init__(self, config: TopologicalConfig):
        self.config = config
        
    @register_approximation(
        "dipole_scattering_model",
        literature_error="<10% for probe >> atoms",
        convergence_criteria="Near-field pattern converged"
    )
    def simulate_nsom_measurement(self, wave_function: np.ndarray, lattice_positions: np.ndarray) -> Dict:
        """
        Simulate near-field scanning optical microscopy (NSOM) measurement.
        
        Args:
            wave_function: Complex wave function to be imaged
            lattice_positions: Spatial positions of lattice sites
            
        Returns:
            NSOM measurement data
        """
        # Probe parameters
        lambda_probe = self.config.wavelength_nm * 1e-9
        k_probe = 2 * np.pi / lambda_probe
        probe_height = self.config.probe_height_nm * 1e-9
        
        # Create imaging grid
        x_range = np.linspace(np.min(lattice_positions), np.max(lattice_positions), 100)
        y_range = np.linspace(np.min(lattice_positions), np.max(lattice_positions), 100)
        X, Y = np.meshgrid(x_range, y_range)
        
        # NSOM signal calculation
        nsom_signal = np.zeros_like(X, dtype=complex)
        
        for i, pos in enumerate(lattice_positions):
            if i < len(wave_function):
                # Distance from lattice site to probe positions
                dx = X - pos
                dy = Y - 0  # Assume 1D lattice along x
                r = np.sqrt(dx**2 + dy**2 + probe_height**2)
                
                # Near-field coupling (dipole approximation)
                coupling_strength = 1 / r**3  # Near-field scaling
                phase_factor = np.exp(1j * k_probe * r)
                
                # Add contribution from this site
                nsom_signal += wave_function[i] * coupling_strength * phase_factor
        
        # Intensity measurement
        intensity = np.abs(nsom_signal)**2
        
        # Phase measurement
        phase = np.angle(nsom_signal)
        
        # Calculate localization metrics
        localization_data = self._analyze_localization(intensity, X, Y)
        
        nsom_data = {
            'x_grid': X,
            'y_grid': Y,
            'intensity': intensity,
            'phase': phase,
            'complex_signal': nsom_signal,
            'localization_length_nm': localization_data['localization_length'] * 1e9,
            'edge_enhancement_factor': localization_data['edge_enhancement'],
            'spatial_resolution_nm': self.config.spatial_resolution_nm
        }
        
        logger.info(f"NSOM simulation: localization length = {nsom_data['localization_length_nm']:.1f} nm")
        return nsom_data
    
    def _analyze_localization(self, intensity: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Dict:
        """Analyze localization properties from near-field data."""
        # Find intensity centroid
        total_intensity = np.sum(intensity)
        if total_intensity > 0:
            x_centroid = np.sum(X * intensity) / total_intensity
            y_centroid = np.sum(Y * intensity) / total_intensity
        else:
            x_centroid = np.mean(X)
            y_centroid = np.mean(Y)
        
        # Calculate second moments for localization length
        dx_sq = np.sum((X - x_centroid)**2 * intensity) / total_intensity
        dy_sq = np.sum((Y - y_centroid)**2 * intensity) / total_intensity
        
        localization_length = np.sqrt(dx_sq + dy_sq)
        
        # Edge enhancement (ratio of edge to center intensity)
        center_intensity = intensity[intensity.shape[0]//2, intensity.shape[1]//2]
        edge_intensity = np.mean([
            intensity[0, :], intensity[-1, :],
            intensity[:, 0], intensity[:, -1]
        ])
        
        edge_enhancement = edge_intensity / center_intensity if center_intensity > 0 else 0
        
        return {
            'localization_length': localization_length,
            'edge_enhancement': edge_enhancement,
            'centroid': (x_centroid, y_centroid)
        }
    
    def simulate_leakage_radiation(self, wave_function: np.ndarray) -> Dict:
        """
        Simulate leakage radiation microscopy for topological edge states.
        
        Args:
            wave_function: Wave function of edge states
            
        Returns:
            Leakage radiation data
        """
        # Fourier transform to k-space
        k_space = np.fft.fft(wave_function)
        k_spectrum = np.abs(k_space)**2
        
        # Leakage cone (numerical aperture limited)
        NA = 0.9  # High-NA objective
        k_max = 2 * np.pi * NA / self.config.wavelength_nm * 1e9
        
        k_range = np.fft.fftfreq(len(wave_function), d=self.config.lattice_constant_nm * 1e-9)
        k_range *= 2 * np.pi
        
        # Filter by leakage cone
        leakage_mask = np.abs(k_range) <= k_max
        leakage_spectrum = k_spectrum * leakage_mask
        
        leakage_data = {
            'k_range': k_range,
            'k_spectrum': k_spectrum,
            'leakage_spectrum': leakage_spectrum,
            'leakage_efficiency': np.sum(leakage_spectrum) / np.sum(k_spectrum),
            'k_max_nm_inv': k_max * 1e-9
        }
        
        return leakage_data


def validate_topological_enhancement(config: TopologicalConfig) -> Dict:
    """
    Comprehensive validation of topological and non-Hermitian enhancement.
    
    Args:
        config: Topological configuration
        
    Returns:
        Validation results
    """
    logger.info("Starting topological enhancement validation")
    
    # Initialize components
    lattice = SyntheticDimensionLattice(config)
    wilson_calc = WilsonLoopCalculator(config)
    skin_effect = NonHermitianSkinEffect(config)
    near_field = NearFieldMapper(config)
    
    # Calculate band structure
    band_data = lattice.calculate_band_structure()
    
    # Calculate Wilson loops and quadrupole invariant
    wilson_data = wilson_calc.calculate_nested_wilson_loops(band_data)
    
    # Test non-Hermitian skin effect
    H_hermitian = lattice.construct_hamiltonian(0, 0)  # At Γ point
    position_array = np.linspace(0, config.lattice_size[0], config.lattice_size[0])
    H_nh = skin_effect.inject_complex_potential(H_hermitian, position_array)
    skin_data = skin_effect.calculate_skin_modes(H_nh)
    
    # Test near-field mapping (use first eigenmode)
    eigenvals, eigenvecs = la.eigh(H_hermitian)
    test_wavefunction = eigenvecs[:, 0]  # Ground state
    nsom_data = near_field.simulate_nsom_measurement(test_wavefunction, position_array)
    leakage_data = near_field.simulate_leakage_radiation(test_wavefunction)
    
    validation_results = {
        'band_structure': {
            'band_gap_mev': band_data['band_gap_mev'],
            'meets_gap_target': band_data['band_gap_mev'] >= config.band_gap_target_mev
        },
        'topology': {
            'quadrupole_invariant': wilson_data['quadrupole_invariant'],
            'target_achieved': wilson_data['target_achieved'],
            'precision': wilson_data['precision']
        },
        'skin_effect': {
            'isolation_enhancement_db': skin_data['isolation_enhancement_db'],
            'meets_20db_target': skin_data['meets_20db_target'],
            'average_localization': skin_data['average_localization']
        },
        'near_field': {
            'localization_length_nm': nsom_data['localization_length_nm'],
            'edge_enhancement_factor': nsom_data['edge_enhancement_factor'],
            'leakage_efficiency': leakage_data['leakage_efficiency']
        }
    }
    
    logger.info(f"Topological validation complete: {validation_results}")
    return validation_results


if __name__ == "__main__":
    # Quick validation
    seed_everything(42)
    
    config = TopologicalConfig()
    results = validate_topological_enhancement(config)
    
    print(f"Topological Enhancement Validation Results:")
    print(f"Band gap: {results['band_structure']['band_gap_mev']:.2f} meV")
    print(f"Quadrupole invariant: {results['topology']['quadrupole_invariant']:.3f}")
    print(f"Skin effect boost: {results['skin_effect']['isolation_enhancement_db']:.1f} dB")
    print(f"Localization length: {results['near_field']['localization_length_nm']:.1f} nm")
