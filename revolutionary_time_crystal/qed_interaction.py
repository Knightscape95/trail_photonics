"""
Quantum Electrodynamics Interaction Hamiltonian Implementation

This module implements the exact interaction Hamiltonian from Equation (9) 
of the supplementary information:

$$
\\hat{H}_{\\text{int},I}(t)= -\\frac{\\varepsilon_0}{2}\\int d^{3}r\\,\\delta\\chi(\\mathbf r,t)\\,\\hat{\\mathbf E}_{I}^{2}(\\mathbf r,t)
$$

Author: Revolutionary Time Crystal Team
Date: July 17, 2025
"""

import numpy as np
from scipy.constants import epsilon_0
from typing import Union


def interaction_hamiltonian_eq9(E_I: np.ndarray,
                                delta_chi: np.ndarray,
                                dV: float) -> float:
    """
    Exact Eq. (9) interaction Hamiltonian (J).
    
    Implements the interaction Hamiltonian from the supplementary information:
    
    $$
    \\hat{H}_{\\text{int},I}(t)= -\\frac{\\varepsilon_0}{2}\\int d^{3}r\\,\\delta\\chi(\\mathbf r,t)\\,\\hat{\\mathbf E}_{I}^{2}(\\mathbf r,t)
    $$
    
    Parameters
    ----------
    E_I : np.ndarray
        Electric field operator samples on grid, shape (Nx, Ny, Nz, 3)
        Units: V/m
    delta_chi : np.ndarray  
        Modulation depth on grid, shape (Nx, Ny, Nz)
        Units: dimensionless
    dV : float
        Scalar volume element Δx·Δy·Δz
        Units: m³
        
    Returns
    -------
    float
        Interaction Hamiltonian energy in Joules
        
    Raises
    ------
    ValueError
        If grid shapes don't match or inputs are invalid
    TypeError
        If inputs are not numeric arrays
        
    Examples
    --------
    >>> import numpy as np
    >>> # 1D slab example
    >>> E_field = np.zeros((10, 1, 1, 3))
    >>> E_field[:, 0, 0, 0] = 1e6  # 1 MV/m in x-direction
    >>> delta_chi_slab = np.ones((10, 1, 1)) * 0.1  # 10% modulation
    >>> dV = 1e-9  # 1 nm³ volume elements
    >>> H_int = interaction_hamiltonian_eq9(E_field, delta_chi_slab, dV)
    """
    # Input validation and type checking
    if not isinstance(E_I, np.ndarray) or not isinstance(delta_chi, np.ndarray):
        raise TypeError("E_I and delta_chi must be numpy arrays")
    
    if not isinstance(dV, (int, float)) or dV <= 0:
        raise ValueError("dV must be a positive scalar")
    
    # Promote to float64 for numerical precision
    E_I = np.asarray(E_I, dtype=np.float64)
    delta_chi = np.asarray(delta_chi, dtype=np.float64)
    dV = float(dV)
    
    # Validate grid shapes
    if E_I.ndim != 4 or E_I.shape[-1] != 3:
        raise ValueError(f"E_I must have shape (Nx, Ny, Nz, 3), got {E_I.shape}")
    
    if delta_chi.ndim != 3:
        raise ValueError(f"delta_chi must have shape (Nx, Ny, Nz), got {delta_chi.shape}")
    
    spatial_shape_E = E_I.shape[:-1]  # (Nx, Ny, Nz)
    spatial_shape_chi = delta_chi.shape  # (Nx, Ny, Nz)
    
    if spatial_shape_E != spatial_shape_chi:
        raise ValueError(f"Spatial grid shapes must match: E_I {spatial_shape_E} vs delta_chi {spatial_shape_chi}")
    
    # Physics-exact implementation following Eq. (9)
    # Compute |E|² using optimized numpy operations for large arrays
    E_squared = np.sum(E_I * E_I, axis=-1)
    
    # Energy density: -½ε₀ δχ |E|² with pre-computed constant
    const_factor = -0.5 * epsilon_0 * dV
    total_energy = const_factor * np.sum(delta_chi * E_squared)
    
    return total_energy
