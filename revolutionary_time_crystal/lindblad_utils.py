"""
lindblad_utils.py

Complete Lindblad master-equation module for quantum system simulation.
Supports arbitrary device Hamiltonians with time-dependent controls and
comprehensive decoherence modeling.

Requirements: Python 3.8+, QuTiP 4.7+, NumPy
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Tuple, Callable, Optional
import warnings

def build_collapse_list(params: dict, a_ops: list[qt.Qobj]) -> list[qt.Qobj]:
    """
    Build collapse operators for comprehensive decoherence modeling.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing decoherence parameters with keys:
        - 'T1': list[float] - Energy relaxation times (seconds) for each mode
        - 'Tphi': list[float] - Pure dephasing times (seconds) for each mode  
        - 'n_th': list[float] - Thermal photon numbers for each mode
        - 'kappa_2ph': list[float] - Two-photon loss rates (Hz) for each mode
        - 'kappa_pur': float - Purcell leakage rate (Hz)
        - 'gamma_comm': float - Collective 1/f dephasing rate (Hz)
        - 'hop_loss': dict - Cross-mode hopping loss {(i,j): rate} (Hz)
        
    a_ops : list[qt.Qobj]
        List of annihilation operators for each mode
        
    Returns:
    --------
    list[qt.Qobj]
        List of collapse operators for Lindblad master equation
    """
    
    N = len(a_ops)
    c_ops = []
    
    # Validate parameters
    required_keys = ['T1', 'Tphi', 'n_th', 'kappa_2ph', 'kappa_pur', 'gamma_comm', 'hop_loss']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Local decoherence channels for each mode
    for i in range(N):
        # 1. Energy relaxation (T1)
        if i < len(params['T1']) and params['T1'][i] > 0:
            gamma_1 = 1.0 / params['T1'][i]
            c_ops.append(np.sqrt(gamma_1) * a_ops[i])
            
        # 2. Thermal excitation
        if (i < len(params['T1']) and params['T1'][i] > 0 and 
            i < len(params['n_th']) and params['n_th'][i] > 0):
            gamma_th = params['n_th'][i] / params['T1'][i]
            c_ops.append(np.sqrt(gamma_th) * a_ops[i].dag())
            
        # 3. Pure dephasing (Tφ)
        if i < len(params['Tphi']) and params['Tphi'][i] > 0:
            gamma_phi = 1.0 / params['Tphi'][i]
            c_ops.append(np.sqrt(gamma_phi) * a_ops[i].dag() * a_ops[i])
            
        # 4. Two-photon loss
        if i < len(params['kappa_2ph']) and params['kappa_2ph'][i] > 0:
            c_ops.append(np.sqrt(params['kappa_2ph'][i]) * a_ops[i] * a_ops[i])
    
    # Collective decoherence channels
    
    # 5. Purcell leakage (collective coupling to environment)
    if params['kappa_pur'] > 0:
        a_collective = sum(a_ops)
        c_ops.append(np.sqrt(params['kappa_pur']) * a_collective)
    
    # 6. Collective 1/f dephasing
    if params['gamma_comm'] > 0:
        collective_dephasing = sum(a_op.dag() * a_op for a_op in a_ops)
        c_ops.append(np.sqrt(params['gamma_comm']) * collective_dephasing)
    
    # 7. Cross-mode hopping loss
    for (i, j), gamma_ij in params['hop_loss'].items():
        if gamma_ij > 0 and i < N and j < N:
            c_ops.append(np.sqrt(gamma_ij) * (a_ops[i] - a_ops[j]))
    
    return c_ops

def propagate_rho(H0: qt.Qobj,
                  controls: list[tuple[qt.Qobj, Callable[[float], float]]],
                  rho0: qt.Qobj,
                  tlist: np.ndarray,
                  c_ops: list[qt.Qobj]) -> tuple[list[qt.Qobj], qt.Qobj]:
    """
    Propagate density matrix using Lindblad master equation.
    
    Parameters:
    -----------
    H0 : qt.Qobj
        Static device Hamiltonian
    controls : list[tuple[qt.Qobj, Callable[[float], float]]]
        List of (control_hamiltonian, control_function) pairs
    rho0 : qt.Qobj
        Initial density matrix
    tlist : np.ndarray
        Time points for evolution
    c_ops : list[qt.Qobj]
        Collapse operators from build_collapse_list()
        
    Returns:
    --------
    tuple[list[qt.Qobj], qt.Qobj]
        (trajectory, final_density_matrix)
    """
    
    # Build time-dependent Hamiltonian
    if controls:
        H_t = [H0]
        for H_ctrl, u_func in controls:
            H_t.append([H_ctrl, u_func])
    else:
        H_t = H0
    
    # Solve master equation
    result = qt.mesolve(H_t, rho0, tlist, c_ops, [])
    
    # Extract trajectory and final state
    trajectory = result.states
    final_rho = result.states[-1]
    
    return trajectory, final_rho

def fidelity(rho: qt.Qobj, target_psi: qt.Qobj) -> float:
    """
    Calculate Uhlmann fidelity between density matrix and target state.
    
    Parameters:
    -----------
    rho : qt.Qobj
        Density matrix (final state)
    target_psi : qt.Qobj
        Target state vector
        
    Returns:
    --------
    float
        Uhlmann fidelity (0 to 1)
    """
    
    # Convert target state to density matrix if needed
    if target_psi.type == 'ket':
        target_rho = target_psi.proj()
    else:
        target_rho = target_psi
    
    # Calculate Uhlmann fidelity
    # For pure target state, simplifies to: F = <psi|rho|psi>
    if target_psi.type == 'ket':
        fid = qt.expect(target_rho, rho)
    else:
        fid = qt.fidelity(rho, target_rho)
    
    return float(np.real(fid))

def demo_5_mode_system():
    """
    Demonstrate 5-mode system with state transfer and decoherence.
    """
    
    print("=" * 60)
    print("5-Mode Lindblad Master Equation Demo")
    print("=" * 60)
    
    # System parameters
    N = 5  # Number of modes
    print(f"System: {N} modes")
    
    # Create annihilation operators
    a_ops = [qt.destroy(N, i) for i in range(N)]
    
    # Build static Hamiltonian (nearest-neighbor coupling)
    H0 = qt.Qobj(dims=[[N], [N]])
    omega = 2 * np.pi * 10e9  # 10 GHz transition frequency
    g = 2 * np.pi * 100e6     # 100 MHz coupling
    
    for i in range(N):
        H0 += omega * a_ops[i].dag() * a_ops[i]  # Mode frequencies
        if i < N - 1:
            H0 += g * (a_ops[i].dag() * a_ops[i+1] + a_ops[i+1].dag() * a_ops[i])
    
    print(f"Static Hamiltonian: {N} modes at {omega/(2*np.pi*1e9):.1f} GHz")
    print(f"Coupling strength: {g/(2*np.pi*1e6):.1f} MHz")
    
    # Time-dependent control (STIRAP-like)
    def stokes_pulse(t, args):
        t_center = 50e-9
        t_width = 20e-9
        return np.exp(-((t - t_center) / t_width)**2)
    
    def pump_pulse(t, args):
        t_center = 100e-9
        t_width = 20e-9
        return np.exp(-((t - t_center) / t_width)**2)
    
    # Control Hamiltonians
    H_stokes = 2 * np.pi * 50e6 * (a_ops[0] + a_ops[0].dag())
    H_pump = 2 * np.pi * 50e6 * (a_ops[-1] + a_ops[-1].dag())
    
    controls = [
        (H_stokes, stokes_pulse),
        (H_pump, pump_pulse)
    ]
    
    print("Control: STIRAP-like pulse sequence")
    
    # Decoherence parameters
    params = {
        'T1': [100e-6] * N,        # 100 μs relaxation time
        'Tphi': [200e-6] * N,      # 200 μs pure dephasing time
        'n_th': [0.01] * N,        # Thermal photon number (cold)
        'kappa_2ph': [0.0] * N,    # No two-photon loss
        'kappa_pur': 2 * np.pi * 1e3,  # 1 kHz Purcell leakage
        'gamma_comm': 2 * np.pi * 500,  # 500 Hz collective dephasing
        'hop_loss': {(0, 1): 2 * np.pi * 100}  # 100 Hz hopping loss
    }
    
    print("Decoherence parameters:")
    print(f"  T1 = {params['T1'][0]*1e6:.0f} μs")
    print(f"  Tφ = {params['Tphi'][0]*1e6:.0f} μs")
    print(f"  Thermal photons = {params['n_th'][0]:.3f}")
    print(f"  Purcell rate = {params['kappa_pur']/(2*np.pi*1e3):.1f} kHz")
    
    # Build collapse operators
    c_ops = build_collapse_list(params, a_ops)
    print(f"Collapse operators: {len(c_ops)} terms")
    
    # Initial and target states
    rho0 = qt.basis(N, 0).proj()  # Start in mode 0
    target_psi = qt.basis(N, N-1)  # Target: mode N-1
    
    print(f"Initial state: |0⟩ (mode 0)")
    print(f"Target state: |{N-1}⟩ (mode {N-1})")
    
    # Time evolution
    t_total = 200e-9  # 200 ns
    tlist = np.linspace(0, t_total, 1001)
    
    print(f"Evolution time: {t_total*1e9:.0f} ns")
    print("Solving master equation...")
    
    # Propagate density matrix
    trajectory, final_rho = propagate_rho(H0, controls, rho0, tlist, c_ops)
    
    # Calculate fidelity
    final_fidelity = fidelity(final_rho, target_psi)
    
    print(f"\nResults:")
    print(f"Final fidelity: {final_fidelity:.4f}")
    print(f"Population in target mode: {qt.expect(a_ops[-1].dag() * a_ops[-1], final_rho):.4f}")
    print(f"Total population: {final_rho.tr():.4f}")
    
    # Population dynamics
    populations = np.zeros((len(tlist), N))
    for i, rho_t in enumerate(trajectory):
        for j in range(N):
            populations[i, j] = qt.expect(a_ops[j].dag() * a_ops[j], rho_t)
    
    print(f"\nPopulation transfer:")
    print(f"  Initial: mode 0 = {populations[0, 0]:.3f}")
    print(f"  Final: mode {N-1} = {populations[-1, -1]:.3f}")
    print(f"  Transfer efficiency: {populations[-1, -1]/populations[0, 0]:.1%}")
    
    # Decoherence analysis
    coherent_fidelity = populations[-1, -1]  # Population-based estimate
    decoherence_impact = 1 - final_fidelity/coherent_fidelity if coherent_fidelity > 0 else 0
    print(f"\nDecoherence impact: {decoherence_impact:.1%} fidelity reduction")
    
    return final_fidelity, trajectory

if __name__ == "__main__":
    # Run demonstration
    final_fidelity, trajectory = demo_5_mode_system()
    
    print(f"\nDemo completed successfully!")
    print(f"Achieved fidelity: {final_fidelity:.4f}")
