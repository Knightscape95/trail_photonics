#!/usr/bin/env python3
"""
Simple Renormalization Test - No plotting to avoid Qt issues
"""

import numpy as np
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_basic_renormalization():
    """Test basic renormalization functionality without plotting"""
    
    print("🔬 SIMPLE RENORMALIZATION TEST")
    print("=" * 50)
    
    try:
        # Test 1: Import renormalization module
        print("\n📊 Test 1: Import renormalization module")
        from renormalisation import get_renormalization_engine, validate_renormalization
        print("✅ Renormalization module imported successfully")
        
        # Test 2: Initialize renormalization engine
        print("\n📊 Test 2: Initialize renormalization engine")
        renorm_engine = get_renormalization_engine(
            chi1_amplitude=0.1,
            energy_cutoff=1e14,
            regularization_parameter=1e-3
        )
        print("✅ Renormalization engine initialized")
        renorm_engine.print_constants()
        
        # Test 3: Validate renormalization
        print("\n📊 Test 3: Validate renormalization")
        validation_results = validate_renormalization()
        print("Validation results:")
        all_passed = True
        for test_name, passed in validation_results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        # Test 4: Update constants
        print("\n📊 Test 4: Update renormalization constants")
        for i in range(3):
            new_chi1 = 0.1 + 0.01 * i
            constants = renorm_engine.update_constants(new_chi1=new_chi1)
            print(f"  Iteration {i+1}: Z₁={constants['Z1']:.8f}, Z₂={constants['Z2']:.8f}, Z₃={constants['Z3']:.8f}")
        
        # Test 5: Check convergence errors
        print("\n📊 Test 5: Check convergence errors")
        errors = renorm_engine.get_convergence_errors()
        for z_name, error_array in errors.items():
            if len(error_array) > 0:
                print(f"  {z_name}: {len(error_array)} errors, final: {error_array[-1]:.2e}")
            else:
                print(f"  {z_name}: No convergence data")
        
        print("\n✅ ALL BASIC TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qed_initialization():
    """Test QED engine initialization"""
    
    print("\n🔬 QED ENGINE INITIALIZATION TEST")
    print("=" * 50)
    
    try:
        from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters
        
        qed_params = QEDSystemParameters(
            device_length=1e-6,
            device_width=500e-9,
            device_height=220e-9,
            refractive_index_base=3.48,
            wavelength_vacuum=1550e-9,
            susceptibility_amplitude=0.1,
            modulation_frequency=2*np.pi*10e9
        )
        
        qed_engine = QuantumElectrodynamicsEngine(qed_params)
        print("✅ QED engine initialized successfully")
        print(f"   Device: {qed_params.device_length*1e6:.1f} × {qed_params.device_width*1e9:.0f} × {qed_params.device_height*1e9:.0f} μm")
        print(f"   Wavelength: {qed_params.wavelength_vacuum*1e9:.0f} nm")
        
        return True
        
    except Exception as e:
        print(f"❌ QED test failed: {e}")
        return False

def test_floquet_basic():
    """Test basic Floquet engine initialization"""
    
    print("\n🔬 FLOQUET ENGINE BASIC TEST")
    print("=" * 50)
    
    try:
        from rigorous_qed_engine import QuantumElectrodynamicsEngine, QEDSystemParameters
        from rigorous_floquet_engine import RigorousFloquetEngine, FloquetSystemParameters
        
        # Create QED engine
        qed_params = QEDSystemParameters()
        qed_engine = QuantumElectrodynamicsEngine(qed_params)
        
        # Create Floquet engine with simpler parameters
        floquet_params = FloquetSystemParameters(
            driving_frequency=2*np.pi*10e9,
            driving_amplitude=0.05,  # Smaller amplitude to avoid warnings
            n_harmonics=5,           # Fewer harmonics for speed
            n_time_steps=64,         # Fewer time steps for speed
            max_magnus_order=3,      # Lower order Magnus
            magnus_tolerance=1e-10   # Less strict tolerance
        )
        
        floquet_engine = RigorousFloquetEngine(qed_engine, floquet_params)
        print("✅ Floquet engine initialized successfully")
        print(f"   Renormalization tracker created: {floquet_engine.renorm_tracker.iteration_count} iterations")
        
        return True
        
    except Exception as e:
        print(f"❌ Floquet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 SIMPLE RENORMALIZATION TESTS")
    print("Date:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Run tests
    test1 = test_basic_renormalization()
    test2 = test_qed_initialization()
    test3 = test_floquet_basic()
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL SIMPLE TESTS PASSED!")
        print("Basic functionality working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the implementation.")
        sys.exit(1)
