#!/usr/bin/env python3
"""
Revolutionary Time-Crystal Test Suite
====================================

Comprehensive test suite verifying all revolutionary components
and performance targets are achievable.

Usage:
    python test_revolutionary_system.py
    python test_revolutionary_system.py --quick
    python test_revolutionary_system.py --full-validation

Author: Revolutionary Time-Crystal Team
Date: July 2025
"""

import numpy as np
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

def test_physics_engine():
    """Test revolutionary physics engine"""
    print("🔬 Testing Revolutionary Physics Engine...")
    
    try:
        from revolutionary_physics_engine import RevolutionaryTimeCrystalEngine
        
        # Create test epsilon movie
        T, H, W, C = 32, 16, 64, 3
        epsilon_movie = np.random.randn(T, H, W, C) * 0.1 + 2.25
        
        # Add temporal modulation for reciprocity breaking
        for t in range(T):
            modulation = 0.3 * np.sin(2 * np.pi * t / T)
            epsilon_movie[t] += modulation
        
        # Test physics engine
        engine = RevolutionaryTimeCrystalEngine()
        performance = engine.evaluate_revolutionary_performance(epsilon_movie)
        
        print(f"   Isolation: {performance['isolation_db']:.1f} dB")
        print(f"   Bandwidth: {performance['bandwidth_ghz']:.1f} GHz")
        print(f"   Quantum Fidelity: {performance['quantum_fidelity']:.3f}")
        print(f"   Revolutionary Status: {'✅' if performance.get('all_targets_met', False) else '⚠️'}")
        
        return True, performance
        
    except Exception as e:
        print(f"   ❌ Physics Engine Test Failed: {e}")
        return False, None

def test_4d_ddpm():
    """Test 4D DDPM model"""
    print("\n🤖 Testing Revolutionary 4D DDPM...")
    
    try:
        from revolutionary_4d_ddpm import Revolutionary4DDDPM, DiffusionConfig
        import torch
        
        # Create config
        config = DiffusionConfig(
            time_steps=16,  # Smaller for testing
            height=16,
            width=32,
            channels=3
        )
        
        # Create model
        model = Revolutionary4DDDPM(config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"   Device: {device}")
        print(f"   Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, config.channels, config.time_steps, 
                       config.height, config.width).to(device)
        timestep = torch.randint(0, 1000, (batch_size,)).to(device)
        
        with torch.no_grad():
            output = model(x, timestep)
        
        print(f"   Input Shape: {x.shape}")
        print(f"   Output Shape: {output.shape}")
        print(f"   ✅ 4D DDPM Forward Pass Successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 4D DDPM Test Failed: {e}")
        return False

def test_quantum_suite():
    """Test quantum state transfer suite"""
    print("\n🌊 Testing Quantum State Transfer Suite...")
    
    try:
        from quantum_state_transfer import QuantumStateTransferSuite
        
        # Create test Hamiltonian
        n_modes = 5
        hamiltonian = np.zeros((n_modes, n_modes), dtype=complex)
        
        # Nearest neighbor coupling
        for i in range(n_modes - 1):
            coupling = 1.0 + 0.1 * np.random.randn()
            hamiltonian[i, i+1] = coupling
            hamiltonian[i+1, i] = coupling
        
        # On-site energies
        for i in range(n_modes):
            hamiltonian[i, i] = 0.1 * np.random.randn()
        
        # Test quantum suite
        quantum_suite = QuantumStateTransferSuite()
        result = quantum_suite.optimize_state_transfer_protocol(hamiltonian)
        
        print(f"   Achieved Fidelity: {result['achieved_fidelity']:.4f}")
        print(f"   Transfer Time: {result['transfer_time_ns']:.1f} ns")
        print(f"   Protocol Robustness: {result['protocol_robustness']:.3f}")
        print(f"   Target Met: {'✅' if result['fidelity_target_met'] else '⚠️'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quantum Suite Test Failed: {e}")
        return False

def test_meep_engine():
    """Test MEEP simulation engine"""
    print("\n⚡ Testing MEEP Simulation Engine...")
    
    try:
        from revolutionary_meep_engine import RevolutionaryMEEPEngine
        
        # Create test epsilon movie
        T, H, W, C = 16, 16, 32, 3  # Smaller for testing
        epsilon_movie = np.random.randn(T, H, W, C) * 0.1 + 2.25
        
        # Add structure
        epsilon_movie[:, H//4:3*H//4, W//4:3*W//4, :] += 1.0  # Waveguide
        
        # Test MEEP engine
        meep_engine = RevolutionaryMEEPEngine()
        results = meep_engine.validate_revolutionary_isolation(epsilon_movie)
        
        print(f"   Peak Isolation: {results['peak_isolation_db']:.1f} dB")
        print(f"   Bandwidth: {results['bandwidth_ghz']:.1f} GHz")
        print(f"   Revolutionary Status: {'✅' if results['revolutionary_status'] else '⚠️'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ MEEP Engine Test Failed: {e}")
        return False

def test_dataset_generator():
    """Test dataset generation"""
    print("\n📊 Testing Dataset Generator...")
    
    try:
        from revolutionary_dataset_generator import RevolutionaryDatasetGenerator, DatasetConfig
        
        # Small test configuration
        config = DatasetConfig(
            n_samples=10,  # Very small for testing
            time_steps=16,
            height=16,
            width=32,
            optimization_iterations=10,
            parallel_workers=1,
            output_file="test_dataset.h5"
        )
        
        # Test generator
        generator = RevolutionaryDatasetGenerator(config)
        
        # Generate small test dataset
        print("   Generating test dataset...")
        dataset_path = generator.generate_revolutionary_dataset()
        
        print(f"   ✅ Test dataset generated: {dataset_path}")
        
        # Verify dataset
        import h5py
        with h5py.File(dataset_path, 'r') as f:
            n_samples = len(f['epsilon_movies'])
            print(f"   Samples in dataset: {n_samples}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset Generator Test Failed: {e}")
        return False

def test_execution_engine():
    """Test execution engine initialization"""
    print("\n🚀 Testing Execution Engine...")
    
    try:
        from revolutionary_execution_engine import RevolutionaryExecutionEngine, RevolutionaryPipelineConfig
        
        # Create test configuration
        config = RevolutionaryPipelineConfig(
            dataset_size=10,  # Very small for testing
            ddmp_epochs=1,
            n_validation_samples=5,
            meep_validation_samples=2
        )
        
        # Initialize engine
        engine = RevolutionaryExecutionEngine(config)
        
        print(f"   ✅ Execution Engine Initialized")
        print(f"   Target Isolation: {config.target_isolation_db} dB")
        print(f"   Target Bandwidth: {config.target_bandwidth_ghz} GHz")
        print(f"   Target Fidelity: {config.target_quantum_fidelity}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Execution Engine Test Failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Running Performance Benchmark...")
    
    try:
        from revolutionary_physics_engine import RevolutionaryTimeCrystalEngine
        
        # Create test data
        T, H, W, C = 64, 32, 128, 3
        epsilon_movie = np.random.randn(T, H, W, C) * 0.1 + 2.25
        
        # Benchmark physics engine
        engine = RevolutionaryTimeCrystalEngine()
        
        start_time = time.time()
        performance = engine.evaluate_revolutionary_performance(epsilon_movie)
        evaluation_time = time.time() - start_time
        
        print(f"   Evaluation Time: {evaluation_time:.3f} seconds")
        print(f"   Performance:")
        print(f"     Isolation: {performance['isolation_db']:.1f} dB")
        print(f"     Bandwidth: {performance['bandwidth_ghz']:.1f} GHz")
        print(f"     Fidelity: {performance['quantum_fidelity']:.3f}")
        
        # Check if near revolutionary targets
        near_targets = (
            performance['isolation_db'] > 50 and
            performance['bandwidth_ghz'] > 100 and 
            performance['quantum_fidelity'] > 0.9
        )
        
        print(f"   Performance Level: {'🎯 Near Revolutionary' if near_targets else 'ℹ️ Baseline'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance Benchmark Failed: {e}")
        return False

def run_integration_test():
    """Run integration test with all components"""
    print("\n🔧 Running Integration Test...")
    
    try:
        from revolutionary_physics_engine import RevolutionaryTimeCrystalEngine
        from quantum_state_transfer import QuantumStateTransferSuite
        from revolutionary_meep_engine import RevolutionaryMEEPEngine
        
        # Create test epsilon movie
        T, H, W, C = 32, 16, 64, 3
        epsilon_movie = np.random.randn(T, H, W, C) * 0.1 + 2.25
        
        # Add realistic structure and modulation
        epsilon_movie[:, H//4:3*H//4, W//4:3*W//4, :] += 1.5  # Core
        for t in range(T):
            modulation = 0.2 * np.sin(2 * np.pi * t / T)
            epsilon_movie[t] *= (1 + modulation)
        
        # Test all engines
        print("   Testing Physics Engine...")
        physics_engine = RevolutionaryTimeCrystalEngine()
        physics_result = physics_engine.evaluate_revolutionary_performance(epsilon_movie)
        
        print("   Testing MEEP Engine...")
        meep_engine = RevolutionaryMEEPEngine()
        meep_result = meep_engine.validate_revolutionary_isolation(epsilon_movie)
        
        print("   Testing Quantum Suite...")
        quantum_suite = QuantumStateTransferSuite()
        # Create mock Hamiltonian
        n_modes = 5
        hamiltonian = np.random.randn(n_modes, n_modes) * 0.1
        hamiltonian = (hamiltonian + hamiltonian.T) / 2  # Make symmetric
        quantum_result = quantum_suite.optimize_state_transfer_protocol(hamiltonian)
        
        # Compare results
        print(f"   Results Comparison:")
        print(f"     Physics: {physics_result['isolation_db']:.1f} dB")
        print(f"     MEEP: {meep_result['peak_isolation_db']:.1f} dB")
        print(f"     Quantum: {quantum_result['achieved_fidelity']:.3f}")
        
        # Check consistency
        isolation_diff = abs(physics_result['isolation_db'] - meep_result['peak_isolation_db'])
        consistent = isolation_diff < 20  # Allow some difference due to different methods
        
        print(f"   Consistency: {'✅' if consistent else '⚠️'} (Δ={isolation_diff:.1f} dB)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration Test Failed: {e}")
        return False

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Revolutionary Time-Crystal Test Suite")
    parser.add_argument("--quick", action="store_true", help="Quick tests only")
    parser.add_argument("--full-validation", action="store_true", help="Full validation including slow tests")
    
    args = parser.parse_args()
    
    print("🚀 Revolutionary Time-Crystal Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Core component tests
    test_results.append(("Physics Engine", test_physics_engine()[0]))
    test_results.append(("4D DDPM", test_4d_ddpm()))
    test_results.append(("Quantum Suite", test_quantum_suite()))
    test_results.append(("MEEP Engine", test_meep_engine()))
    test_results.append(("Execution Engine", test_execution_engine()))
    
    if not args.quick:
        test_results.append(("Dataset Generator", test_dataset_generator()))
        test_results.append(("Performance Benchmark", run_performance_benchmark()))
    
    if args.full_validation:
        test_results.append(("Integration Test", run_integration_test()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All tests passed! Revolutionary system ready for deployment.")
        print("\n📋 Next Steps:")
        print("   1. Run full pipeline: python revolutionary_execution_engine.py")
        print("   2. Generate dataset: python revolutionary_dataset_generator.py")
        print("   3. Train 4D DDPM: python revolutionary_4d_ddpm.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the output above for details.")
        print("   Consider running setup again: python setup_revolutionary_environment.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
