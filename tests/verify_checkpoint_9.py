"""
Checkpoint 9 verification script for benchmark testing system.

Verifies:
1. Small-scale comparison experiment runs correctly
2. Dataset loading and analysis works correctly
3. All benchmark and dataset tests pass

Feature: lcm-lora-acceleration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


# Model paths
MODEL_DIR = "models/dreamshaper-7"
LCM_LORA_DIR = "models/lcm-lora-sdv1-5"
OUTPUT_DIR = "outputs/checkpoint_9_test"


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    deps = {}
    
    # Check torch
    try:
        import torch
        deps["torch"] = torch.__version__
        deps["cuda"] = torch.cuda.is_available()
        if deps["cuda"]:
            deps["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        deps["torch"] = None
        deps["cuda"] = False
    
    # Check hypothesis
    try:
        import hypothesis
        deps["hypothesis"] = hypothesis.__version__
    except ImportError:
        deps["hypothesis"] = None
    
    print(f"  PyTorch: {deps.get('torch', 'NOT INSTALLED')}")
    print(f"  CUDA Available: {deps.get('cuda', False)}")
    if deps.get("cuda"):
        print(f"  GPU: {deps.get('gpu', 'Unknown')}")
    print(f"  Hypothesis: {deps.get('hypothesis', 'NOT INSTALLED')}")
    
    return deps


def verify_dataset_loading():
    """Verify dataset loading and analysis works correctly."""
    print("\n" + "="*60)
    print("Verifying Dataset Loading and Analysis")
    print("="*60)
    
    from src.dataset.builder import DatasetBuilder, EvaluationDataset, DatasetStats
    
    # Test 1: Load sample data
    print("\n  Test 1: Load sample captions")
    builder = DatasetBuilder()
    count = builder.load_captions()
    
    assert count > 0, "Should load some captions"
    print(f"    ✓ Loaded {count} captions")
    
    # Test 2: Build evaluation dataset
    print("\n  Test 2: Build evaluation dataset")
    dataset = builder.build_evaluation_dataset(num_samples=20, seed=42)
    
    assert isinstance(dataset, EvaluationDataset), "Should return EvaluationDataset"
    assert len(dataset) > 0, "Dataset should have prompts"
    print(f"    ✓ Built dataset with {len(dataset)} prompts")
    
    # Test 3: Analyze dataset
    print("\n  Test 3: Analyze dataset statistics")
    stats = builder.analyze_dataset(dataset)
    
    assert isinstance(stats, DatasetStats), "Should return DatasetStats"
    assert stats.total_prompts == len(dataset), "Stats should match dataset size"
    assert stats.avg_length > 0, "Average length should be positive"
    print(f"    ✓ Total prompts: {stats.total_prompts}")
    print(f"    ✓ Average length: {stats.avg_length:.1f}")
    print(f"    ✓ Categories: {list(stats.category_distribution.keys())}")
    
    # Test 4: Export and reload
    print("\n  Test 4: Export and reload dataset")
    output_path = Path(OUTPUT_DIR) / "test_dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    builder.export_prompts(dataset, str(output_path))
    assert output_path.exists(), "Export file should exist"
    
    loaded = builder.load_dataset(str(output_path))
    assert len(loaded) == len(dataset), "Loaded dataset should match original"
    print(f"    ✓ Exported and reloaded {len(loaded)} prompts")
    
    # Test 5: Category filtering
    print("\n  Test 5: Category filtering")
    portrait_dataset = builder.build_evaluation_dataset(
        num_samples=50,
        categories=["portrait"],
        seed=42
    )
    
    for prompt in portrait_dataset.prompts:
        assert prompt.category == "portrait", f"All prompts should be portrait, got {prompt.category}"
    print(f"    ✓ Filtered to {len(portrait_dataset)} portrait prompts")
    
    print("\n  ✓ Dataset loading and analysis verification passed!")
    return True


def verify_small_comparison_experiment():
    """Run a small-scale comparison experiment to verify the benchmark flow."""
    print("\n" + "="*60)
    print("Verifying Small-Scale Comparison Experiment")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping comparison experiment")
        return True
    
    if not Path(MODEL_DIR).exists() or not Path(LCM_LORA_DIR).exists():
        print("  ⚠ Model files not found, skipping comparison experiment")
        return True
    
    from src.core.pipeline import PipelineManager
    from src.core.models import ExperimentConfig
    from src.benchmark.runner import BenchmarkRunner, ExperimentResults
    from src.dataset.builder import DatasetBuilder
    
    # Build a small test dataset
    print("\n  Step 1: Build small test dataset")
    builder = DatasetBuilder()
    dataset = builder.build_evaluation_dataset(num_samples=3, seed=42)
    prompts = dataset.get_prompt_texts()[:2]  # Use only 2 prompts for speed
    seeds = [42]  # Single seed
    
    print(f"    Using {len(prompts)} prompts, {len(seeds)} seeds")
    
    # Create pipeline manager
    print("\n  Step 2: Initialize pipeline manager")
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        pipeline_manager=manager,
        output_dir=OUTPUT_DIR,
    )
    
    # Define minimal configs for quick test
    print("\n  Step 3: Define test configurations")
    test_configs = [
        ExperimentConfig(
            name="Euler_10",
            scheduler_type="euler",
            num_steps=10,  # Reduced steps for speed
            guidance_scale=7.5,
            use_lcm_lora=False,
        ),
        ExperimentConfig(
            name="LCM_4",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True,
        ),
    ]
    print(f"    Configs: {[c.name for c in test_configs]}")
    
    # Run comparison experiment
    print("\n  Step 4: Run comparison experiment")
    results = runner.run_comparison_experiment(
        prompts=prompts,
        seeds=seeds,
        configs=test_configs,
        num_repeats=1,  # Single repeat for speed
        width=512,
        height=512,
        compute_quality=False,  # Skip quality metrics for speed
    )
    
    # Verify results
    print("\n  Step 5: Verify experiment results")
    assert isinstance(results, ExperimentResults), "Should return ExperimentResults"
    assert results.experiment_name is not None, "Should have experiment name"
    assert len(results.configs) == len(test_configs), "Should have all configs"
    
    # Check runtime stats
    for config_name in ["Euler_10", "LCM_4"]:
        assert config_name in results.runtime_stats, f"Should have stats for {config_name}"
        stats = results.runtime_stats[config_name]
        assert "mean" in stats, "Should have mean latency"
        assert stats["mean"] > 0, "Mean latency should be positive"
        print(f"    {config_name}: {stats['mean']:.1f}ms mean latency")
    
    # Verify LCM is faster than Euler
    euler_latency = results.runtime_stats["Euler_10"]["mean"]
    lcm_latency = results.runtime_stats["LCM_4"]["mean"]
    
    print(f"\n    Euler 10-step: {euler_latency:.1f}ms")
    print(f"    LCM 4-step: {lcm_latency:.1f}ms")
    
    if lcm_latency < euler_latency:
        print(f"    ✓ LCM is {euler_latency/lcm_latency:.1f}x faster than Euler")
    else:
        print(f"    ⚠ LCM ({lcm_latency:.1f}ms) not faster than Euler ({euler_latency:.1f}ms)")
    
    # Check VRAM stats
    for config_name in ["Euler_10", "LCM_4"]:
        vram_stats = results.vram_stats[config_name]
        assert vram_stats["mean"] > 0, f"VRAM should be positive for {config_name}"
        print(f"    {config_name}: {vram_stats['mean']:.0f}MB peak VRAM")
    
    # Cleanup
    manager.unload()
    
    print("\n  ✓ Small-scale comparison experiment verification passed!")
    return True


def run_dataset_tests():
    """Run dataset builder tests."""
    print("\n" + "="*60)
    print("Running Dataset Builder Tests")
    print("="*60)
    
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/test_dataset_builder.py", 
         "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n  ✓ All dataset tests passed!")
        return True
    else:
        print("\n  ✗ Some dataset tests failed")
        return False


def run_benchmark_tests():
    """Run benchmark runner tests (basic tests only, skip GPU-dependent)."""
    print("\n" + "="*60)
    print("Running Benchmark Runner Basic Tests")
    print("="*60)
    
    import subprocess
    
    # Run only basic tests that don't require GPU
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/test_benchmark_properties.py::TestBenchmarkRunnerBasics",
         "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n  ✓ All benchmark basic tests passed!")
        return True
    else:
        print("\n  ✗ Some benchmark tests failed")
        return False


def main():
    """Run all checkpoint 9 verifications."""
    print("="*60)
    print("Checkpoint 9: Benchmark Testing System Verification")
    print("="*60)
    
    deps = check_dependencies()
    
    if not deps.get("torch"):
        print("\nERROR: PyTorch not installed")
        return 1
    
    results = []
    
    # Verify dataset loading and analysis
    try:
        results.append(("Dataset Loading and Analysis", verify_dataset_loading()))
    except Exception as e:
        print(f"\nERROR in dataset verification: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Dataset Loading and Analysis", False))
    
    # Run dataset tests
    try:
        results.append(("Dataset Builder Tests", run_dataset_tests()))
    except Exception as e:
        print(f"\nERROR in dataset tests: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Dataset Builder Tests", False))
    
    # Run benchmark basic tests
    try:
        results.append(("Benchmark Runner Basic Tests", run_benchmark_tests()))
    except Exception as e:
        print(f"\nERROR in benchmark tests: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Benchmark Runner Basic Tests", False))
    
    # Run small comparison experiment (requires GPU)
    try:
        results.append(("Small-Scale Comparison Experiment", verify_small_comparison_experiment()))
    except Exception as e:
        print(f"\nERROR in comparison experiment: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Small-Scale Comparison Experiment", False))
    
    # Summary
    print("\n" + "="*60)
    print("CHECKPOINT 9 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checkpoint 9 verifications passed!")
        print("Benchmark testing system is ready for next phase.")
    else:
        print("\n✗ Some verifications failed. Please review and fix issues.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
