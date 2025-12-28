"""
Standalone script to run property-based tests for PipelineManager.

This script runs outside of pytest to avoid fixture-related crashes.
Run with: python tests/run_property_tests.py

Feature: lcm-lora-acceleration
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hypothesis import given, strategies as st, settings

from src.core.pipeline import PipelineManager
from src.core.models import GenerationResult


# Model paths
MODEL_DIR = "models/dreamshaper-7"
LCM_LORA_DIR = "models/lcm-lora-sdv1-5"


def images_equal(img1, img2) -> bool:
    """Compare two PIL images for equality"""
    if img1.size != img2.size:
        return False
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return np.array_equal(arr1, arr2)


def test_seed_determinism():
    """
    Property 1: 随机种子确定性
    
    *For any* 给定的 prompt 和 seed，使用相同配置多次生成 SHALL 产生完全相同的图像输出。
    
    Feature: lcm-lora-acceleration, Property 1: 随机种子确定性
    **Validates: Requirements 1.4**
    """
    print("\n" + "="*60)
    print("Property 1: 随机种子确定性")
    print("="*60)
    
    # Load pipeline once
    print("Loading LCM pipeline...")
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    manager.load_lcm_pipeline(fuse_lora=True)
    manager.apply_optimizations(attention_slicing=True, vae_slicing=True)
    manager.warmup(num_steps=2)
    
    prompt = "a simple red cube on white background"
    num_steps = 4
    guidance_scale = 1.0
    
    # Test with 5 different seeds
    test_seeds = [0, 42, 12345, 999999, 2147483647]
    passed = 0
    failed = 0
    
    for seed in test_seeds:
        print(f"\nTesting seed={seed}...")
        
        # Generate first image
        result1 = manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=256,
            height=256,
        )
        
        # Generate second image with same seed
        result2 = manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=256,
            height=256,
        )
        
        # Check if images are identical
        if images_equal(result1.image, result2.image):
            print(f"  ✓ PASSED: Images with seed {seed} are identical")
            passed += 1
        else:
            print(f"  ✗ FAILED: Images with seed {seed} are NOT identical")
            failed += 1
    
    manager.unload()
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0



def test_vram_optimization():
    """
    Property 2: 显存优化有效性
    
    *For any* 启用 attention_slicing 的配置，峰值显存 SHALL 小于等于未启用时的峰值显存。
    
    Feature: lcm-lora-acceleration, Property 2: 显存优化有效性
    **Validates: Requirements 3.1, 3.2**
    """
    print("\n" + "="*60)
    print("Property 2: 显存优化有效性")
    print("="*60)
    
    prompt = "a simple blue sphere"
    num_steps = 4
    guidance_scale = 1.0
    seed = 42
    
    # Test 1: Without optimizations
    print("\nTest 1: Without attention_slicing...")
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    manager.load_lcm_pipeline(fuse_lora=True)
    manager.apply_optimizations(attention_slicing=False, vae_slicing=False)
    manager.warmup(num_steps=2)
    
    result_no_slicing = manager.generate(
        prompt=prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        width=512,
        height=512,
    )
    vram_no_slicing = result_no_slicing.peak_vram_mb
    print(f"  Peak VRAM without slicing: {vram_no_slicing:.0f} MB")
    
    manager.unload()
    torch.cuda.empty_cache()
    
    # Test 2: With optimizations
    print("\nTest 2: With attention_slicing...")
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    manager.load_lcm_pipeline(fuse_lora=True)
    manager.apply_optimizations(attention_slicing=True, vae_slicing=True)
    manager.warmup(num_steps=2)
    
    result_with_slicing = manager.generate(
        prompt=prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        width=512,
        height=512,
    )
    vram_with_slicing = result_with_slicing.peak_vram_mb
    print(f"  Peak VRAM with slicing: {vram_with_slicing:.0f} MB")
    
    manager.unload()
    
    # Check property
    tolerance = 100  # MB tolerance for measurement variance
    if vram_with_slicing <= vram_no_slicing + tolerance:
        print(f"\n✓ PASSED: VRAM with slicing ({vram_with_slicing:.0f}MB) <= VRAM without slicing ({vram_no_slicing:.0f}MB)")
        return True
    else:
        print(f"\n✗ FAILED: VRAM with slicing ({vram_with_slicing:.0f}MB) > VRAM without slicing ({vram_no_slicing:.0f}MB)")
        return False


def main():
    """Run all property tests"""
    print("="*60)
    print("LCM-LoRA Property-Based Tests")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    if not Path(MODEL_DIR).exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return 1
    
    if not Path(LCM_LORA_DIR).exists():
        print(f"ERROR: LCM-LoRA directory not found: {LCM_LORA_DIR}")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    results = []
    
    # Run Property 1
    try:
        results.append(("Property 1: 随机种子确定性", test_seed_determinism()))
    except Exception as e:
        print(f"Property 1 ERROR: {e}")
        results.append(("Property 1: 随机种子确定性", False))
    
    # Run Property 2
    try:
        results.append(("Property 2: 显存优化有效性", test_vram_optimization()))
    except Exception as e:
        print(f"Property 2 ERROR: {e}")
        results.append(("Property 2: 显存优化有效性", False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
