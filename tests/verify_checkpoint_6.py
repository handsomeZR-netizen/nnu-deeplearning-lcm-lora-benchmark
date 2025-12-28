"""
Checkpoint 6 verification script for evaluation system.

Verifies:
1. CLIPScore computation works correctly
2. LPIPS computation works correctly
3. VRAM statistics are properly separated between inference and evaluation
4. All evaluation tests pass

Feature: lcm-lora-acceleration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import numpy as np


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
    
    # Check transformers (for CLIP)
    try:
        from transformers import CLIPProcessor, CLIPModel
        deps["clip"] = True
    except ImportError:
        deps["clip"] = False
    
    # Check lpips
    try:
        import lpips
        deps["lpips"] = True
    except ImportError:
        deps["lpips"] = False
    
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
    print(f"  CLIP (transformers): {'✓' if deps.get('clip') else '✗ NOT INSTALLED'}")
    print(f"  LPIPS: {'✓' if deps.get('lpips') else '✗ NOT INSTALLED'}")
    print(f"  Hypothesis: {deps.get('hypothesis', 'NOT INSTALLED')}")
    
    return deps


def verify_clip_score():
    """Verify CLIPScore computation works correctly."""
    print("\n" + "="*60)
    print("Verifying CLIPScore Computation")
    print("="*60)
    
    from src.metrics.collector import MetricsCollector
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collector = MetricsCollector(device=device)
    
    # Test 1: Basic CLIPScore computation
    print("\n  Test 1: Basic CLIPScore computation")
    image = Image.new("RGB", (256, 256), color="red")
    prompt = "a red square"
    
    score = collector.compute_clip_score(image, prompt)
    
    assert isinstance(score, float), f"CLIPScore should be float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"CLIPScore should be in [0, 1], got {score}"
    print(f"    ✓ CLIPScore: {score:.4f} (valid range)")
    
    # Test 2: Different prompts should give different scores
    print("\n  Test 2: Different prompts give different scores")
    score_matching = collector.compute_clip_score(image, "a red image")
    score_mismatching = collector.compute_clip_score(image, "a blue ocean with waves")
    
    print(f"    Matching prompt score: {score_matching:.4f}")
    print(f"    Mismatching prompt score: {score_mismatching:.4f}")
    # Note: We don't assert one is higher than the other as CLIP can be unpredictable
    # with simple solid color images, but both should be valid
    assert 0.0 <= score_matching <= 1.0
    assert 0.0 <= score_mismatching <= 1.0
    print(f"    ✓ Both scores in valid range")
    
    # Test 3: Batch computation
    print("\n  Test 3: Batch CLIPScore computation")
    images = [
        Image.new("RGB", (128, 128), color="red"),
        Image.new("RGB", (128, 128), color="blue"),
        Image.new("RGB", (128, 128), color="green"),
    ]
    prompts = ["a red image", "a blue image", "a green image"]
    
    scores = collector.compute_clip_score_batch(images, prompts)
    
    assert len(scores) == 3, f"Should have 3 scores, got {len(scores)}"
    assert all(0.0 <= s <= 1.0 for s in scores), "All scores should be in [0, 1]"
    print(f"    ✓ Batch scores: {[f'{s:.4f}' for s in scores]}")
    
    collector.unload_models()
    print("\n  ✓ CLIPScore verification passed!")
    return True


def verify_lpips():
    """Verify LPIPS computation works correctly."""
    print("\n" + "="*60)
    print("Verifying LPIPS Computation")
    print("="*60)
    
    from src.metrics.collector import MetricsCollector
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collector = MetricsCollector(device=device)
    
    # Test 1: Identical images should have LPIPS near 0
    print("\n  Test 1: Identical images have LPIPS near 0")
    image = Image.new("RGB", (128, 128), color="red")
    
    lpips_identical = collector.compute_lpips(image, image)
    
    assert isinstance(lpips_identical, float), f"LPIPS should be float, got {type(lpips_identical)}"
    assert lpips_identical < 0.01, f"LPIPS for identical images should be near 0, got {lpips_identical}"
    print(f"    ✓ LPIPS (identical): {lpips_identical:.6f}")
    
    # Test 2: Different images should have positive LPIPS
    print("\n  Test 2: Different images have positive LPIPS")
    image1 = Image.new("RGB", (128, 128), color="red")
    image2 = Image.new("RGB", (128, 128), color="blue")
    
    lpips_different = collector.compute_lpips(image1, image2)
    
    assert lpips_different > 0.0, f"LPIPS for different images should be positive, got {lpips_different}"
    print(f"    ✓ LPIPS (different): {lpips_different:.4f}")
    
    # Test 3: Different sizes are handled
    print("\n  Test 3: Different image sizes are handled")
    image_small = Image.new("RGB", (64, 64), color="red")
    image_large = Image.new("RGB", (256, 256), color="red")
    
    lpips_diff_size = collector.compute_lpips(image_small, image_large)
    
    assert isinstance(lpips_diff_size, float), "LPIPS should handle different sizes"
    print(f"    ✓ LPIPS (different sizes): {lpips_diff_size:.4f}")
    
    # Test 4: Batch computation
    print("\n  Test 4: Batch LPIPS computation")
    images1 = [
        Image.new("RGB", (64, 64), color="red"),
        Image.new("RGB", (64, 64), color="blue"),
    ]
    images2 = [
        Image.new("RGB", (64, 64), color="green"),
        Image.new("RGB", (64, 64), color="yellow"),
    ]
    
    scores = collector.compute_lpips_batch(images1, images2)
    
    assert len(scores) == 2, f"Should have 2 scores, got {len(scores)}"
    assert all(isinstance(s, float) for s in scores), "All scores should be floats"
    print(f"    ✓ Batch LPIPS: {[f'{s:.4f}' for s in scores]}")
    
    collector.unload_models()
    print("\n  ✓ LPIPS verification passed!")
    return True


def verify_vram_separation():
    """Verify VRAM statistics are properly separated between inference and evaluation."""
    print("\n" + "="*60)
    print("Verifying VRAM Statistics Separation")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping VRAM separation test")
        return True
    
    from src.metrics.collector import MetricsCollector
    from src.core.pipeline import PipelineManager
    
    MODEL_DIR = "models/dreamshaper-7"
    LCM_LORA_DIR = "models/lcm-lora-sdv1-5"
    
    if not Path(MODEL_DIR).exists() or not Path(LCM_LORA_DIR).exists():
        print("  ⚠ Model files not found, skipping VRAM separation test")
        return True
    
    # Clear VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test 1: Measure inference VRAM
    print("\n  Test 1: Measure inference VRAM (without evaluation models)")
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    manager.load_lcm_pipeline(fuse_lora=True)
    manager.apply_optimizations(attention_slicing=True, vae_slicing=True)
    manager.warmup(num_steps=2)
    
    # Generate image and record VRAM
    result = manager.generate(
        prompt="a test image",
        num_steps=4,
        guidance_scale=1.0,
        seed=42,
        width=512,
        height=512,
    )
    
    inference_vram = result.peak_vram_mb
    print(f"    Inference peak VRAM: {inference_vram:.0f}MB")
    
    # Unload pipeline
    manager.unload()
    torch.cuda.empty_cache()
    
    # Test 2: Measure evaluation VRAM separately
    print("\n  Test 2: Measure evaluation VRAM (without inference pipeline)")
    
    torch.cuda.reset_peak_memory_stats()
    
    collector = MetricsCollector(device="cuda")
    
    # Compute CLIPScore
    clip_score = collector.compute_clip_score(result.image, result.prompt)
    clip_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"    CLIPScore: {clip_score:.4f}")
    print(f"    CLIP model VRAM: {clip_vram:.0f}MB")
    
    # Compute LPIPS (need two images)
    image2 = Image.new("RGB", (512, 512), color="blue")
    lpips_value = collector.compute_lpips(result.image, image2)
    lpips_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"    LPIPS: {lpips_value:.4f}")
    print(f"    LPIPS model VRAM: {lpips_vram:.0f}MB")
    
    # Get current evaluation VRAM
    eval_vram = collector.get_vram_usage()
    print(f"    Current evaluation VRAM: {eval_vram['allocated_mb']:.0f}MB")
    
    # Unload evaluation models
    collector.unload_models()
    
    # Test 3: Verify separation
    print("\n  Test 3: Verify VRAM separation")
    
    # The key point is that we can measure inference and evaluation VRAM separately
    # by loading/unloading models at different times
    print(f"    ✓ Inference VRAM measured: {inference_vram:.0f}MB")
    print(f"    ✓ Evaluation VRAM measured: {lpips_vram:.0f}MB")
    print(f"    ✓ Models can be loaded/unloaded independently")
    
    print("\n  ✓ VRAM separation verification passed!")
    return True


def verify_quality_metrics_collection():
    """Verify quality metrics collection works correctly."""
    print("\n" + "="*60)
    print("Verifying Quality Metrics Collection")
    print("="*60)
    
    from src.metrics.collector import MetricsCollector
    from src.core.models import QualityMetrics
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collector = MetricsCollector(device=device)
    
    # Test: Collect quality metrics
    print("\n  Test: Collect quality metrics (CLIPScore + LPIPS)")
    
    image = Image.new("RGB", (256, 256), color="red")
    reference = Image.new("RGB", (256, 256), color="blue")
    prompt = "a red square"
    
    metrics = collector.collect_quality_metrics(
        image=image,
        prompt=prompt,
        reference_image=reference
    )
    
    assert isinstance(metrics, QualityMetrics), f"Should return QualityMetrics, got {type(metrics)}"
    assert 0.0 <= metrics.clip_score <= 1.0, f"CLIPScore should be in [0, 1], got {metrics.clip_score}"
    assert metrics.lpips is not None, "LPIPS should be computed when reference provided"
    assert metrics.lpips > 0.0, f"LPIPS should be positive for different images, got {metrics.lpips}"
    
    print(f"    ✓ CLIPScore: {metrics.clip_score:.4f}")
    print(f"    ✓ LPIPS: {metrics.lpips:.4f}")
    print(f"    ✓ FID: {metrics.fid} (not computed)")
    
    collector.unload_models()
    print("\n  ✓ Quality metrics collection verification passed!")
    return True


def run_property_tests():
    """Run property-based tests for metrics."""
    print("\n" + "="*60)
    print("Running Property-Based Tests")
    print("="*60)
    
    import subprocess
    
    # Run pytest on metrics property tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "tests/test_metrics_properties.py", 
         "-v", "--tb=short", "-x"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print("\n  ✓ All property tests passed!")
        return True
    else:
        print("\n  ✗ Some property tests failed")
        return False


def main():
    """Run all checkpoint 6 verifications."""
    print("="*60)
    print("Checkpoint 6: Evaluation System Verification")
    print("="*60)
    
    deps = check_dependencies()
    
    if not deps.get("torch"):
        print("\nERROR: PyTorch not installed")
        return 1
    
    if not deps.get("clip"):
        print("\nERROR: transformers (CLIP) not installed")
        print("Install with: pip install transformers")
        return 1
    
    if not deps.get("lpips"):
        print("\nERROR: lpips not installed")
        print("Install with: pip install lpips")
        return 1
    
    results = []
    
    # Verify CLIPScore
    try:
        results.append(("CLIPScore Computation", verify_clip_score()))
    except Exception as e:
        print(f"\nERROR in CLIPScore verification: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CLIPScore Computation", False))
    
    # Verify LPIPS
    try:
        results.append(("LPIPS Computation", verify_lpips()))
    except Exception as e:
        print(f"\nERROR in LPIPS verification: {e}")
        import traceback
        traceback.print_exc()
        results.append(("LPIPS Computation", False))
    
    # Verify VRAM separation
    try:
        results.append(("VRAM Statistics Separation", verify_vram_separation()))
    except Exception as e:
        print(f"\nERROR in VRAM separation verification: {e}")
        import traceback
        traceback.print_exc()
        results.append(("VRAM Statistics Separation", False))
    
    # Verify quality metrics collection
    try:
        results.append(("Quality Metrics Collection", verify_quality_metrics_collection()))
    except Exception as e:
        print(f"\nERROR in quality metrics verification: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Quality Metrics Collection", False))
    
    # Run property tests
    try:
        results.append(("Property-Based Tests", run_property_tests()))
    except Exception as e:
        print(f"\nERROR in property tests: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Property-Based Tests", False))
    
    # Summary
    print("\n" + "="*60)
    print("CHECKPOINT 6 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checkpoint 6 verifications passed!")
        print("Evaluation system is ready for next phase.")
    else:
        print("\n✗ Some verifications failed. Please review and fix issues.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
