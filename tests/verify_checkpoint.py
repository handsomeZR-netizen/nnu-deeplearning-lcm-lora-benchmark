"""
Checkpoint verification script for Task 3.

Verifies:
1. Baseline pipeline (Euler scheduler) can generate images
2. LCM-LoRA pipeline can generate images
3. Memory optimizations work correctly
4. All core functionality is operational

Feature: lcm-lora-acceleration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.core.pipeline import PipelineManager
from src.core.models import GenerationResult


# Model paths
MODEL_DIR = "models/dreamshaper-7"
LCM_LORA_DIR = "models/lcm-lora-sdv1-5"


def verify_baseline_pipeline():
    """Verify baseline pipeline with Euler scheduler can generate images."""
    print("\n" + "="*60)
    print("Verifying Baseline Pipeline (Euler Scheduler)")
    print("="*60)
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    
    # Load baseline with Euler scheduler
    manager.load_baseline_pipeline(scheduler_type="euler")
    manager.apply_optimizations(attention_slicing=True, vae_slicing=True)
    manager.warmup(num_steps=2)
    
    # Generate image
    result = manager.generate(
        prompt="a beautiful sunset over mountains",
        num_steps=20,
        guidance_scale=7.5,
        seed=42,
        width=512,
        height=512,
    )
    
    # Verify result
    assert result.image is not None, "Image should not be None"
    assert result.image.size == (512, 512), f"Image size should be (512, 512), got {result.image.size}"
    assert result.latency_ms > 0, "Latency should be positive"
    assert result.peak_vram_mb > 0, "VRAM should be positive"
    assert result.scheduler_type == "euler", f"Scheduler should be euler, got {result.scheduler_type}"
    
    print(f"  ✓ Image generated successfully")
    print(f"  ✓ Resolution: {result.image.size}")
    print(f"  ✓ Latency: {result.latency_ms:.1f}ms")
    print(f"  ✓ Peak VRAM: {result.peak_vram_mb:.0f}MB")
    print(f"  ✓ Scheduler: {result.scheduler_type}")
    
    manager.unload()
    return True


def verify_dpm_solver_pipeline():
    """Verify baseline pipeline with DPM-Solver scheduler can generate images."""
    print("\n" + "="*60)
    print("Verifying Baseline Pipeline (DPM-Solver Scheduler)")
    print("="*60)
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    
    # Load baseline with DPM-Solver scheduler
    manager.load_baseline_pipeline(scheduler_type="dpm_solver")
    manager.apply_optimizations(attention_slicing=True, vae_slicing=True)
    manager.warmup(num_steps=2)
    
    # Generate image
    result = manager.generate(
        prompt="a futuristic city at night",
        num_steps=20,
        guidance_scale=7.5,
        seed=42,
        width=512,
        height=512,
    )
    
    # Verify result
    assert result.image is not None, "Image should not be None"
    assert result.scheduler_type == "dpm_solver", f"Scheduler should be dpm_solver, got {result.scheduler_type}"
    
    print(f"  ✓ Image generated successfully")
    print(f"  ✓ Resolution: {result.image.size}")
    print(f"  ✓ Latency: {result.latency_ms:.1f}ms")
    print(f"  ✓ Peak VRAM: {result.peak_vram_mb:.0f}MB")
    print(f"  ✓ Scheduler: {result.scheduler_type}")
    
    manager.unload()
    return True


def verify_lcm_pipeline():
    """Verify LCM-LoRA pipeline can generate images with few steps."""
    print("\n" + "="*60)
    print("Verifying LCM-LoRA Pipeline")
    print("="*60)
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    
    # Load LCM pipeline
    manager.load_lcm_pipeline(fuse_lora=True)
    manager.apply_optimizations(attention_slicing=True, vae_slicing=True)
    manager.warmup(num_steps=2)
    
    # Test with different step counts (2, 4, 6, 8)
    step_counts = [2, 4, 6, 8]
    
    for steps in step_counts:
        result = manager.generate(
            prompt="a photo of an astronaut riding a horse",
            num_steps=steps,
            guidance_scale=1.0,
            seed=42,
            width=512,
            height=512,
        )
        
        assert result.image is not None, f"Image should not be None for {steps} steps"
        assert result.num_steps == steps, f"Steps should be {steps}, got {result.num_steps}"
        assert result.scheduler_type == "lcm", f"Scheduler should be lcm, got {result.scheduler_type}"
        
        print(f"  ✓ {steps} steps: {result.latency_ms:.1f}ms, {result.peak_vram_mb:.0f}MB")
    
    manager.unload()
    return True


def verify_memory_optimizations():
    """Verify memory optimizations are applied correctly."""
    print("\n" + "="*60)
    print("Verifying Memory Optimizations")
    print("="*60)
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    
    manager.load_lcm_pipeline(fuse_lora=True)
    
    # Apply all optimizations
    manager.apply_optimizations(
        attention_slicing=True,
        vae_slicing=True,
        vae_tiling=False,
        xformers=False,
        sdpa=True,
    )
    
    # Check optimization state
    opts = manager.optimizations
    assert opts["attention_slicing"] == True, "attention_slicing should be enabled"
    assert opts["vae_slicing"] == True, "vae_slicing should be enabled"
    
    print(f"  ✓ attention_slicing: {opts['attention_slicing']}")
    print(f"  ✓ vae_slicing: {opts['vae_slicing']}")
    print(f"  ✓ vae_tiling: {opts['vae_tiling']}")
    print(f"  ✓ sdpa: {opts['sdpa']}")
    
    # Generate to verify optimizations work
    result = manager.generate(
        prompt="test image",
        num_steps=4,
        guidance_scale=1.0,
        seed=42,
        width=512,
        height=512,
    )
    
    assert result.image is not None, "Image should be generated with optimizations"
    assert result.optimizations["attention_slicing"] == True, "Result should record optimizations"
    
    print(f"  ✓ Generation with optimizations successful")
    print(f"  ✓ Peak VRAM: {result.peak_vram_mb:.0f}MB")
    
    manager.unload()
    return True


def main():
    """Run all checkpoint verifications."""
    print("="*60)
    print("Checkpoint 3: Core Pipeline Verification")
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
    
    # Verify baseline pipeline (Euler)
    try:
        results.append(("Baseline Pipeline (Euler)", verify_baseline_pipeline()))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append(("Baseline Pipeline (Euler)", False))
    
    # Verify baseline pipeline (DPM-Solver)
    try:
        results.append(("Baseline Pipeline (DPM-Solver)", verify_dpm_solver_pipeline()))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append(("Baseline Pipeline (DPM-Solver)", False))
    
    # Verify LCM pipeline
    try:
        results.append(("LCM-LoRA Pipeline", verify_lcm_pipeline()))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append(("LCM-LoRA Pipeline", False))
    
    # Verify memory optimizations
    try:
        results.append(("Memory Optimizations", verify_memory_optimizations()))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append(("Memory Optimizations", False))
    
    # Summary
    print("\n" + "="*60)
    print("CHECKPOINT VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checkpoint verifications passed!")
        print("Core pipeline is ready for next phase.")
    else:
        print("\n✗ Some verifications failed. Please review and fix issues.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
