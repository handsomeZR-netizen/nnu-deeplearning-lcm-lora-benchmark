"""
Property-based tests for PipelineManager.

Tests correctness properties using hypothesis for property-based testing.
These tests require GPU and actual model files to run.

Feature: lcm-lora-acceleration

NOTE: Property tests (TestSeedDeterminism, TestVRAMOptimization) require:
- CUDA-capable GPU
- Model files in models/dreamshaper-7 and models/lcm-lora-sdv1-5
- Run with: pytest tests/test_pipeline_properties.py -v -k "property" --tb=short
"""

import pytest
import sys
from pathlib import Path
from typing import Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if hypothesis is available
try:
    from hypothesis import given, strategies as st, settings, assume, HealthCheck
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Check if torch and CUDA are available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    torch = None

from src.core.pipeline import PipelineManager, ModelLoadError, VRAMError
from src.core.models import GenerationResult


# Model paths - adjust these to your local setup
MODEL_DIR = "models/dreamshaper-7"
LCM_LORA_DIR = "models/lcm-lora-sdv1-5"


def images_equal(img1, img2) -> bool:
    """Compare two PIL images for equality"""
    if img1.size != img2.size:
        return False
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return np.array_equal(arr1, arr2)


def models_exist() -> bool:
    """Check if model files exist"""
    return Path(MODEL_DIR).exists() and Path(LCM_LORA_DIR).exists()


# Mark property tests to skip by default (require GPU + models)
requires_gpu_and_models = pytest.mark.skipif(
    not (models_exist() and CUDA_AVAILABLE and HYPOTHESIS_AVAILABLE),
    reason="Requires GPU, models, and hypothesis"
)


@pytest.fixture(scope="module")
def lcm_pipeline_manager():
    """
    Create a PipelineManager with LCM pipeline loaded.
    Module-scoped to avoid reloading for each test.
    """
    if not models_exist() or not CUDA_AVAILABLE:
        pytest.skip("Models or CUDA not available")
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    manager.load_lcm_pipeline(fuse_lora=True)
    manager.apply_optimizations(
        attention_slicing=True,
        vae_slicing=True,
    )
    # Warmup
    manager.warmup(num_steps=2)
    
    yield manager
    
    # Cleanup
    manager.unload()



@requires_gpu_and_models
class TestSeedDeterminism:
    """
    Property 1: 随机种子确定性
    
    *For any* 给定的 prompt 和 seed，使用相同配置多次生成 SHALL 产生完全相同的图像输出。
    
    **Validates: Requirements 1.4**
    """
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @settings(max_examples=5, deadline=None)
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_seed_determinism_property(self, lcm_pipeline_manager, seed):
        """
        Feature: lcm-lora-acceleration, Property 1: 随机种子确定性
        
        For any seed, generating twice with the same seed should produce identical images.
        **Validates: Requirements 1.4**
        """
        prompt = "a simple red cube on white background"
        num_steps = 4
        guidance_scale = 1.0
        
        # Generate first image
        result1 = lcm_pipeline_manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=256,  # Small size for faster testing
            height=256,
        )
        
        # Generate second image with same seed
        result2 = lcm_pipeline_manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=256,
            height=256,
        )
        
        # Images should be identical
        assert images_equal(result1.image, result2.image), \
            f"Images with seed {seed} are not identical"



@pytest.fixture(scope="function")
def fresh_pipeline_manager():
    """
    Create a fresh PipelineManager for each test.
    Used for tests that need to compare different optimization settings.
    """
    if not models_exist() or not CUDA_AVAILABLE:
        pytest.skip("Models or CUDA not available")
    
    manager = PipelineManager(
        model_dir=MODEL_DIR,
        lcm_lora_dir=LCM_LORA_DIR,
        device="cuda"
    )
    
    yield manager
    
    # Cleanup
    manager.unload()


@requires_gpu_and_models
class TestVRAMOptimization:
    """
    Property 2: 显存优化有效性
    
    *For any* 启用 attention_slicing 的配置，峰值显存 SHALL 小于等于未启用时的峰值显存。
    
    **Validates: Requirements 3.1, 3.2**
    """
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @settings(
        max_examples=3, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        enable_slicing=st.booleans(),
    )
    def test_attention_slicing_vram_property(self, fresh_pipeline_manager, seed, enable_slicing):
        """
        Feature: lcm-lora-acceleration, Property 2: 显存优化有效性
        
        For any configuration, enabling attention_slicing should not increase peak VRAM usage.
        **Validates: Requirements 3.1, 3.2**
        """
        prompt = "a simple blue sphere"
        num_steps = 4
        guidance_scale = 1.0
        
        # Load LCM pipeline
        fresh_pipeline_manager.load_lcm_pipeline(fuse_lora=True)
        
        # First, generate WITHOUT attention slicing to get baseline VRAM
        fresh_pipeline_manager.apply_optimizations(
            attention_slicing=False,
            vae_slicing=False,
        )
        fresh_pipeline_manager.warmup(num_steps=2)
        
        result_no_slicing = fresh_pipeline_manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=512,
            height=512,
        )
        vram_no_slicing = result_no_slicing.peak_vram_mb
        
        # Clear cache and reload
        fresh_pipeline_manager.unload()
        fresh_pipeline_manager.load_lcm_pipeline(fuse_lora=True)
        
        # Now generate WITH attention slicing
        fresh_pipeline_manager.apply_optimizations(
            attention_slicing=True,
            vae_slicing=True,
        )
        fresh_pipeline_manager.warmup(num_steps=2)
        
        result_with_slicing = fresh_pipeline_manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=512,
            height=512,
        )
        vram_with_slicing = result_with_slicing.peak_vram_mb
        
        # VRAM with slicing should be <= VRAM without slicing
        # Allow small tolerance for measurement variance
        tolerance = 50  # MB
        assert vram_with_slicing <= vram_no_slicing + tolerance, \
            f"VRAM with slicing ({vram_with_slicing:.0f}MB) > VRAM without slicing ({vram_no_slicing:.0f}MB)"


# Additional unit tests for basic functionality
class TestPipelineManagerBasics:
    """Basic unit tests for PipelineManager - no GPU required"""
    
    def test_model_load_error_invalid_path(self):
        """Test that invalid model path raises ModelLoadError"""
        with pytest.raises(ModelLoadError):
            PipelineManager(
                model_dir="/nonexistent/path",
                lcm_lora_dir="/nonexistent/lora",
            )
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_lcm_lora_load_error_invalid_path(self):
        """Test that invalid LCM-LoRA path raises ModelLoadError"""
        with pytest.raises(ModelLoadError):
            PipelineManager(
                model_dir=MODEL_DIR,
                lcm_lora_dir="/nonexistent/path",
            )
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_generate_without_loading_raises_error(self):
        """Test that generating without loading pipeline raises error"""
        manager = PipelineManager(
            model_dir=MODEL_DIR,
            lcm_lora_dir=LCM_LORA_DIR,
        )
        
        with pytest.raises(RuntimeError):
            manager.generate(
                prompt="test",
                num_steps=4,
                guidance_scale=1.0,
                seed=42,
            )
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_baseline_pipeline_rejects_lcm_scheduler(self):
        """Test that baseline pipeline rejects LCM scheduler"""
        manager = PipelineManager(
            model_dir=MODEL_DIR,
            lcm_lora_dir=LCM_LORA_DIR,
        )
        
        with pytest.raises(ValueError):
            manager.load_baseline_pipeline(scheduler_type="lcm")
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_apply_optimizations_without_loading_raises_error(self):
        """Test that applying optimizations without loading pipeline raises error"""
        manager = PipelineManager(
            model_dir=MODEL_DIR,
            lcm_lora_dir=LCM_LORA_DIR,
        )
        
        with pytest.raises(RuntimeError):
            manager.apply_optimizations()
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_warmup_without_loading_raises_error(self):
        """Test that warmup without loading pipeline raises error"""
        manager = PipelineManager(
            model_dir=MODEL_DIR,
            lcm_lora_dir=LCM_LORA_DIR,
        )
        
        with pytest.raises(RuntimeError):
            manager.warmup()
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_initial_state(self):
        """Test initial state of PipelineManager"""
        manager = PipelineManager(
            model_dir=MODEL_DIR,
            lcm_lora_dir=LCM_LORA_DIR,
        )
        
        assert manager.is_loaded is False
        assert manager.current_scheduler_type is None
        assert manager.pipeline is None
        
        # Check optimizations are all False initially
        opts = manager.optimizations
        assert all(v is False for v in opts.values())
