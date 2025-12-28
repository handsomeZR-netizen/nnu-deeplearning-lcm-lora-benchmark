"""
Property-based tests for BenchmarkRunner.

Tests correctness properties using hypothesis for property-based testing.
These tests require GPU and actual model files to run.

Feature: lcm-lora-acceleration

NOTE: Property tests require:
- CUDA-capable GPU
- Model files in models/dreamshaper-7 and models/lcm-lora-sdv1-5
- Run with: pytest tests/test_benchmark_properties.py -v -k "property" --tb=short
"""

import pytest
import sys
from pathlib import Path
from typing import Tuple

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

from src.core.pipeline import PipelineManager, ModelLoadError
from src.core.models import ExperimentConfig, GenerationResult
from src.benchmark.runner import BenchmarkRunner, ExperimentResults


# Model paths - adjust these to your local setup
MODEL_DIR = "models/dreamshaper-7"
LCM_LORA_DIR = "models/lcm-lora-sdv1-5"
OUTPUT_DIR = "outputs/benchmark_tests"


def models_exist() -> bool:
    """Check if model files exist"""
    return Path(MODEL_DIR).exists() and Path(LCM_LORA_DIR).exists()


# Mark property tests to skip by default (require GPU + models)
requires_gpu_and_models = pytest.mark.skipif(
    not (models_exist() and CUDA_AVAILABLE and HYPOTHESIS_AVAILABLE),
    reason="Requires GPU, models, and hypothesis"
)


@pytest.fixture(scope="module")
def pipeline_manager():
    """
    Create a PipelineManager for benchmark tests.
    Module-scoped to avoid reloading for each test.
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


@pytest.fixture(scope="module")
def benchmark_runner(pipeline_manager):
    """
    Create a BenchmarkRunner for tests.
    """
    runner = BenchmarkRunner(
        pipeline_manager=pipeline_manager,
        output_dir=OUTPUT_DIR,
    )
    
    return runner


@requires_gpu_and_models
class TestLCMAccelerationEffectiveness:
    """
    Property 3: LCM 加速有效性
    
    *For any* 使用 LCM-LoRA 的 4 步配置，推理延迟 SHALL 小于 Euler 20 步配置的延迟。
    
    **Validates: Requirements 2.3, 6.3**
    """
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @settings(
        max_examples=3,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_lcm_faster_than_euler_property(self, pipeline_manager, seed):
        """
        Feature: lcm-lora-acceleration, Property 3: LCM 加速有效性
        
        For any seed, LCM 4-step generation should be faster than Euler 20-step generation.
        **Validates: Requirements 2.3, 6.3**
        """
        prompt = "a simple landscape with mountains"
        width, height = 512, 512
        
        # Test with Euler 20 steps (baseline)
        pipeline_manager.unload()
        pipeline_manager.load_baseline_pipeline(scheduler_type="euler")
        pipeline_manager.apply_optimizations(
            attention_slicing=True,
            vae_slicing=True,
        )
        pipeline_manager.warmup(num_steps=2)
        
        euler_result = pipeline_manager.generate(
            prompt=prompt,
            num_steps=20,
            guidance_scale=7.5,
            seed=seed,
            width=width,
            height=height,
        )
        euler_latency = euler_result.latency_ms
        
        # Test with LCM 4 steps
        pipeline_manager.unload()
        pipeline_manager.load_lcm_pipeline(fuse_lora=True)
        pipeline_manager.apply_optimizations(
            attention_slicing=True,
            vae_slicing=True,
        )
        pipeline_manager.warmup(num_steps=2)
        
        lcm_result = pipeline_manager.generate(
            prompt=prompt,
            num_steps=4,
            guidance_scale=1.0,
            seed=seed,
            width=width,
            height=height,
        )
        lcm_latency = lcm_result.latency_ms
        
        # LCM should be faster than Euler
        # Allow some tolerance for measurement variance
        assert lcm_latency < euler_latency, \
            f"LCM 4-step ({lcm_latency:.1f}ms) should be faster than Euler 20-step ({euler_latency:.1f}ms)"


# Additional unit tests for BenchmarkRunner
class TestBenchmarkRunnerBasics:
    """Basic unit tests for BenchmarkRunner - no GPU required"""
    
    def test_default_comparison_configs(self):
        """Test that default comparison configs are properly defined"""
        configs = BenchmarkRunner.DEFAULT_COMPARISON_CONFIGS
        
        assert len(configs) >= 4, "Should have at least 4 default configs"
        
        # Check for required configs
        config_names = [c.name for c in configs]
        assert "Euler_20" in config_names, "Should have Euler_20 config"
        assert "DPM_Solver_20" in config_names, "Should have DPM_Solver_20 config"
        assert "LCM_4" in config_names, "Should have LCM_4 config"
    
    def test_experiment_config_serialization(self):
        """Test ExperimentConfig serialization/deserialization"""
        config = ExperimentConfig(
            name="test_config",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True,
            optimizations={"attention_slicing": True},
        )
        
        # Serialize
        data = config.to_dict()
        assert data["name"] == "test_config"
        assert data["scheduler_type"] == "lcm"
        assert data["num_steps"] == 4
        
        # Deserialize
        restored = ExperimentConfig.from_dict(data)
        assert restored.name == config.name
        assert restored.scheduler_type == config.scheduler_type
        assert restored.num_steps == config.num_steps
        assert restored.use_lcm_lora == config.use_lcm_lora
    
    @pytest.mark.skipif(not models_exist(), reason="Models not available")
    def test_benchmark_runner_initialization(self):
        """Test BenchmarkRunner initialization"""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        
        manager = PipelineManager(
            model_dir=MODEL_DIR,
            lcm_lora_dir=LCM_LORA_DIR,
            device="cuda"
        )
        
        runner = BenchmarkRunner(
            pipeline_manager=manager,
            output_dir=OUTPUT_DIR,
        )
        
        assert runner.pipeline_manager is manager
        assert runner.output_dir.exists()
        
        manager.unload()
