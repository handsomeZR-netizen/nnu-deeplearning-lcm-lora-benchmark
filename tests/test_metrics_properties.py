"""
Property-based tests for MetricsCollector.

Tests correctness properties using hypothesis for property-based testing.
These tests require GPU and CLIP model to run.

Feature: lcm-lora-acceleration

NOTE: Property tests require:
- CUDA-capable GPU (or CPU fallback)
- transformers and lpips packages installed
- Run with: pytest tests/test_metrics_properties.py -v --tb=short
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

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None

# Check if required packages are available
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from PIL import Image

from src.metrics.collector import MetricsCollector
from src.core.models import QualityMetrics, RuntimeMetrics


# Strategy for generating random RGB images
@st.composite
def random_image_strategy(draw, min_size=64, max_size=128):
    """Generate random PIL RGB images with fixed size for efficiency"""
    # Use fixed small size to avoid health check issues
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate random RGB pixels using lists (more efficient for hypothesis)
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


# Strategy for generating random prompts
@st.composite
def random_prompt_strategy(draw):
    """Generate random text prompts"""
    words = [
        "a", "the", "beautiful", "simple", "red", "blue", "green", "yellow",
        "cat", "dog", "house", "tree", "car", "flower", "mountain", "ocean",
        "sunset", "portrait", "landscape", "abstract", "realistic", "painting",
        "photo", "digital", "art", "scene", "object", "person", "animal"
    ]
    num_words = draw(st.integers(min_value=2, max_value=10))
    selected_words = [draw(st.sampled_from(words)) for _ in range(num_words)]
    return " ".join(selected_words)


# Mark property tests to skip if dependencies not available
requires_clip = pytest.mark.skipif(
    not (TORCH_AVAILABLE and CLIP_AVAILABLE and HYPOTHESIS_AVAILABLE),
    reason="Requires torch, transformers (CLIP), and hypothesis"
)

requires_lpips = pytest.mark.skipif(
    not (TORCH_AVAILABLE and LPIPS_AVAILABLE and HYPOTHESIS_AVAILABLE),
    reason="Requires torch, lpips, and hypothesis"
)


@pytest.fixture(scope="module")
def metrics_collector():
    """
    Create a MetricsCollector instance.
    Module-scoped to avoid reloading models for each test.
    """
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    collector = MetricsCollector(device=device)
    
    yield collector
    
    # Cleanup
    collector.unload_models()


@requires_clip
class TestCLIPScoreRangeValidity:
    """
    Property 6: CLIPScore 范围有效性
    
    *For any* 计算的 CLIPScore，其值 SHALL 在 [0, 1] 范围内。
    
    **Validates: Requirements 5.1**
    """
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large]
    )
    @given(
        prompt=random_prompt_strategy(),
    )
    def test_clip_score_range_property(self, metrics_collector, prompt):
        """
        Feature: lcm-lora-acceleration, Property 6: CLIPScore 范围有效性
        
        For any image and prompt, CLIPScore should be in [0, 1] range.
        **Validates: Requirements 5.1**
        """
        # Generate random image directly (avoid hypothesis binary strategy)
        image = Image.fromarray(
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8), 
            mode='RGB'
        )
        
        # Compute CLIPScore
        clip_score = metrics_collector.compute_clip_score(image, prompt)
        
        # Verify range
        assert 0.0 <= clip_score <= 1.0, \
            f"CLIPScore {clip_score} is outside valid range [0, 1]"


# Additional unit tests for MetricsCollector
class TestMetricsCollectorBasics:
    """Basic unit tests for MetricsCollector - minimal dependencies"""
    
    def test_initialization(self):
        """Test MetricsCollector initialization"""
        collector = MetricsCollector(device="cpu")
        assert collector.device == "cpu"
        assert collector._clip_model is None  # Lazy loading
        assert collector._lpips_model is None
        assert collector._inception_model is None
    
    def test_runtime_metrics_collection(self):
        """Test runtime metrics collection"""
        collector = MetricsCollector(device="cpu")
        
        metrics = collector.collect_runtime_metrics(
            latency_ms=100.0,
            peak_vram_mb=1024.0,
            batch_size=1
        )
        
        assert isinstance(metrics, RuntimeMetrics)
        assert metrics.latency_ms == 100.0
        assert metrics.peak_vram_mb == 1024.0
        assert metrics.throughput == 10.0  # 1 / 0.1s = 10 images/s
    
    def test_runtime_metrics_throughput_calculation(self):
        """Test throughput calculation with different batch sizes"""
        collector = MetricsCollector(device="cpu")
        
        # Batch size 2, 200ms latency
        metrics = collector.collect_runtime_metrics(
            latency_ms=200.0,
            peak_vram_mb=2048.0,
            batch_size=2
        )
        
        # Throughput = 2 / 0.2s = 10 images/s
        assert metrics.throughput == 10.0
    
    def test_vram_usage_cpu(self):
        """Test VRAM usage returns zeros on CPU"""
        collector = MetricsCollector(device="cpu")
        
        vram = collector.get_vram_usage()
        
        assert vram["allocated_mb"] == 0.0
        assert vram["reserved_mb"] == 0.0


@requires_clip
class TestCLIPScoreComputation:
    """Unit tests for CLIPScore computation"""
    
    def test_clip_score_basic(self, metrics_collector):
        """Test basic CLIPScore computation"""
        # Create a simple test image
        image = Image.new("RGB", (64, 64), color="red")
        prompt = "a red square"
        
        score = metrics_collector.compute_clip_score(image, prompt)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_clip_score_batch(self, metrics_collector):
        """Test batch CLIPScore computation"""
        images = [
            Image.new("RGB", (64, 64), color="red"),
            Image.new("RGB", (64, 64), color="blue"),
        ]
        prompts = ["a red image", "a blue image"]
        
        scores = metrics_collector.compute_clip_score_batch(images, prompts)
        
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    def test_clip_score_batch_mismatch_raises_error(self, metrics_collector):
        """Test that mismatched batch sizes raise error"""
        images = [Image.new("RGB", (64, 64), color="red")]
        prompts = ["prompt1", "prompt2"]
        
        with pytest.raises(ValueError):
            metrics_collector.compute_clip_score_batch(images, prompts)


@requires_lpips
class TestLPIPSComputation:
    """Unit tests for LPIPS computation"""
    
    def test_lpips_identical_images(self, metrics_collector):
        """Test LPIPS for identical images should be near zero"""
        image = Image.new("RGB", (64, 64), color="red")
        
        lpips_value = metrics_collector.compute_lpips(image, image)
        
        assert isinstance(lpips_value, float)
        assert lpips_value < 0.01  # Should be very small for identical images
    
    def test_lpips_different_images(self, metrics_collector):
        """Test LPIPS for different images should be positive"""
        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (64, 64), color="blue")
        
        lpips_value = metrics_collector.compute_lpips(image1, image2)
        
        assert isinstance(lpips_value, float)
        assert lpips_value > 0.0  # Should be positive for different images
    
    def test_lpips_different_sizes(self, metrics_collector):
        """Test LPIPS handles different image sizes"""
        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (128, 128), color="red")
        
        # Should not raise error - images are resized
        lpips_value = metrics_collector.compute_lpips(image1, image2)
        
        assert isinstance(lpips_value, float)
    
    def test_lpips_batch(self, metrics_collector):
        """Test batch LPIPS computation"""
        images1 = [
            Image.new("RGB", (64, 64), color="red"),
            Image.new("RGB", (64, 64), color="blue"),
        ]
        images2 = [
            Image.new("RGB", (64, 64), color="green"),
            Image.new("RGB", (64, 64), color="yellow"),
        ]
        
        scores = metrics_collector.compute_lpips_batch(images1, images2)
        
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestQualityMetricsCollection:
    """Unit tests for quality metrics collection"""
    
    @requires_clip
    def test_collect_quality_metrics_clip_only(self, metrics_collector):
        """Test quality metrics collection with CLIPScore only"""
        image = Image.new("RGB", (64, 64), color="red")
        prompt = "a red square"
        
        metrics = metrics_collector.collect_quality_metrics(image, prompt)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.clip_score <= 1.0
        assert metrics.lpips is None
        assert metrics.fid is None
    
    @requires_clip
    @requires_lpips
    def test_collect_quality_metrics_with_lpips(self, metrics_collector):
        """Test quality metrics collection with LPIPS"""
        image = Image.new("RGB", (64, 64), color="red")
        reference = Image.new("RGB", (64, 64), color="blue")
        prompt = "a red square"
        
        metrics = metrics_collector.collect_quality_metrics(
            image, prompt, reference_image=reference
        )
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.clip_score <= 1.0
        assert metrics.lpips is not None
        assert metrics.lpips > 0.0
