"""
Unit tests for core data models.
Tests serialization/deserialization consistency.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import (
    RuntimeMetrics,
    QualityMetrics,
    ExperimentConfig,
    GenerationResult,
    ExperimentSummary
)


class TestRuntimeMetrics:
    """Tests for RuntimeMetrics serialization"""
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict -> from_dict produces equivalent object"""
        original = RuntimeMetrics(
            latency_ms=150.5,
            peak_vram_mb=4096.0,
            throughput=2.5
        )
        
        serialized = original.to_dict()
        restored = RuntimeMetrics.from_dict(serialized)
        
        assert restored.latency_ms == original.latency_ms
        assert restored.peak_vram_mb == original.peak_vram_mb
        assert restored.throughput == original.throughput


class TestQualityMetrics:
    """Tests for QualityMetrics serialization"""
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test full metrics roundtrip"""
        original = QualityMetrics(
            clip_score=0.85,
            fid=25.3,
            lpips=0.12
        )
        
        serialized = original.to_dict()
        restored = QualityMetrics.from_dict(serialized)
        
        assert restored.clip_score == original.clip_score
        assert restored.fid == original.fid
        assert restored.lpips == original.lpips
    
    def test_optional_fields_none(self):
        """Test with optional fields as None"""
        original = QualityMetrics(clip_score=0.75)
        
        serialized = original.to_dict()
        restored = QualityMetrics.from_dict(serialized)
        
        assert restored.clip_score == original.clip_score
        assert restored.fid is None
        assert restored.lpips is None


class TestExperimentConfig:
    """Tests for ExperimentConfig serialization"""
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test config roundtrip"""
        original = ExperimentConfig(
            name="LCM_4",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True,
            optimizations={
                "attention_slicing": True,
                "vae_slicing": True,
                "xformers": False
            }
        )
        
        serialized = original.to_dict()
        restored = ExperimentConfig.from_dict(serialized)
        
        assert restored.name == original.name
        assert restored.scheduler_type == original.scheduler_type
        assert restored.num_steps == original.num_steps
        assert restored.guidance_scale == original.guidance_scale
        assert restored.use_lcm_lora == original.use_lcm_lora
        assert restored.optimizations == original.optimizations


class TestGenerationResult:
    """Tests for GenerationResult serialization"""
    
    def test_to_dict_from_dict_without_image(self):
        """Test roundtrip without image"""
        original = GenerationResult(
            image=None,
            prompt="a beautiful sunset",
            seed=42,
            num_steps=4,
            guidance_scale=1.0,
            resolution=(512, 512),
            latency_ms=200.0,
            peak_vram_mb=3500.0,
            scheduler_type="lcm",
            optimizations={"attention_slicing": True},
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        serialized = original.to_dict(include_image=False)
        restored = GenerationResult.from_dict(serialized)
        
        assert restored.prompt == original.prompt
        assert restored.seed == original.seed
        assert restored.num_steps == original.num_steps
        assert restored.guidance_scale == original.guidance_scale
        assert restored.resolution == original.resolution
        assert restored.latency_ms == original.latency_ms
        assert restored.peak_vram_mb == original.peak_vram_mb
        assert restored.scheduler_type == original.scheduler_type
        assert restored.optimizations == original.optimizations
        assert restored.timestamp == original.timestamp
    
    def test_to_dict_from_dict_with_image(self):
        """Test roundtrip with image included"""
        # Create a simple test image
        test_image = Image.new("RGB", (64, 64), color="red")
        
        original = GenerationResult(
            image=test_image,
            prompt="test prompt",
            seed=123,
            num_steps=4,
            guidance_scale=1.5,
            resolution=(64, 64),
            latency_ms=100.0,
            peak_vram_mb=2000.0,
            scheduler_type="lcm",
            optimizations={},
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        serialized = original.to_dict(include_image=True)
        assert "image_base64" in serialized
        
        restored = GenerationResult.from_dict(serialized)
        
        assert restored.image is not None
        assert restored.image.size == original.image.size
        assert restored.prompt == original.prompt


class TestExperimentSummary:
    """Tests for ExperimentSummary serialization"""
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test full summary roundtrip"""
        config1 = ExperimentConfig(
            name="Euler_20",
            scheduler_type="euler",
            num_steps=20,
            guidance_scale=7.5,
            use_lcm_lora=False
        )
        config2 = ExperimentConfig(
            name="LCM_4",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True
        )
        
        original = ExperimentSummary(
            experiment_name="comparison_test",
            total_runs=10,
            configs=[config1, config2],
            latency_stats={
                "Euler_20": {"mean": 2500.0, "std": 50.0},
                "LCM_4": {"mean": 500.0, "std": 20.0}
            },
            vram_stats={
                "Euler_20": {"mean": 4000.0, "std": 100.0},
                "LCM_4": {"mean": 3800.0, "std": 80.0}
            },
            quality_stats={
                "Euler_20": {"mean": 0.85, "std": 0.02},
                "LCM_4": {"mean": 0.82, "std": 0.03}
            },
            best_speed_config="LCM_4",
            best_quality_config="Euler_20",
            best_tradeoff_config="LCM_4",
            gpu_info="NVIDIA RTX 3060",
            cuda_version="12.1",
            pytorch_version="2.1.0"
        )
        
        serialized = original.to_dict()
        restored = ExperimentSummary.from_dict(serialized)
        
        assert restored.experiment_name == original.experiment_name
        assert restored.total_runs == original.total_runs
        assert len(restored.configs) == len(original.configs)
        assert restored.configs[0].name == original.configs[0].name
        assert restored.latency_stats == original.latency_stats
        assert restored.vram_stats == original.vram_stats
        assert restored.quality_stats == original.quality_stats
        assert restored.best_speed_config == original.best_speed_config
        assert restored.gpu_info == original.gpu_info
    
    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip"""
        original = ExperimentSummary(
            experiment_name="json_test",
            total_runs=5,
            configs=[],
            gpu_info="Test GPU"
        )
        
        json_str = original.to_json()
        restored = ExperimentSummary.from_json(json_str)
        
        assert restored.experiment_name == original.experiment_name
        assert restored.total_runs == original.total_runs
        assert restored.gpu_info == original.gpu_info
