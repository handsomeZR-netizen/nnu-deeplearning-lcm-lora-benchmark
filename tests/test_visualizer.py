"""
Tests for the Visualizer module.

Validates chart generation for comparison experiments, ablation studies,
parameter sensitivity analysis, and image comparison grids.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from src.visualization import Visualizer
from src.core.models import ExperimentConfig


# Mock data classes for testing
@dataclass
class MockExperimentResults:
    """Mock ExperimentResults for testing"""
    experiment_name: str = "test_experiment"
    configs: List[ExperimentConfig] = field(default_factory=list)
    results: Dict[str, List[Any]] = field(default_factory=dict)
    quality_metrics: Dict[str, List[Any]] = field(default_factory=dict)
    runtime_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vram_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MockAblationResults:
    """Mock AblationResults for testing"""
    experiment_name: str = "test_ablation"
    configs: List[Any] = field(default_factory=list)
    results: Dict[str, List[Any]] = field(default_factory=dict)
    runtime_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vram_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    contributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MockParameterResults:
    """Mock ParameterResults for testing"""
    experiment_name: str = "test_param"
    parameter_name: str = "guidance_scale"
    parameter_values: List[Any] = field(default_factory=list)
    results: Dict[str, List[Any]] = field(default_factory=dict)
    runtime_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vram_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def visualizer(temp_output_dir):
    """Create a Visualizer instance"""
    return Visualizer(output_dir=temp_output_dir, style="paper")


@pytest.fixture
def mock_experiment_results():
    """Create mock experiment results"""
    configs = [
        ExperimentConfig(
            name="Euler_20",
            scheduler_type="euler",
            num_steps=20,
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
        ExperimentConfig(
            name="LCM_8",
            scheduler_type="lcm",
            num_steps=8,
            guidance_scale=1.0,
            use_lcm_lora=True,
        ),
    ]
    
    return MockExperimentResults(
        configs=configs,
        runtime_stats={
            "Euler_20": {"mean": 2500.0, "std": 100.0, "min": 2400.0, "max": 2600.0},
            "LCM_4": {"mean": 500.0, "std": 20.0, "min": 480.0, "max": 520.0},
            "LCM_8": {"mean": 900.0, "std": 30.0, "min": 870.0, "max": 930.0},
        },
        vram_stats={
            "Euler_20": {"mean": 4500.0, "std": 50.0, "min": 4450.0, "max": 4550.0},
            "LCM_4": {"mean": 4200.0, "std": 40.0, "min": 4160.0, "max": 4240.0},
            "LCM_8": {"mean": 4300.0, "std": 45.0, "min": 4255.0, "max": 4345.0},
        },
        quality_stats={
            "Euler_20": {"mean": 0.32, "std": 0.02, "min": 0.30, "max": 0.34},
            "LCM_4": {"mean": 0.28, "std": 0.03, "min": 0.25, "max": 0.31},
            "LCM_8": {"mean": 0.30, "std": 0.02, "min": 0.28, "max": 0.32},
        },
    )


@pytest.fixture
def mock_ablation_results():
    """Create mock ablation results"""
    return MockAblationResults(
        runtime_stats={
            "full_optimization": {"mean": 500.0, "std": 20.0, "min": 480.0, "max": 520.0},
            "no_lcm_lora": {"mean": 2500.0, "std": 100.0, "min": 2400.0, "max": 2600.0},
            "no_sdpa": {"mean": 600.0, "std": 25.0, "min": 575.0, "max": 625.0},
            "no_attention_slicing": {"mean": 520.0, "std": 22.0, "min": 498.0, "max": 542.0},
        },
        vram_stats={
            "full_optimization": {"mean": 4200.0, "std": 40.0, "min": 4160.0, "max": 4240.0},
            "no_lcm_lora": {"mean": 4500.0, "std": 50.0, "min": 4450.0, "max": 4550.0},
            "no_sdpa": {"mean": 4800.0, "std": 60.0, "min": 4740.0, "max": 4860.0},
            "no_attention_slicing": {"mean": 5000.0, "std": 70.0, "min": 4930.0, "max": 5070.0},
        },
    )


@pytest.fixture
def mock_parameter_results():
    """Create mock parameter results"""
    return MockParameterResults(
        parameter_name="guidance_scale",
        parameter_values=[0.0, 1.0, 1.5, 2.0],
        runtime_stats={
            "0.0": {"mean": 480.0, "std": 15.0, "min": 465.0, "max": 495.0},
            "1.0": {"mean": 500.0, "std": 20.0, "min": 480.0, "max": 520.0},
            "1.5": {"mean": 510.0, "std": 18.0, "min": 492.0, "max": 528.0},
            "2.0": {"mean": 520.0, "std": 22.0, "min": 498.0, "max": 542.0},
        },
        vram_stats={
            "0.0": {"mean": 4100.0, "std": 35.0, "min": 4065.0, "max": 4135.0},
            "1.0": {"mean": 4200.0, "std": 40.0, "min": 4160.0, "max": 4240.0},
            "1.5": {"mean": 4250.0, "std": 42.0, "min": 4208.0, "max": 4292.0},
            "2.0": {"mean": 4300.0, "std": 45.0, "min": 4255.0, "max": 4345.0},
        },
        quality_stats={
            "0.0": {"mean": 0.25, "std": 0.03, "min": 0.22, "max": 0.28},
            "1.0": {"mean": 0.28, "std": 0.02, "min": 0.26, "max": 0.30},
            "1.5": {"mean": 0.30, "std": 0.02, "min": 0.28, "max": 0.32},
            "2.0": {"mean": 0.29, "std": 0.03, "min": 0.26, "max": 0.32},
        },
    )


class TestVisualizerInit:
    """Test Visualizer initialization"""
    
    def test_init_creates_output_dir(self, temp_output_dir):
        """Test that Visualizer creates output directory"""
        output_path = os.path.join(temp_output_dir, "charts")
        viz = Visualizer(output_dir=output_path)
        assert os.path.exists(output_path)
    
    def test_init_with_paper_style(self, temp_output_dir):
        """Test initialization with paper style"""
        viz = Visualizer(output_dir=temp_output_dir, style="paper")
        assert viz.style == "paper"
    
    def test_init_with_presentation_style(self, temp_output_dir):
        """Test initialization with presentation style"""
        viz = Visualizer(output_dir=temp_output_dir, style="presentation")
        assert viz.style == "presentation"


class TestComparisonBars:
    """Test plot_comparison_bars method - Requirements: 11.1"""
    
    def test_generates_png_and_pdf(self, visualizer, mock_experiment_results):
        """Test that comparison bars generates PNG and PDF files"""
        saved_paths = visualizer.plot_comparison_bars(
            results=mock_experiment_results,
            metrics=['latency', 'vram', 'clip_score'],
            filename="test_comparison",
        )
        
        assert len(saved_paths) == 2
        assert any(p.endswith('.png') for p in saved_paths)
        assert any(p.endswith('.pdf') for p in saved_paths)
        
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_handles_single_metric(self, visualizer, mock_experiment_results):
        """Test with single metric"""
        saved_paths = visualizer.plot_comparison_bars(
            results=mock_experiment_results,
            metrics=['latency'],
            filename="test_single_metric",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_handles_empty_results(self, visualizer):
        """Test with empty results"""
        empty_results = MockExperimentResults()
        saved_paths = visualizer.plot_comparison_bars(
            results=empty_results,
            filename="test_empty",
        )
        
        assert saved_paths == []


class TestStepsCurve:
    """Test plot_steps_curve method - Requirements: 11.3"""
    
    def test_generates_steps_curve(self, visualizer, mock_experiment_results):
        """Test that steps curve is generated"""
        saved_paths = visualizer.plot_steps_curve(
            results=mock_experiment_results,
            filename="test_steps_curve",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_handles_empty_results(self, visualizer):
        """Test with empty results"""
        empty_results = MockExperimentResults()
        saved_paths = visualizer.plot_steps_curve(
            results=empty_results,
            filename="test_empty_steps",
        )
        
        assert saved_paths == []


class TestAblationTable:
    """Test plot_ablation_table method - Requirements: 11.2"""
    
    def test_generates_ablation_table(self, visualizer, mock_ablation_results):
        """Test that ablation table is generated"""
        saved_paths = visualizer.plot_ablation_table(
            results=mock_ablation_results,
            filename="test_ablation_table",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_handles_empty_results(self, visualizer):
        """Test with empty results"""
        empty_results = MockAblationResults()
        saved_paths = visualizer.plot_ablation_table(
            results=empty_results,
            filename="test_empty_ablation",
        )
        
        assert saved_paths == []


class TestParameterSensitivity:
    """Test plot_parameter_sensitivity method - Requirements: 11.4"""
    
    def test_generates_parameter_sensitivity(self, visualizer, mock_parameter_results):
        """Test that parameter sensitivity plot is generated"""
        saved_paths = visualizer.plot_parameter_sensitivity(
            results=mock_parameter_results,
            filename="test_param_sensitivity",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_handles_dict_input(self, visualizer, mock_parameter_results):
        """Test with dict of parameter results"""
        param_dict = {"guidance_scale": mock_parameter_results}
        saved_paths = visualizer.plot_parameter_sensitivity(
            results=param_dict,
            filename="test_param_dict",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)


class TestComparisonGrid:
    """Test create_comparison_grid method - Requirements: 11.5"""
    
    def test_generates_single_prompt_grid(self, visualizer):
        """Test single prompt comparison grid"""
        # Create test images
        images = {
            "Euler_20": Image.new('RGB', (64, 64), color='red'),
            "LCM_4": Image.new('RGB', (64, 64), color='green'),
            "LCM_8": Image.new('RGB', (64, 64), color='blue'),
        }
        
        saved_paths = visualizer.create_comparison_grid(
            images=images,
            prompts=["A test prompt"],
            filename="test_single_grid",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_generates_multi_prompt_grid(self, visualizer):
        """Test multi-prompt comparison grid"""
        # Create test images
        images = {
            "Euler_20": [
                Image.new('RGB', (64, 64), color='red'),
                Image.new('RGB', (64, 64), color='darkred'),
            ],
            "LCM_4": [
                Image.new('RGB', (64, 64), color='green'),
                Image.new('RGB', (64, 64), color='darkgreen'),
            ],
        }
        
        saved_paths = visualizer.create_comparison_grid(
            images=images,
            prompts=["Prompt 1", "Prompt 2"],
            filename="test_multi_grid",
        )
        
        assert len(saved_paths) == 2
        for path in saved_paths:
            assert os.path.exists(path)
    
    def test_handles_empty_images(self, visualizer):
        """Test with empty images dict"""
        saved_paths = visualizer.create_comparison_grid(
            images={},
            filename="test_empty_grid",
        )
        
        assert saved_paths == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
