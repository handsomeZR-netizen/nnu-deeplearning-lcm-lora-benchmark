"""
Tests for ReportGenerator module.

Tests report generation, template filling, and LaTeX table generation.

Requirements: 13.5, 13.6, 13.7
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.core.models import ExperimentConfig, ExperimentSummary
from src.report.generator import ReportGenerator


@pytest.fixture
def sample_configs():
    """Create sample experiment configs."""
    return [
        ExperimentConfig(
            name="Euler_20",
            scheduler_type="euler",
            num_steps=20,
            guidance_scale=7.5,
            use_lcm_lora=False,
            optimizations={"attention_slicing": True},
        ),
        ExperimentConfig(
            name="LCM_4",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True,
            optimizations={"attention_slicing": True, "vae_slicing": True},
        ),
    ]


@pytest.fixture
def sample_summary(sample_configs):
    """Create sample experiment summary."""
    return ExperimentSummary(
        experiment_name="test_experiment",
        total_runs=10,
        configs=sample_configs,
        latency_stats={
            "Euler_20": {"mean": 2500.0, "std": 100.0, "min": 2300.0, "max": 2700.0},
            "LCM_4": {"mean": 500.0, "std": 20.0, "min": 480.0, "max": 520.0},
        },
        vram_stats={
            "Euler_20": {"mean": 4500.0, "std": 50.0, "min": 4400.0, "max": 4600.0},
            "LCM_4": {"mean": 4200.0, "std": 30.0, "min": 4150.0, "max": 4250.0},
        },
        quality_stats={
            "Euler_20": {"mean": 0.32, "std": 0.02, "min": 0.30, "max": 0.34},
            "LCM_4": {"mean": 0.30, "std": 0.02, "min": 0.28, "max": 0.32},
        },
        best_speed_config="LCM_4",
        best_quality_config="Euler_20",
        best_tradeoff_config="LCM_4",
        gpu_info="NVIDIA RTX 3080",
        cuda_version="11.8",
        pytorch_version="2.0.0",
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    def test_init(self, temp_output_dir):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        assert generator.output_dir == Path(temp_output_dir)
        assert generator.output_dir.exists()
    
    def test_generate_experiment_report(self, temp_output_dir, sample_summary):
        """Test full experiment report generation."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        report_path = generator.generate_experiment_report(
            experiment_summary=sample_summary,
            charts=["charts/comparison.png", "charts/steps_curve.png"],
            sample_images=["images/sample1.png"],
            csv_path="data/results.csv",
            json_path="data/results.json",
        )
        
        assert Path(report_path).exists()
        
        # Read and verify content
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check key sections exist
        assert "test_experiment" in content
        assert "实验报告" in content
        assert "NVIDIA RTX 3080" in content
        assert "Euler_20" in content
        assert "LCM_4" in content
        assert "延迟" in content
        assert "显存" in content
        assert "CLIPScore" in content
    
    def test_generate_minimal_report(self, temp_output_dir, sample_summary):
        """Test minimal report generation."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        report_path = generator.generate_minimal_report(
            experiment_summary=sample_summary,
        )
        
        assert Path(report_path).exists()
        
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "test_experiment" in content
        assert "LCM_4" in content
        assert "500.0" in content or "500" in content
    
    def test_generate_latex_comparison_table(self, temp_output_dir, sample_summary):
        """Test LaTeX comparison table generation."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        latex = generator.generate_latex_tables(
            experiment_summary=sample_summary,
            table_type="comparison",
            caption="Test Comparison",
            label="test_comparison",
        )
        
        # Check LaTeX structure
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\begin{tabular}" in latex
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex
        assert "Test Comparison" in latex
        assert "test_comparison" in latex
        assert "Euler_20" in latex
        assert "LCM_4" in latex
    
    def test_generate_latex_quality_table(self, temp_output_dir, sample_summary):
        """Test LaTeX quality table generation."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        latex = generator.generate_latex_tables(
            experiment_summary=sample_summary,
            table_type="quality",
        )
        
        assert "\\begin{table}" in latex
        assert "CLIPScore" in latex
    
    def test_export_latex_tables_to_file(self, temp_output_dir, sample_summary):
        """Test exporting LaTeX tables to file."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        latex_path = generator.export_latex_tables_to_file(
            experiment_summary=sample_summary,
        )
        
        assert Path(latex_path).exists()
        assert latex_path.endswith(".tex")
        
        with open(latex_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "\\begin{table}" in content
    
    def test_invalid_table_type(self, temp_output_dir, sample_summary):
        """Test error handling for invalid table type."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        with pytest.raises(ValueError, match="Unknown table type"):
            generator.generate_latex_tables(
                experiment_summary=sample_summary,
                table_type="invalid_type",
            )
    
    def test_empty_summary(self, temp_output_dir):
        """Test handling of empty experiment summary."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        empty_summary = ExperimentSummary(
            experiment_name="empty_test",
            total_runs=0,
            configs=[],
        )
        
        report_path = generator.generate_experiment_report(
            experiment_summary=empty_summary,
        )
        
        assert Path(report_path).exists()
        
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "empty_test" in content
        assert "无配置数据" in content or "N/A" in content


class TestReportTemplates:
    """Tests for report templates."""
    
    def test_configs_table_generation(self, temp_output_dir, sample_configs):
        """Test configuration table generation."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        table = generator._generate_configs_table(sample_configs)
        
        assert "Euler_20" in table
        assert "LCM_4" in table
        assert "euler" in table
        assert "lcm" in table
        assert "✓" in table  # LCM-LoRA enabled
        assert "✗" in table  # LCM-LoRA disabled
    
    def test_metrics_table_generation(self, temp_output_dir):
        """Test metrics table generation."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        stats = {
            "config1": {"mean": 100.0, "std": 10.0, "min": 90.0, "max": 110.0},
            "config2": {"mean": 200.0, "std": 20.0, "min": 180.0, "max": 220.0},
        }
        
        table = generator._generate_metrics_table(stats, "Test Metric")
        
        assert "config1" in table
        assert "config2" in table
        assert "100" in table
        assert "200" in table
    
    def test_empty_configs_table(self, temp_output_dir):
        """Test empty configuration table."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        table = generator._generate_configs_table([])
        
        assert "无配置数据" in table
    
    def test_empty_metrics_table(self, temp_output_dir):
        """Test empty metrics table."""
        generator = ReportGenerator(output_dir=temp_output_dir)
        
        table = generator._generate_metrics_table({}, "Test")
        
        assert "无" in table and "数据" in table
