"""
Property-based tests for ExperimentLogger.

Tests correctness properties using hypothesis for property-based testing.

Feature: lcm-lora-acceleration

Properties tested:
- Property 4: 指标记录完整性 (Validates: Requirements 4.1, 4.2, 13.2)
- Property 5: CSV 导出一致性 (Validates: Requirements 4.5)
"""

import csv
import json
import pytest
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if hypothesis is available
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    st = None

from src.benchmark.logger import ExperimentLogger
from src.core.models import GenerationResult, ExperimentConfig


def create_generation_result(
    prompt: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    resolution: tuple,
    latency_ms: float,
    peak_vram_mb: float,
    scheduler_type: str,
    optimizations: Dict[str, bool]
) -> GenerationResult:
    """Helper to create a GenerationResult for testing"""
    return GenerationResult(
        image=None,  # No image for testing
        prompt=prompt,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        resolution=resolution,
        latency_ms=latency_ms,
        peak_vram_mb=peak_vram_mb,
        scheduler_type=scheduler_type,
        optimizations=optimizations,
        timestamp=datetime.now()
    )


# Define strategies only if hypothesis is available
if HYPOTHESIS_AVAILABLE:
    # Strategy for generating valid prompts
    prompt_strategy = st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S', 'Z')),
        min_size=1,
        max_size=200
    ).filter(lambda x: x.strip())
    
    # Strategy for generating seeds
    seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)
    
    # Strategy for generating num_steps
    steps_strategy = st.integers(min_value=1, max_value=50)
    
    # Strategy for generating guidance_scale
    guidance_strategy = st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False)
    
    # Strategy for generating resolution
    resolution_strategy = st.tuples(
        st.sampled_from([256, 512, 768, 1024]),
        st.sampled_from([256, 512, 768, 1024])
    )
    
    # Strategy for generating latency
    latency_strategy = st.floats(min_value=0.1, max_value=60000.0, allow_nan=False, allow_infinity=False)
    
    # Strategy for generating VRAM
    vram_strategy = st.floats(min_value=0.0, max_value=48000.0, allow_nan=False, allow_infinity=False)
    
    # Strategy for generating scheduler type
    scheduler_strategy = st.sampled_from(["euler", "dpm_solver", "ddim", "lcm"])
    
    # Strategy for generating optimizations dict
    optimizations_strategy = st.fixed_dictionaries({
        "attention_slicing": st.booleans(),
        "vae_slicing": st.booleans(),
        "vae_tiling": st.booleans(),
        "xformers": st.booleans(),
        "sdpa": st.booleans(),
    })
    
    # Composite strategy for a single result
    result_data_strategy = st.tuples(
        prompt_strategy,
        seed_strategy,
        steps_strategy,
        guidance_strategy,
        resolution_strategy,
        latency_strategy,
        vram_strategy,
        scheduler_strategy,
        optimizations_strategy,
    )


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestLogRecordCompleteness:
    """
    Property 4: 指标记录完整性
    
    *For any* 完成的生成任务，日志 SHALL 包含 latency、peak_vram、prompt、seed、config 等完整信息。
    
    **Validates: Requirements 4.1, 4.2, 13.2**
    """
    
    def test_log_result_completeness_property(self):
        """
        Feature: lcm-lora-acceleration, Property 4: 指标记录完整性
        
        For any generation result, the logged entry SHALL contain all required fields:
        latency, peak_vram, prompt, seed, and config information.
        
        **Validates: Requirements 4.1, 4.2, 13.2**
        """
        @settings(max_examples=100, deadline=None)
        @given(
            prompt=prompt_strategy,
            seed=seed_strategy,
            num_steps=steps_strategy,
            guidance_scale=guidance_strategy,
            resolution=resolution_strategy,
            latency_ms=latency_strategy,
            peak_vram_mb=vram_strategy,
            scheduler_type=scheduler_strategy,
            optimizations=optimizations_strategy,
        )
        def inner_test(
            prompt: str,
            seed: int,
            num_steps: int,
            guidance_scale: float,
            resolution: tuple,
            latency_ms: float,
            peak_vram_mb: float,
            scheduler_type: str,
            optimizations: Dict[str, bool]
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_completeness")
                
                # Create and log a generation result
                result = create_generation_result(
                    prompt=prompt,
                    seed=seed,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    resolution=resolution,
                    latency_ms=latency_ms,
                    peak_vram_mb=peak_vram_mb,
                    scheduler_type=scheduler_type,
                    optimizations=optimizations,
                )
                
                logger.log_result(result)
                
                # Retrieve logged results
                logged_results = logger.get_all_results()
                
                # Verify exactly one result was logged
                assert len(logged_results) == 1, "Expected exactly one logged result"
                
                logged = logged_results[0]
                
                # Verify all required fields are present
                required_fields = [
                    "timestamp",
                    "experiment_id",
                    "experiment_name",
                    "prompt",
                    "seed",
                    "num_steps",
                    "guidance_scale",
                    "resolution",
                    "latency_ms",
                    "peak_vram_mb",
                    "scheduler_type",
                    "optimizations",
                ]
                
                for field in required_fields:
                    assert field in logged, f"Missing required field: {field}"
                
                # Verify field values match input
                assert logged["prompt"] == prompt, "Prompt mismatch"
                assert logged["seed"] == seed, "Seed mismatch"
                assert logged["num_steps"] == num_steps, "num_steps mismatch"
                assert abs(logged["guidance_scale"] - guidance_scale) < 1e-6, "guidance_scale mismatch"
                assert tuple(logged["resolution"]) == resolution, "Resolution mismatch"
                assert abs(logged["latency_ms"] - latency_ms) < 1e-6, "latency_ms mismatch"
                assert abs(logged["peak_vram_mb"] - peak_vram_mb) < 1e-6, "peak_vram_mb mismatch"
                assert logged["scheduler_type"] == scheduler_type, "scheduler_type mismatch"
                assert logged["optimizations"] == optimizations, "optimizations mismatch"
        
        inner_test()


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestCSVExportConsistency:
    """
    Property 5: CSV 导出一致性
    
    *For any* 导出的 CSV 文件，解析后的数据 SHALL 与内存中的实验结果完全一致。
    
    **Validates: Requirements 4.5**
    """
    
    def test_csv_export_consistency_property(self):
        """
        Feature: lcm-lora-acceleration, Property 5: CSV 导出一致性
        
        For any set of logged results, exporting to CSV and parsing back
        SHALL produce data consistent with the original results.
        
        **Validates: Requirements 4.5**
        """
        @settings(max_examples=100, deadline=None)
        @given(
            results_data=st.lists(
                result_data_strategy,
                min_size=1,
                max_size=20
            )
        )
        def inner_test(results_data):
            with tempfile.TemporaryDirectory() as tmpdir:
                logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_csv")
                
                # Log multiple results
                for (prompt, seed, num_steps, guidance_scale, resolution,
                     latency_ms, peak_vram_mb, scheduler_type, optimizations) in results_data:
                    
                    result = create_generation_result(
                        prompt=prompt,
                        seed=seed,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        resolution=resolution,
                        latency_ms=latency_ms,
                        peak_vram_mb=peak_vram_mb,
                        scheduler_type=scheduler_type,
                        optimizations=optimizations,
                    )
                    logger.log_result(result)
                
                # Export to CSV
                csv_path = logger.export_csv("test_results.csv")
                
                # Read back the CSV
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    csv_rows = list(reader)
                
                # Get original results
                original_results = logger.get_all_results()
                
                # Verify same number of rows
                assert len(csv_rows) == len(original_results), \
                    f"Row count mismatch: CSV has {len(csv_rows)}, memory has {len(original_results)}"
                
                # Verify each row matches
                for i, (csv_row, original) in enumerate(zip(csv_rows, original_results)):
                    # Check key fields match
                    assert csv_row["prompt"] == original["prompt"], \
                        f"Row {i}: prompt mismatch"
                    assert int(csv_row["seed"]) == original["seed"], \
                        f"Row {i}: seed mismatch"
                    assert int(csv_row["num_steps"]) == original["num_steps"], \
                        f"Row {i}: num_steps mismatch"
                    assert abs(float(csv_row["guidance_scale"]) - original["guidance_scale"]) < 1e-6, \
                        f"Row {i}: guidance_scale mismatch"
                    assert int(csv_row["resolution_w"]) == original["resolution"][0], \
                        f"Row {i}: resolution_w mismatch"
                    assert int(csv_row["resolution_h"]) == original["resolution"][1], \
                        f"Row {i}: resolution_h mismatch"
                    assert abs(float(csv_row["latency_ms"]) - original["latency_ms"]) < 1e-6, \
                        f"Row {i}: latency_ms mismatch"
                    assert abs(float(csv_row["peak_vram_mb"]) - original["peak_vram_mb"]) < 1e-6, \
                        f"Row {i}: peak_vram_mb mismatch"
                    assert csv_row["scheduler_type"] == original["scheduler_type"], \
                        f"Row {i}: scheduler_type mismatch"
                    
                    # Parse and verify optimizations
                    csv_optimizations = json.loads(csv_row["optimizations"])
                    assert csv_optimizations == original["optimizations"], \
                        f"Row {i}: optimizations mismatch"
        
        inner_test()



# Additional unit tests for ExperimentLogger
class TestExperimentLoggerBasics:
    """Basic unit tests for ExperimentLogger - no hypothesis required"""
    
    def test_initialization(self):
        """Test logger initialization creates directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = ExperimentLogger(log_dir=str(log_dir), experiment_name="test")
            
            assert log_dir.exists()
            assert logger.experiment_name == "test"
            assert len(logger.experiment_id) == 8
            assert logger.results_count == 0
            assert logger.configs_count == 0
    
    def test_log_config(self):
        """Test logging experiment configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test")
            
            config = {
                "name": "LCM_4",
                "scheduler_type": "lcm",
                "num_steps": 4,
                "guidance_scale": 1.0,
                "use_lcm_lora": True,
            }
            
            logger.log_config(config)
            
            assert logger.configs_count == 1
            configs = logger.get_all_configs()
            assert configs[0]["config"] == config
    
    def test_log_metrics(self):
        """Test logging quality metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test")
            
            metrics = {
                "clip_score": 0.85,
                "fid": 25.3,
                "config_name": "LCM_4",
            }
            
            logger.log_metrics(metrics)
            
            all_metrics = logger.get_all_metrics()
            assert len(all_metrics) == 1
            assert all_metrics[0]["metrics"] == metrics
    
    def test_export_json(self):
        """Test JSON export contains all data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_json")
            
            # Log some data
            logger.log_config({"name": "test_config"})
            
            result = create_generation_result(
                prompt="test prompt",
                seed=42,
                num_steps=4,
                guidance_scale=1.0,
                resolution=(512, 512),
                latency_ms=200.0,
                peak_vram_mb=3000.0,
                scheduler_type="lcm",
                optimizations={"attention_slicing": True},
            )
            logger.log_result(result)
            logger.log_metrics({"clip_score": 0.8})
            
            # Export
            json_path = logger.export_json("test.json")
            
            # Read and verify
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert data["experiment_name"] == "test_json"
            assert len(data["configs"]) == 1
            assert len(data["results"]) == 1
            assert len(data["metrics"]) == 1
    
    def test_generate_summary_empty(self):
        """Test summary generation with no data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_empty")
            
            summary = logger.generate_summary()
            
            assert summary.experiment_name == "test_empty"
            assert summary.total_runs == 0
            assert len(summary.configs) == 0
    
    def test_generate_summary_with_data(self):
        """Test summary generation with logged data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_summary")
            
            # Log multiple results for same config
            for i in range(3):
                result = create_generation_result(
                    prompt=f"test prompt {i}",
                    seed=i,
                    num_steps=4,
                    guidance_scale=1.0,
                    resolution=(512, 512),
                    latency_ms=200.0 + i * 10,
                    peak_vram_mb=3000.0,
                    scheduler_type="lcm",
                    optimizations={},
                )
                logger.log_result(result)
            
            summary = logger.generate_summary()
            
            assert summary.total_runs == 3
            assert "lcm_4" in summary.latency_stats
            assert summary.latency_stats["lcm_4"]["mean"] == 210.0  # (200 + 210 + 220) / 3
    
    def test_export_csv_empty(self):
        """Test CSV export with no results creates empty file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_empty")
            
            csv_path = logger.export_csv("empty.csv")
            
            assert csv_path.exists()
