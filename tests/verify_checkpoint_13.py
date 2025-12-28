"""
Checkpoint 13 verification script - System Integration Verification.

Verifies:
1. Gradio interface can be built and components work correctly
2. All modules are properly integrated
3. All existing tests pass

Feature: lcm-lora-acceleration
Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 12.1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_module_imports():
    """Verify all modules can be imported correctly."""
    print("\n" + "="*60)
    print("Verifying Module Imports")
    print("="*60)
    
    modules_to_check = [
        ("src.core.models", "Core data models"),
        ("src.core.pipeline", "Pipeline manager"),
        ("src.benchmark.runner", "Benchmark runner"),
        ("src.benchmark.logger", "Experiment logger"),
        ("src.metrics.collector", "Metrics collector"),
        ("src.visualization.visualizer", "Visualizer"),
        ("src.report.generator", "Report generator"),
        ("src.report.templates", "Report templates"),
        ("src.dataset.builder", "Dataset builder"),
        ("src.ui.app", "Gradio app"),
    ]
    
    all_passed = True
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {description} ({module_name})")
        except ImportError as e:
            print(f"  ✗ {description} ({module_name}): {e}")
            all_passed = False
    
    return all_passed


def verify_gradio_interface_build():
    """Verify Gradio interface can be built without errors."""
    print("\n" + "="*60)
    print("Verifying Gradio Interface Build")
    print("="*60)
    
    try:
        from src.ui.app import GradioApp, create_app
        
        # Test factory function
        print("  Testing create_app factory function...")
        app = create_app(
            model_dir="models/dreamshaper-7",
            lcm_lora_dir="models/lcm-lora-sdv1-5",
            output_dir="outputs/test_gradio",
            device="cuda"
        )
        print("  ✓ GradioApp instance created")
        
        # Test interface building
        print("  Testing interface build...")
        interface = app.build_interface()
        assert interface is not None, "Interface should not be None"
        print("  ✓ Gradio interface built successfully")
        
        # Verify interface has expected components
        print("  Verifying interface components...")
        # The interface is a gr.Blocks object
        assert hasattr(interface, 'launch'), "Interface should have launch method"
        print("  ✓ Interface has launch method")
        
        # Test resolution parsing
        print("  Testing resolution parsing...")
        width, height = app._parse_resolution("512x512")
        assert width == 512 and height == 512, "Resolution parsing failed"
        width, height = app._parse_resolution("768x768")
        assert width == 768 and height == 768, "Resolution parsing failed"
        print("  ✓ Resolution parsing works correctly")
        
        # Test history management
        print("  Testing history management...")
        initial_count = app.get_generation_count()
        assert initial_count == 0, "Initial count should be 0"
        clear_msg = app.clear_history()
        assert "0" in clear_msg, "Clear message should mention count"
        print("  ✓ History management works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_module_integration():
    """Verify all modules integrate correctly."""
    print("\n" + "="*60)
    print("Verifying Module Integration")
    print("="*60)
    
    try:
        # Test core models
        print("  Testing core models...")
        from src.core.models import (
            GenerationResult, ExperimentConfig, RuntimeMetrics,
            QualityMetrics, ExperimentSummary
        )
        
        # Create and serialize ExperimentConfig
        config = ExperimentConfig(
            name="test_config",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True,
            optimizations={"attention_slicing": True}
        )
        config_dict = config.to_dict()
        config_restored = ExperimentConfig.from_dict(config_dict)
        assert config.name == config_restored.name, "Config serialization failed"
        print("  ✓ ExperimentConfig serialization works")
        
        # Create RuntimeMetrics
        runtime = RuntimeMetrics(latency_ms=100.0, peak_vram_mb=4000.0, throughput=10.0)
        runtime_dict = runtime.to_dict()
        runtime_restored = RuntimeMetrics.from_dict(runtime_dict)
        assert runtime.latency_ms == runtime_restored.latency_ms, "RuntimeMetrics serialization failed"
        print("  ✓ RuntimeMetrics serialization works")
        
        # Create QualityMetrics
        quality = QualityMetrics(clip_score=0.85, fid=15.5, lpips=0.12)
        quality_dict = quality.to_dict()
        quality_restored = QualityMetrics.from_dict(quality_dict)
        assert quality.clip_score == quality_restored.clip_score, "QualityMetrics serialization failed"
        print("  ✓ QualityMetrics serialization works")
        
        # Test ExperimentLogger
        print("  Testing ExperimentLogger...")
        from src.benchmark.logger import ExperimentLogger
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_logger = ExperimentLogger(log_dir=tmpdir, experiment_name="test_exp")
            exp_logger.log_config({"test": "config"})
            exp_logger.log_metrics({"clip_score": 0.85})
            
            # Export CSV
            csv_path = exp_logger.export_csv("test.csv")
            assert Path(csv_path).exists(), "CSV file should exist"
            print("  ✓ ExperimentLogger CSV export works")
            
            # Export JSON
            json_path = exp_logger.export_json("test.json")
            assert Path(json_path).exists(), "JSON file should exist"
            print("  ✓ ExperimentLogger JSON export works")
            
            # Generate summary
            summary = exp_logger.generate_summary()
            assert summary is not None, "Summary should not be None"
            print("  ✓ ExperimentLogger summary generation works")
        
        # Test DatasetBuilder
        print("  Testing DatasetBuilder...")
        from src.dataset.builder import DatasetBuilder
        
        builder = DatasetBuilder()
        num_loaded = builder.load_captions()
        assert num_loaded > 0, "Should load some captions"
        print(f"  ✓ DatasetBuilder loaded {num_loaded} sample captions")
        
        dataset = builder.build_evaluation_dataset(num_samples=10, seed=42)
        assert len(dataset) > 0, "Dataset should have prompts"
        print(f"  ✓ DatasetBuilder built dataset with {len(dataset)} prompts")
        
        stats = builder.analyze_dataset(dataset)
        assert stats.total_prompts > 0, "Stats should have prompts"
        print(f"  ✓ DatasetBuilder analyzed dataset: {stats.total_prompts} prompts")
        
        # Test Visualizer (without actual plotting)
        print("  Testing Visualizer initialization...")
        from src.visualization.visualizer import Visualizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Visualizer(output_dir=tmpdir, style="paper")
            assert viz.output_dir.exists(), "Output dir should exist"
            print("  ✓ Visualizer initialized correctly")
        
        # Test ReportGenerator
        print("  Testing ReportGenerator...")
        from src.report.generator import ReportGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_gen = ReportGenerator(output_dir=tmpdir)
            
            # Create a minimal summary for testing
            test_summary = ExperimentSummary(
                experiment_name="test_experiment",
                total_runs=10,
                configs=[config],
                latency_stats={"test_config": {"mean": 100.0, "std": 10.0, "min": 90.0, "max": 110.0}},
                vram_stats={"test_config": {"mean": 4000.0, "std": 100.0, "min": 3900.0, "max": 4100.0}},
                quality_stats={"test_config": {"mean": 0.85, "std": 0.02, "min": 0.83, "max": 0.87}},
                best_speed_config="test_config",
                best_quality_config="test_config",
                best_tradeoff_config="test_config",
                gpu_info="Test GPU",
                cuda_version="12.0",
                pytorch_version="2.0.0",
            )
            
            # Generate minimal report
            report_path = report_gen.generate_minimal_report(test_summary)
            assert Path(report_path).exists(), "Report file should exist"
            print("  ✓ ReportGenerator minimal report works")
            
            # Generate LaTeX tables
            latex = report_gen.generate_latex_tables(test_summary, table_type="comparison")
            assert "\\begin{table}" in latex, "LaTeX should contain table"
            print("  ✓ ReportGenerator LaTeX tables work")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gradio_app_methods():
    """Verify GradioApp methods work correctly (without actual generation)."""
    print("\n" + "="*60)
    print("Verifying GradioApp Methods")
    print("="*60)
    
    try:
        from src.ui.app import GradioApp
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            app = GradioApp(
                model_dir="models/dreamshaper-7",
                lcm_lora_dir="models/lcm-lora-sdv1-5",
                output_dir=tmpdir,
                device="cuda"
            )
            
            # Test is_pipeline_loaded property
            print("  Testing is_pipeline_loaded property...")
            assert app.is_pipeline_loaded == False, "Pipeline should not be loaded initially"
            print("  ✓ is_pipeline_loaded returns False initially")
            
            # Test export_logs_csv with empty history
            print("  Testing export_logs_csv with empty history...")
            filepath, status = app.export_logs_csv()
            assert filepath is None, "Should return None for empty history"
            assert "没有" in status or "no" in status.lower(), "Status should indicate no records"
            print("  ✓ export_logs_csv handles empty history correctly")
            
            # Test clear_history
            print("  Testing clear_history...")
            msg = app.clear_history()
            assert "0" in msg, "Should indicate 0 records cleared"
            print("  ✓ clear_history works correctly")
            
            # Test _format_metrics (mock a GenerationResult)
            print("  Testing _format_metrics...")
            from src.core.models import GenerationResult
            from datetime import datetime
            from PIL import Image
            
            mock_result = GenerationResult(
                image=Image.new('RGB', (512, 512), color='red'),
                prompt="test prompt",
                seed=42,
                num_steps=4,
                guidance_scale=1.0,
                resolution=(512, 512),
                latency_ms=150.5,
                peak_vram_mb=4000.0,
                scheduler_type="lcm",
                optimizations={"attention_slicing": True},
                timestamp=datetime.now()
            )
            
            metrics_text = app._format_metrics(mock_result)
            assert "150.5" in metrics_text, "Should contain latency"
            assert "4000" in metrics_text, "Should contain VRAM"
            assert "lcm" in metrics_text, "Should contain scheduler type"
            print("  ✓ _format_metrics works correctly")
            
            # Test _generate_comparison_summary with insufficient history
            print("  Testing _generate_comparison_summary...")
            summary = app._generate_comparison_summary(
                (None, "", "error"),
                (None, "", "error"),
                20, 4
            )
            assert "不完整" in summary or "incomplete" in summary.lower() or "⚠️" in summary, \
                "Should indicate incomplete comparison"
            print("  ✓ _generate_comparison_summary handles errors correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_existing_tests():
    """Run existing unit tests to verify they pass."""
    print("\n" + "="*60)
    print("Running Existing Tests")
    print("="*60)
    
    import subprocess
    
    # Run pytest on specific test files (excluding property tests that need GPU)
    test_files = [
        "tests/test_models.py",
        "tests/test_dataset_builder.py",
        "tests/test_report_generator.py",
        "tests/test_visualizer.py",
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"  ⚠ Test file not found: {test_file}")
            continue
        
        print(f"  Running {test_file}...")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"  ✓ {test_file} passed")
            else:
                print(f"  ✗ {test_file} failed")
                print(f"    stdout: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
                print(f"    stderr: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠ {test_file} timed out")
            all_passed = False
        except Exception as e:
            print(f"  ✗ Error running {test_file}: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all checkpoint 13 verifications."""
    print("="*60)
    print("Checkpoint 13: System Integration Verification")
    print("="*60)
    
    results = []
    
    # 1. Verify module imports
    try:
        results.append(("Module Imports", verify_module_imports()))
    except Exception as e:
        print(f"ERROR in module imports: {e}")
        results.append(("Module Imports", False))
    
    # 2. Verify Gradio interface build
    try:
        results.append(("Gradio Interface Build", verify_gradio_interface_build()))
    except Exception as e:
        print(f"ERROR in Gradio interface build: {e}")
        results.append(("Gradio Interface Build", False))
    
    # 3. Verify module integration
    try:
        results.append(("Module Integration", verify_module_integration()))
    except Exception as e:
        print(f"ERROR in module integration: {e}")
        results.append(("Module Integration", False))
    
    # 4. Verify GradioApp methods
    try:
        results.append(("GradioApp Methods", verify_gradio_app_methods()))
    except Exception as e:
        print(f"ERROR in GradioApp methods: {e}")
        results.append(("GradioApp Methods", False))
    
    # 5. Run existing tests
    try:
        results.append(("Existing Tests", run_existing_tests()))
    except Exception as e:
        print(f"ERROR running existing tests: {e}")
        results.append(("Existing Tests", False))
    
    # Summary
    print("\n" + "="*60)
    print("CHECKPOINT 13 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checkpoint 13 verifications passed!")
        print("System integration is complete and ready for next phase.")
    else:
        print("\n✗ Some verifications failed. Please review and fix issues.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
