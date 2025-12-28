#!/usr/bin/env python
"""
完整实验执行脚本 - LCM-LoRA 扩散模型加速系统

执行完整的对比实验、消融实验和参数分析实验，并生成最终报告。

Requirements: 6.1, 6.2, 6.3, 6.5, 7.5, 8.4, 11.6, 13.5, 13.6, 13.7

Usage:
    python run_full_experiments.py [--quick] [--no-quality] [--output-dir OUTPUT_DIR]
    
Options:
    --quick         快速模式，减少重复次数和 prompts 数量
    --no-quality    跳过质量评估 (CLIPScore)
    --output-dir    指定输出目录，默认 outputs/experiments
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def setup_paths():
    """设置项目路径"""
    # 模型路径
    model_dir = Path("models/dreamshaper-7")
    lcm_lora_dir = Path("models/lcm-lora-sdv1-5")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not lcm_lora_dir.exists():
        raise FileNotFoundError(f"LCM-LoRA 目录不存在: {lcm_lora_dir}")
    
    return str(model_dir), str(lcm_lora_dir)


def get_test_prompts(num_prompts: int = 10) -> List[str]:
    """获取测试 prompts"""
    # 使用 DatasetBuilder 获取 prompts
    from src.dataset.builder import DatasetBuilder
    
    builder = DatasetBuilder()
    dataset = builder.build_evaluation_dataset(num_samples=num_prompts, seed=42)
    return dataset.get_prompt_texts()


def run_comparison_experiment(
    pipeline_manager,
    metrics_collector,
    output_dir: Path,
    prompts: List[str],
    seeds: List[int],
    num_repeats: int = 3,
    compute_quality: bool = True,
) -> Any:
    """
    执行完整对比实验
    
    运行 Euler 20步、DPM-Solver 20步、LCM 2/4/6/8步
    
    Requirements: 6.1, 6.2, 6.3, 6.5
    """
    from src.benchmark.runner import BenchmarkRunner
    from src.core.models import ExperimentConfig
    
    logger.info("=" * 60)
    logger.info("开始对比实验")
    logger.info("=" * 60)
    
    # 定义对比实验配置
    comparison_configs = [
        # 基线配置 (Requirements: 6.1)
        ExperimentConfig(
            name="Euler_20",
            scheduler_type="euler",
            num_steps=20,
            guidance_scale=7.5,
            use_lcm_lora=False,
        ),
        # 强对照配置 (Requirements: 6.2)
        ExperimentConfig(
            name="DPM_Solver_20",
            scheduler_type="dpm_solver",
            num_steps=20,
            guidance_scale=7.5,
            use_lcm_lora=False,
        ),
        # LCM 实验配置 (Requirements: 6.3)
        ExperimentConfig(
            name="LCM_2",
            scheduler_type="lcm",
            num_steps=2,
            guidance_scale=1.0,
            use_lcm_lora=True,
        ),
        ExperimentConfig(
            name="LCM_4",
            scheduler_type="lcm",
            num_steps=4,
            guidance_scale=1.0,
            use_lcm_lora=True,
        ),
        ExperimentConfig(
            name="LCM_6",
            scheduler_type="lcm",
            num_steps=6,
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
    
    runner = BenchmarkRunner(
        pipeline_manager=pipeline_manager,
        output_dir=str(output_dir),
        metrics_collector=metrics_collector if compute_quality else None,
    )
    
    logger.info(f"配置数量: {len(comparison_configs)}")
    logger.info(f"Prompts 数量: {len(prompts)}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"重复次数: {num_repeats}")
    
    results = runner.run_comparison_experiment(
        prompts=prompts,
        seeds=seeds,
        configs=comparison_configs,
        num_repeats=num_repeats,
        compute_quality=compute_quality,
    )
    
    logger.info("对比实验完成")
    logger.info(f"实验名称: {results.experiment_name}")
    
    # 打印结果摘要
    logger.info("\n--- 对比实验结果摘要 ---")
    for config_name in results.runtime_stats:
        latency = results.runtime_stats[config_name].get("mean", 0)
        vram = results.vram_stats.get(config_name, {}).get("mean", 0)
        quality = results.quality_stats.get(config_name, {}).get("mean", 0)
        logger.info(f"{config_name}: 延迟={latency:.1f}ms, 显存={vram:.0f}MB, CLIPScore={quality:.4f}")
    
    return results


def run_ablation_experiment(
    pipeline_manager,
    output_dir: Path,
    prompts: List[str],
    seeds: List[int],
    num_repeats: int = 3,
) -> Any:
    """
    执行消融实验
    
    测试各优化项的独立贡献
    
    Requirements: 7.5
    """
    from src.benchmark.runner import BenchmarkRunner
    
    logger.info("=" * 60)
    logger.info("开始消融实验")
    logger.info("=" * 60)
    
    runner = BenchmarkRunner(
        pipeline_manager=pipeline_manager,
        output_dir=str(output_dir),
    )
    
    results = runner.run_ablation_experiment(
        prompts=prompts,
        seeds=seeds,
        num_repeats=num_repeats,
    )
    
    logger.info("消融实验完成")
    logger.info(f"实验名称: {results.experiment_name}")
    
    # 打印贡献分析
    logger.info("\n--- 各优化项贡献分析 ---")
    for opt_name, contribution in results.contributions.items():
        latency_diff = contribution.get("latency_diff_ms", 0)
        vram_diff = contribution.get("vram_diff_mb", 0)
        logger.info(f"{opt_name}: 延迟变化={latency_diff:+.1f}ms, 显存变化={vram_diff:+.0f}MB")
    
    return results


def run_parameter_analysis(
    pipeline_manager,
    metrics_collector,
    output_dir: Path,
    prompts: List[str],
    seeds: List[int],
    num_repeats: int = 3,
) -> Dict[str, Any]:
    """
    执行参数分析实验
    
    分析 guidance_scale、分辨率、batch_size 影响
    
    Requirements: 8.4
    """
    from src.benchmark.runner import BenchmarkRunner
    
    logger.info("=" * 60)
    logger.info("开始参数分析实验")
    logger.info("=" * 60)
    
    runner = BenchmarkRunner(
        pipeline_manager=pipeline_manager,
        output_dir=str(output_dir),
        metrics_collector=metrics_collector,
    )
    
    # 定义参数范围 (Requirements: 8.1, 8.2, 8.3)
    parameter_ranges = {
        "guidance_scale": [0.0, 1.0, 1.5, 2.0],
        "resolution": [(512, 512), (768, 768)],
    }
    
    results = runner.run_parameter_analysis(
        prompts=prompts,
        parameter_ranges=parameter_ranges,
        seeds=seeds,
        num_repeats=num_repeats,
    )
    
    logger.info("参数分析实验完成")
    
    # 打印结果摘要
    for param_name, param_result in results.items():
        logger.info(f"\n--- {param_name} 分析结果 ---")
        for value_key in param_result.runtime_stats:
            latency = param_result.runtime_stats[value_key].get("mean", 0)
            logger.info(f"  {param_name}={value_key}: 延迟={latency:.1f}ms")
    
    return results


def generate_visualizations(
    comparison_results: Any,
    ablation_results: Any,
    parameter_results: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    """
    生成所有图表
    
    Requirements: 11.6
    """
    from src.visualization.visualizer import Visualizer
    
    logger.info("=" * 60)
    logger.info("生成可视化图表")
    logger.info("=" * 60)
    
    charts_dir = output_dir / "charts"
    visualizer = Visualizer(output_dir=str(charts_dir), style="paper")
    
    all_charts = []
    
    # 1. 对比实验柱状图
    if comparison_results:
        logger.info("生成对比实验柱状图...")
        charts = visualizer.plot_comparison_bars(
            results=comparison_results,
            metrics=['latency', 'vram', 'clip_score'],
            title="LCM-LoRA 加速效果对比",
            filename="comparison_bars",
        )
        all_charts.extend(charts)
        
        # 步数-质量-速度曲线
        logger.info("生成步数曲线图...")
        charts = visualizer.plot_steps_curve(
            results=comparison_results,
            title="步数-质量-速度关系",
            filename="steps_curve",
        )
        all_charts.extend(charts)
    
    # 2. 消融实验表格
    if ablation_results:
        logger.info("生成消融实验表格...")
        charts = visualizer.plot_ablation_table(
            results=ablation_results,
            title="消融实验结果",
            filename="ablation_table",
        )
        all_charts.extend(charts)
    
    # 3. 参数敏感性图
    if parameter_results:
        logger.info("生成参数敏感性图...")
        charts = visualizer.plot_parameter_sensitivity(
            results=parameter_results,
            title="参数敏感性分析",
            filename="parameter_sensitivity",
        )
        all_charts.extend(charts)
    
    logger.info(f"共生成 {len(all_charts)} 个图表文件")
    return all_charts


def generate_final_report(
    comparison_results: Any,
    ablation_results: Any,
    parameter_results: Dict[str, Any],
    charts: List[str],
    output_dir: Path,
) -> str:
    """
    生成最终实验报告
    
    Requirements: 13.5, 13.6, 13.7
    """
    from src.report.generator import ReportGenerator
    from src.benchmark.logger import ExperimentLogger
    from src.core.models import ExperimentSummary
    
    logger.info("=" * 60)
    logger.info("生成最终实验报告")
    logger.info("=" * 60)
    
    reports_dir = output_dir / "reports"
    report_generator = ReportGenerator(output_dir=str(reports_dir))
    
    # 从对比实验结果构建 ExperimentSummary
    if comparison_results:
        # 获取环境信息
        import torch
        gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_version = torch.version.cuda or "N/A"
        pytorch_version = torch.__version__
        
        # 确定最优配置
        best_speed_config = ""
        best_quality_config = ""
        best_tradeoff_config = ""
        
        if comparison_results.runtime_stats:
            best_speed_config = min(
                comparison_results.runtime_stats.keys(),
                key=lambda k: comparison_results.runtime_stats[k].get("mean", float("inf"))
            )
        
        if comparison_results.quality_stats:
            best_quality_config = max(
                comparison_results.quality_stats.keys(),
                key=lambda k: comparison_results.quality_stats[k].get("mean", 0)
            )
            
            # 最佳权衡: 质量/延迟比
            tradeoff_scores = {}
            for config_name in comparison_results.runtime_stats:
                if config_name in comparison_results.quality_stats:
                    latency = comparison_results.runtime_stats[config_name].get("mean", float("inf"))
                    quality = comparison_results.quality_stats[config_name].get("mean", 0)
                    if latency > 0:
                        tradeoff_scores[config_name] = quality / latency * 1000
            if tradeoff_scores:
                best_tradeoff_config = max(tradeoff_scores.keys(), key=lambda k: tradeoff_scores[k])
        
        summary = ExperimentSummary(
            experiment_name=comparison_results.experiment_name,
            total_runs=sum(len(v) for v in comparison_results.results.values()),
            configs=comparison_results.configs,
            latency_stats=comparison_results.runtime_stats,
            vram_stats=comparison_results.vram_stats,
            quality_stats=comparison_results.quality_stats,
            best_speed_config=best_speed_config,
            best_quality_config=best_quality_config,
            best_tradeoff_config=best_tradeoff_config,
            gpu_info=gpu_info,
            cuda_version=cuda_version,
            pytorch_version=pytorch_version,
        )
        
        # 生成完整报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_generator.generate_experiment_report(
            experiment_summary=summary,
            charts=charts,
            sample_images=[],
            output_filename=f"full_experiment_report_{timestamp}.md",
        )
        
        # 生成 LaTeX 表格
        latex_path = report_generator.export_latex_tables_to_file(
            experiment_summary=summary,
            output_filename=f"experiment_tables_{timestamp}.tex",
        )
        
        logger.info(f"报告已生成: {report_path}")
        logger.info(f"LaTeX 表格已生成: {latex_path}")
        
        return report_path
    
    return ""


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LCM-LoRA 完整实验执行脚本")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--no-quality", action="store_true", help="跳过质量评估")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments", help="输出目录")
    args = parser.parse_args()
    
    # 设置参数
    if args.quick:
        num_prompts = 3
        num_repeats = 1
        seeds = [42]
    else:
        num_prompts = 10
        num_repeats = 3
        seeds = [42, 123, 456]
    
    compute_quality = not args.no_quality
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置文件日志
    file_handler = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 60)
    logger.info("LCM-LoRA 完整实验执行")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"快速模式: {args.quick}")
    logger.info(f"质量评估: {compute_quality}")
    logger.info(f"Prompts 数量: {num_prompts}")
    logger.info(f"重复次数: {num_repeats}")
    logger.info(f"Seeds: {seeds}")
    
    try:
        # 设置路径
        model_dir, lcm_lora_dir = setup_paths()
        logger.info(f"模型目录: {model_dir}")
        logger.info(f"LCM-LoRA 目录: {lcm_lora_dir}")
        
        # 初始化管线管理器
        from src.core.pipeline import PipelineManager
        pipeline_manager = PipelineManager(
            model_dir=model_dir,
            lcm_lora_dir=lcm_lora_dir,
            device="cuda",
        )
        
        # 初始化指标收集器
        metrics_collector = None
        if compute_quality:
            from src.metrics.collector import MetricsCollector
            metrics_collector = MetricsCollector(device="cuda")
        
        # 获取测试 prompts
        prompts = get_test_prompts(num_prompts)
        logger.info(f"加载 {len(prompts)} 条测试 prompts")
        
        # 1. 执行对比实验 (Requirements: 6.1, 6.2, 6.3, 6.5)
        comparison_results = run_comparison_experiment(
            pipeline_manager=pipeline_manager,
            metrics_collector=metrics_collector,
            output_dir=output_dir,
            prompts=prompts,
            seeds=seeds,
            num_repeats=num_repeats,
            compute_quality=compute_quality,
        )
        
        # 2. 执行消融实验 (Requirements: 7.5)
        ablation_results = run_ablation_experiment(
            pipeline_manager=pipeline_manager,
            output_dir=output_dir,
            prompts=prompts[:3],  # 消融实验使用较少 prompts
            seeds=[42],
            num_repeats=num_repeats,
        )
        
        # 3. 执行参数分析实验 (Requirements: 8.4)
        parameter_results = run_parameter_analysis(
            pipeline_manager=pipeline_manager,
            metrics_collector=metrics_collector,
            output_dir=output_dir,
            prompts=prompts[:3],  # 参数分析使用较少 prompts
            seeds=[42],
            num_repeats=num_repeats,
        )
        
        # 4. 生成可视化图表 (Requirements: 11.6)
        charts = generate_visualizations(
            comparison_results=comparison_results,
            ablation_results=ablation_results,
            parameter_results=parameter_results,
            output_dir=output_dir,
        )
        
        # 5. 生成最终报告 (Requirements: 13.5, 13.6, 13.7)
        report_path = generate_final_report(
            comparison_results=comparison_results,
            ablation_results=ablation_results,
            parameter_results=parameter_results,
            charts=charts,
            output_dir=output_dir,
        )
        
        # 清理资源
        pipeline_manager.unload()
        if metrics_collector:
            metrics_collector.unload_models()
        
        logger.info("=" * 60)
        logger.info("实验完成!")
        logger.info("=" * 60)
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"实验报告: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"实验执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
