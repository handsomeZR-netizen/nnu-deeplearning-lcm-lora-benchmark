"""
BenchmarkRunner - Systematic benchmark testing for LCM-LoRA acceleration system.

Executes comparison experiments, ablation studies, and parameter analysis.
Collects metrics and generates statistical summaries.

Requirements: 4.4, 6.1, 6.2, 6.3, 6.4, 7.1-7.5, 8.1-8.4
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..core.models import (
    ExperimentConfig,
    ExperimentSummary,
    GenerationResult,
    QualityMetrics,
    RuntimeMetrics,
)
from ..core.pipeline import PipelineManager
from .logger import ExperimentLogger

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """对比实验结果"""
    experiment_name: str
    configs: List[ExperimentConfig]
    results: Dict[str, List[GenerationResult]]  # config_name -> results
    quality_metrics: Dict[str, List[QualityMetrics]]  # config_name -> metrics
    runtime_stats: Dict[str, Dict[str, float]]  # config_name -> {mean, std, min, max}
    vram_stats: Dict[str, Dict[str, float]]
    quality_stats: Dict[str, Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "experiment_name": self.experiment_name,
            "configs": [c.to_dict() for c in self.configs],
            "runtime_stats": self.runtime_stats,
            "vram_stats": self.vram_stats,
            "quality_stats": self.quality_stats,
            "timestamp": self.timestamp.isoformat(),
            "num_results": {k: len(v) for k, v in self.results.items()},
        }


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    use_lcm_lora: bool = True
    use_xformers: bool = False
    use_sdpa: bool = True
    use_attention_slicing: bool = True
    use_vae_slicing: bool = True
    use_vae_tiling: bool = False
    num_steps: int = 4
    guidance_scale: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "use_lcm_lora": self.use_lcm_lora,
            "use_xformers": self.use_xformers,
            "use_sdpa": self.use_sdpa,
            "use_attention_slicing": self.use_attention_slicing,
            "use_vae_slicing": self.use_vae_slicing,
            "use_vae_tiling": self.use_vae_tiling,
            "num_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
        }


@dataclass
class AblationResults:
    """消融实验结果"""
    experiment_name: str
    configs: List[AblationConfig]
    results: Dict[str, List[GenerationResult]]  # config_name -> results
    runtime_stats: Dict[str, Dict[str, float]]
    vram_stats: Dict[str, Dict[str, float]]
    contributions: Dict[str, Dict[str, float]]  # optimization -> {latency_diff, vram_diff}
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "experiment_name": self.experiment_name,
            "configs": [c.to_dict() for c in self.configs],
            "runtime_stats": self.runtime_stats,
            "vram_stats": self.vram_stats,
            "contributions": self.contributions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ParameterResults:
    """参数分析结果"""
    experiment_name: str
    parameter_name: str
    parameter_values: List[Any]
    results: Dict[str, List[GenerationResult]]  # param_value -> results
    runtime_stats: Dict[str, Dict[str, float]]
    vram_stats: Dict[str, Dict[str, float]]
    quality_stats: Dict[str, Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "experiment_name": self.experiment_name,
            "parameter_name": self.parameter_name,
            "parameter_values": [str(v) for v in self.parameter_values],
            "runtime_stats": self.runtime_stats,
            "vram_stats": self.vram_stats,
            "quality_stats": self.quality_stats,
            "timestamp": self.timestamp.isoformat(),
        }


class BenchmarkRunner:
    """
    执行系统化的基准测试实验
    
    Supports:
    - Comparison experiments (Euler/DPM-Solver/LCM)
    - Ablation studies (toggle optimizations)
    - Parameter sensitivity analysis
    
    Requirements: 4.4, 6.1-6.5, 7.1-7.5, 8.1-8.4
    """
    
    # Default experiment configurations
    DEFAULT_COMPARISON_CONFIGS = [
        ExperimentConfig(
            name="Euler_20",
            scheduler_type="euler",
            num_steps=20,
            guidance_scale=7.5,
            use_lcm_lora=False,
        ),
        ExperimentConfig(
            name="Euler_30",
            scheduler_type="euler",
            num_steps=30,
            guidance_scale=7.5,
            use_lcm_lora=False,
        ),
        ExperimentConfig(
            name="DPM_Solver_20",
            scheduler_type="dpm_solver",
            num_steps=20,
            guidance_scale=7.5,
            use_lcm_lora=False,
        ),
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
    
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        output_dir: str,
        metrics_collector: Optional[Any] = None,
    ):
        """
        初始化基准测试运行器
        
        Args:
            pipeline_manager: 管线管理器实例
            output_dir: 输出目录
            metrics_collector: 可选的指标收集器 (MetricsCollector)
        """
        self.pipeline_manager = pipeline_manager
        self.output_dir = Path(output_dir)
        self.metrics_collector = metrics_collector
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BenchmarkRunner initialized: output_dir={self.output_dir}")

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """计算统计信息: mean, std, min, max"""
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        
        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }
    
    def _load_pipeline_for_config(self, config: ExperimentConfig) -> None:
        """
        根据配置加载对应的管线
        
        Args:
            config: 实验配置
        """
        # 卸载现有管线
        self.pipeline_manager.unload()
        
        if config.use_lcm_lora:
            # 加载 LCM 管线
            self.pipeline_manager.load_lcm_pipeline(fuse_lora=True)
        else:
            # 加载基线管线
            self.pipeline_manager.load_baseline_pipeline(
                scheduler_type=config.scheduler_type
            )
        
        # 应用优化
        optimizations = config.optimizations or {}
        self.pipeline_manager.apply_optimizations(
            attention_slicing=optimizations.get("attention_slicing", True),
            vae_slicing=optimizations.get("vae_slicing", True),
            vae_tiling=optimizations.get("vae_tiling", False),
            xformers=optimizations.get("xformers", False),
            sdpa=optimizations.get("sdpa", True),
        )
        
        # 预热
        self.pipeline_manager.warmup(num_steps=min(config.num_steps, 2))
    
    def run_comparison_experiment(
        self,
        prompts: List[str],
        seeds: List[int],
        configs: Optional[List[ExperimentConfig]] = None,
        num_repeats: int = 3,
        width: int = 512,
        height: int = 512,
        compute_quality: bool = True,
    ) -> ExperimentResults:
        """
        运行对比实验
        
        Args:
            prompts: 测试 prompts 列表
            seeds: 随机种子列表
            configs: 实验配置列表，默认使用 DEFAULT_COMPARISON_CONFIGS
            num_repeats: 每个配置重复运行次数 (Requirements: 4.4)
            width: 图像宽度
            height: 图像高度
            compute_quality: 是否计算质量指标
        
        Returns:
            ExperimentResults 包含所有实验结果
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 4.4
        """
        if configs is None:
            configs = self.DEFAULT_COMPARISON_CONFIGS
        
        experiment_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_logger = ExperimentLogger(
            log_dir=str(self.output_dir / "logs"),
            experiment_name=experiment_name,
        )
        
        logger.info(
            f"Starting comparison experiment: {len(configs)} configs, "
            f"{len(prompts)} prompts, {len(seeds)} seeds, {num_repeats} repeats"
        )
        
        # 存储结果
        all_results: Dict[str, List[GenerationResult]] = {}
        all_quality: Dict[str, List[QualityMetrics]] = {}
        
        for config in configs:
            config_name = config.name
            all_results[config_name] = []
            all_quality[config_name] = []
            
            logger.info(f"Running config: {config_name}")
            exp_logger.log_config(config.to_dict())
            
            # 加载对应管线
            self._load_pipeline_for_config(config)
            
            # 运行实验
            for prompt in prompts:
                for seed in seeds:
                    for repeat in range(num_repeats):
                        try:
                            result = self.pipeline_manager.generate(
                                prompt=prompt,
                                num_steps=config.num_steps,
                                guidance_scale=config.guidance_scale,
                                seed=seed,
                                width=width,
                                height=height,
                            )
                            
                            all_results[config_name].append(result)
                            exp_logger.log_result(result)
                            
                            # 计算质量指标
                            if compute_quality and self.metrics_collector is not None:
                                clip_score = self.metrics_collector.compute_clip_score(
                                    result.image, prompt
                                )
                                quality = QualityMetrics(clip_score=clip_score)
                                all_quality[config_name].append(quality)
                                exp_logger.log_metrics({
                                    "config_name": config_name,
                                    "clip_score": clip_score,
                                    "prompt": prompt,
                                    "seed": seed,
                                })
                            
                            logger.debug(
                                f"  {config_name}: prompt={prompt[:30]}..., "
                                f"seed={seed}, repeat={repeat+1}, "
                                f"latency={result.latency_ms:.1f}ms"
                            )
                        
                        except Exception as e:
                            logger.error(f"Error in {config_name}: {e}")
                            continue
        
        # 计算统计信息
        runtime_stats = {}
        vram_stats = {}
        quality_stats = {}
        
        for config_name, results in all_results.items():
            latencies = [r.latency_ms for r in results]
            vrams = [r.peak_vram_mb for r in results]
            
            runtime_stats[config_name] = self._compute_stats(latencies)
            vram_stats[config_name] = self._compute_stats(vrams)
            
            if config_name in all_quality and all_quality[config_name]:
                scores = [q.clip_score for q in all_quality[config_name]]
                quality_stats[config_name] = self._compute_stats(scores)
        
        # 导出日志
        exp_logger.export_csv(f"{experiment_name}_results.csv")
        exp_logger.export_json(f"{experiment_name}_full.json")
        
        logger.info(f"Comparison experiment completed: {experiment_name}")
        
        return ExperimentResults(
            experiment_name=experiment_name,
            configs=configs,
            results=all_results,
            quality_metrics=all_quality,
            runtime_stats=runtime_stats,
            vram_stats=vram_stats,
            quality_stats=quality_stats,
        )

    def run_ablation_experiment(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
        ablation_configs: Optional[List[AblationConfig]] = None,
        num_repeats: int = 3,
        width: int = 512,
        height: int = 512,
    ) -> AblationResults:
        """
        运行消融实验
        
        测试各优化项的独立贡献
        
        Args:
            prompts: 测试 prompts 列表
            seeds: 随机种子列表，默认 [42]
            ablation_configs: 消融配置列表，默认生成标准消融配置
            num_repeats: 每个配置重复运行次数
            width: 图像宽度
            height: 图像高度
        
        Returns:
            AblationResults 包含消融实验结果
        
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        if seeds is None:
            seeds = [42]
        
        if ablation_configs is None:
            ablation_configs = self._generate_default_ablation_configs()
        
        experiment_name = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_logger = ExperimentLogger(
            log_dir=str(self.output_dir / "logs"),
            experiment_name=experiment_name,
        )
        
        logger.info(
            f"Starting ablation experiment: {len(ablation_configs)} configs, "
            f"{len(prompts)} prompts, {num_repeats} repeats"
        )
        
        # 存储结果
        all_results: Dict[str, List[GenerationResult]] = {}
        
        for config in ablation_configs:
            config_name = config.name
            all_results[config_name] = []
            
            logger.info(f"Running ablation config: {config_name}")
            exp_logger.log_config(config.to_dict())
            
            # 卸载现有管线
            self.pipeline_manager.unload()
            
            # 根据配置加载管线
            if config.use_lcm_lora:
                self.pipeline_manager.load_lcm_pipeline(fuse_lora=True)
            else:
                self.pipeline_manager.load_baseline_pipeline(scheduler_type="euler")
            
            # 应用优化配置
            self.pipeline_manager.apply_optimizations(
                attention_slicing=config.use_attention_slicing,
                vae_slicing=config.use_vae_slicing,
                vae_tiling=config.use_vae_tiling,
                xformers=config.use_xformers,
                sdpa=config.use_sdpa,
            )
            
            # 预热
            self.pipeline_manager.warmup(num_steps=min(config.num_steps, 2))
            
            # 运行实验
            for prompt in prompts:
                for seed in seeds:
                    for repeat in range(num_repeats):
                        try:
                            result = self.pipeline_manager.generate(
                                prompt=prompt,
                                num_steps=config.num_steps,
                                guidance_scale=config.guidance_scale,
                                seed=seed,
                                width=width,
                                height=height,
                            )
                            
                            all_results[config_name].append(result)
                            exp_logger.log_result(result)
                            
                            logger.debug(
                                f"  {config_name}: latency={result.latency_ms:.1f}ms, "
                                f"vram={result.peak_vram_mb:.0f}MB"
                            )
                        
                        except Exception as e:
                            logger.error(f"Error in {config_name}: {e}")
                            continue
        
        # 计算统计信息
        runtime_stats = {}
        vram_stats = {}
        
        for config_name, results in all_results.items():
            latencies = [r.latency_ms for r in results]
            vrams = [r.peak_vram_mb for r in results]
            
            runtime_stats[config_name] = self._compute_stats(latencies)
            vram_stats[config_name] = self._compute_stats(vrams)
        
        # 计算各优化项的贡献
        contributions = self._compute_ablation_contributions(
            ablation_configs, runtime_stats, vram_stats
        )
        
        # 导出日志
        exp_logger.export_csv(f"{experiment_name}_results.csv")
        exp_logger.export_json(f"{experiment_name}_full.json")
        
        logger.info(f"Ablation experiment completed: {experiment_name}")
        
        return AblationResults(
            experiment_name=experiment_name,
            configs=ablation_configs,
            results=all_results,
            runtime_stats=runtime_stats,
            vram_stats=vram_stats,
            contributions=contributions,
        )
    
    def _generate_default_ablation_configs(self) -> List[AblationConfig]:
        """
        生成默认消融实验配置
        
        包含:
        - 完整优化 (baseline)
        - 关闭 LCM-LoRA
        - 关闭 xFormers/SDPA
        - 关闭 attention_slicing
        - 关闭 VAE_slicing/tiling
        
        Requirements: 7.1, 7.2, 7.3, 7.4
        """
        return [
            # 完整优化配置 (baseline)
            AblationConfig(
                name="full_optimization",
                use_lcm_lora=True,
                use_sdpa=True,
                use_attention_slicing=True,
                use_vae_slicing=True,
                num_steps=4,
                guidance_scale=1.0,
            ),
            # 关闭 LCM-LoRA (Requirements: 7.1)
            AblationConfig(
                name="no_lcm_lora",
                use_lcm_lora=False,
                use_sdpa=True,
                use_attention_slicing=True,
                use_vae_slicing=True,
                num_steps=20,  # 基线需要更多步数
                guidance_scale=7.5,
            ),
            # 关闭 SDPA (Requirements: 7.2)
            AblationConfig(
                name="no_sdpa",
                use_lcm_lora=True,
                use_sdpa=False,
                use_attention_slicing=True,
                use_vae_slicing=True,
                num_steps=4,
                guidance_scale=1.0,
            ),
            # 关闭 attention_slicing (Requirements: 7.3)
            AblationConfig(
                name="no_attention_slicing",
                use_lcm_lora=True,
                use_sdpa=True,
                use_attention_slicing=False,
                use_vae_slicing=True,
                num_steps=4,
                guidance_scale=1.0,
            ),
            # 关闭 VAE_slicing (Requirements: 7.4)
            AblationConfig(
                name="no_vae_slicing",
                use_lcm_lora=True,
                use_sdpa=True,
                use_attention_slicing=True,
                use_vae_slicing=False,
                num_steps=4,
                guidance_scale=1.0,
            ),
            # 关闭所有优化
            AblationConfig(
                name="no_optimization",
                use_lcm_lora=True,
                use_sdpa=False,
                use_attention_slicing=False,
                use_vae_slicing=False,
                num_steps=4,
                guidance_scale=1.0,
            ),
        ]
    
    def _compute_ablation_contributions(
        self,
        configs: List[AblationConfig],
        runtime_stats: Dict[str, Dict[str, float]],
        vram_stats: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        计算各优化项的独立贡献
        
        通过对比完整优化与关闭单项优化的差异来计算
        
        Requirements: 7.5
        """
        contributions = {}
        
        # 获取基线 (完整优化) 的统计
        baseline_name = "full_optimization"
        if baseline_name not in runtime_stats:
            return contributions
        
        baseline_latency = runtime_stats[baseline_name].get("mean", 0)
        baseline_vram = vram_stats[baseline_name].get("mean", 0)
        
        # 计算各优化项的贡献
        optimization_map = {
            "lcm_lora": "no_lcm_lora",
            "sdpa": "no_sdpa",
            "attention_slicing": "no_attention_slicing",
            "vae_slicing": "no_vae_slicing",
        }
        
        for opt_name, config_name in optimization_map.items():
            if config_name in runtime_stats:
                opt_latency = runtime_stats[config_name].get("mean", 0)
                opt_vram = vram_stats[config_name].get("mean", 0)
                
                contributions[opt_name] = {
                    "latency_diff_ms": opt_latency - baseline_latency,
                    "latency_diff_pct": (
                        (opt_latency - baseline_latency) / baseline_latency * 100
                        if baseline_latency > 0 else 0
                    ),
                    "vram_diff_mb": opt_vram - baseline_vram,
                    "vram_diff_pct": (
                        (opt_vram - baseline_vram) / baseline_vram * 100
                        if baseline_vram > 0 else 0
                    ),
                }
        
        return contributions

    def run_parameter_analysis(
        self,
        prompts: List[str],
        parameter_ranges: Optional[Dict[str, List[Any]]] = None,
        seeds: Optional[List[int]] = None,
        num_repeats: int = 3,
        base_config: Optional[ExperimentConfig] = None,
    ) -> Dict[str, ParameterResults]:
        """
        运行参数敏感性分析
        
        Args:
            prompts: 测试 prompts 列表
            parameter_ranges: 参数范围字典，默认包含 guidance_scale, resolution, batch_size
            seeds: 随机种子列表，默认 [42]
            num_repeats: 每个配置重复运行次数
            base_config: 基础配置，默认使用 LCM_4
        
        Returns:
            Dict[str, ParameterResults] 各参数的分析结果
        
        Requirements: 8.1, 8.2, 8.3, 8.4
        """
        if seeds is None:
            seeds = [42]
        
        if parameter_ranges is None:
            parameter_ranges = {
                "guidance_scale": [0.0, 1.0, 1.5, 2.0],  # Requirements: 8.1
                "resolution": [(512, 512), (768, 768)],  # Requirements: 8.2
                "batch_size": [1],  # Requirements: 8.3 (batch_size 通过多次生成模拟)
            }
        
        if base_config is None:
            base_config = ExperimentConfig(
                name="LCM_4_base",
                scheduler_type="lcm",
                num_steps=4,
                guidance_scale=1.0,
                use_lcm_lora=True,
            )
        
        all_parameter_results = {}
        
        for param_name, param_values in parameter_ranges.items():
            logger.info(f"Analyzing parameter: {param_name} with values {param_values}")
            
            result = self._run_single_parameter_analysis(
                prompts=prompts,
                seeds=seeds,
                parameter_name=param_name,
                parameter_values=param_values,
                num_repeats=num_repeats,
                base_config=base_config,
            )
            
            all_parameter_results[param_name] = result
        
        return all_parameter_results
    
    def _run_single_parameter_analysis(
        self,
        prompts: List[str],
        seeds: List[int],
        parameter_name: str,
        parameter_values: List[Any],
        num_repeats: int,
        base_config: ExperimentConfig,
    ) -> ParameterResults:
        """
        运行单个参数的敏感性分析
        
        Args:
            prompts: 测试 prompts 列表
            seeds: 随机种子列表
            parameter_name: 参数名称
            parameter_values: 参数值列表
            num_repeats: 重复次数
            base_config: 基础配置
        
        Returns:
            ParameterResults 参数分析结果
        """
        experiment_name = f"param_{parameter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_logger = ExperimentLogger(
            log_dir=str(self.output_dir / "logs"),
            experiment_name=experiment_name,
        )
        
        # 存储结果
        all_results: Dict[str, List[GenerationResult]] = {}
        all_quality: Dict[str, List[QualityMetrics]] = {}
        
        for param_value in parameter_values:
            value_key = str(param_value)
            all_results[value_key] = []
            all_quality[value_key] = []
            
            logger.info(f"Testing {parameter_name}={param_value}")
            
            # 确定生成参数
            width, height = 512, 512
            guidance_scale = base_config.guidance_scale
            
            if parameter_name == "resolution":
                width, height = param_value
            elif parameter_name == "guidance_scale":
                guidance_scale = param_value
            
            # 加载管线 (每个参数值重新加载以确保一致性)
            self._load_pipeline_for_config(base_config)
            
            # 运行实验
            for prompt in prompts:
                for seed in seeds:
                    for repeat in range(num_repeats):
                        try:
                            result = self.pipeline_manager.generate(
                                prompt=prompt,
                                num_steps=base_config.num_steps,
                                guidance_scale=guidance_scale,
                                seed=seed,
                                width=width,
                                height=height,
                            )
                            
                            all_results[value_key].append(result)
                            exp_logger.log_result(result)
                            
                            # 计算质量指标
                            if self.metrics_collector is not None:
                                clip_score = self.metrics_collector.compute_clip_score(
                                    result.image, prompt
                                )
                                quality = QualityMetrics(clip_score=clip_score)
                                all_quality[value_key].append(quality)
                                exp_logger.log_metrics({
                                    "parameter_name": parameter_name,
                                    "parameter_value": str(param_value),
                                    "clip_score": clip_score,
                                })
                            
                            logger.debug(
                                f"  {parameter_name}={param_value}: "
                                f"latency={result.latency_ms:.1f}ms"
                            )
                        
                        except Exception as e:
                            logger.error(f"Error with {parameter_name}={param_value}: {e}")
                            continue
        
        # 计算统计信息
        runtime_stats = {}
        vram_stats = {}
        quality_stats = {}
        
        for value_key, results in all_results.items():
            latencies = [r.latency_ms for r in results]
            vrams = [r.peak_vram_mb for r in results]
            
            runtime_stats[value_key] = self._compute_stats(latencies)
            vram_stats[value_key] = self._compute_stats(vrams)
            
            if value_key in all_quality and all_quality[value_key]:
                scores = [q.clip_score for q in all_quality[value_key]]
                quality_stats[value_key] = self._compute_stats(scores)
        
        # 导出日志
        exp_logger.export_csv(f"{experiment_name}_results.csv")
        exp_logger.export_json(f"{experiment_name}_full.json")
        
        return ParameterResults(
            experiment_name=experiment_name,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            results=all_results,
            runtime_stats=runtime_stats,
            vram_stats=vram_stats,
            quality_stats=quality_stats,
        )
    
    def get_steps_quality_speed_data(
        self,
        experiment_results: ExperimentResults,
    ) -> Dict[str, Dict[str, float]]:
        """
        从实验结果中提取步数-质量-速度曲线数据
        
        Args:
            experiment_results: 对比实验结果
        
        Returns:
            Dict 包含每个配置的 steps, latency_mean, quality_mean
        
        Requirements: 6.5
        """
        data = {}
        
        for config in experiment_results.configs:
            config_name = config.name
            
            if config_name in experiment_results.runtime_stats:
                latency = experiment_results.runtime_stats[config_name].get("mean", 0)
                quality = experiment_results.quality_stats.get(config_name, {}).get("mean", 0)
                
                data[config_name] = {
                    "steps": config.num_steps,
                    "latency_ms": latency,
                    "quality": quality,
                    "scheduler": config.scheduler_type,
                    "use_lcm": config.use_lcm_lora,
                }
        
        return data
