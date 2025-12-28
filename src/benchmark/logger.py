"""
ExperimentLogger - Experiment logging and result management.

Records experiment configurations, generation results, and metrics.
Supports export to CSV and JSON formats.

Requirements: 13.1, 13.2, 13.3, 13.4, 4.5
"""

import csv
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics

from ..core.models import (
    GenerationResult,
    ExperimentConfig,
    ExperimentSummary,
    QualityMetrics,
    RuntimeMetrics,
)

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    记录实验过程和结果
    
    Supports:
    - Logging experiment configurations
    - Recording generation results with full metrics
    - Exporting to CSV and JSON formats
    - Generating experiment summaries with statistics
    
    Requirements: 13.1, 13.2, 13.3, 13.4
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化实验日志器
        
        Args:
            log_dir: 日志输出目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.experiment_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        
        # 确保目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储数据
        self._configs: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []
        self._metrics: List[Dict[str, Any]] = []
        
        logger.info(
            f"实验日志器初始化: name={experiment_name}, "
            f"id={self.experiment_id}, dir={self.log_dir}"
        )
    
    @property
    def experiment_timestamp(self) -> str:
        """获取实验开始时间戳"""
        return self.start_time.isoformat()
    
    @property
    def results_count(self) -> int:
        """获取已记录的结果数量"""
        return len(self._results)
    
    @property
    def configs_count(self) -> int:
        """获取已记录的配置数量"""
        return len(self._configs)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        记录实验配置
        
        Args:
            config: 实验配置字典，可以是 ExperimentConfig.to_dict() 的结果
                   或任意配置字典
        
        Requirements: 13.1
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "config": config,
        }
        self._configs.append(entry)
        logger.debug(f"记录配置: {config.get('name', 'unnamed')}")
    
    def log_result(self, result: GenerationResult) -> None:
        """
        记录单次生成结果
        
        Args:
            result: GenerationResult 实例
        
        Requirements: 13.2
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "prompt": result.prompt,
            "seed": result.seed,
            "num_steps": result.num_steps,
            "guidance_scale": result.guidance_scale,
            "resolution": list(result.resolution),
            "latency_ms": result.latency_ms,
            "peak_vram_mb": result.peak_vram_mb,
            "scheduler_type": result.scheduler_type,
            "optimizations": result.optimizations,
            "generation_timestamp": result.timestamp.isoformat(),
        }
        self._results.append(entry)
        logger.debug(
            f"记录结果: prompt={result.prompt[:30]}..., "
            f"latency={result.latency_ms:.1f}ms"
        )
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        记录评估指标
        
        Args:
            metrics: 指标字典，可包含 clip_score, fid, lpips 等
        
        Requirements: 13.2
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "metrics": metrics,
        }
        self._metrics.append(entry)
        logger.debug(f"记录指标: {list(metrics.keys())}")
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """获取所有记录的结果"""
        return self._results.copy()
    
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """获取所有记录的配置"""
        return self._configs.copy()
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """获取所有记录的指标"""
        return self._metrics.copy()

    def export_csv(self, filename: str) -> Path:
        """
        导出 CSV 格式结果
        
        Args:
            filename: 输出文件名 (不含路径)
        
        Returns:
            导出文件的完整路径
        
        Requirements: 4.5
        """
        filepath = self.log_dir / filename
        
        if not self._results:
            logger.warning("没有结果可导出")
            # 创建空文件
            filepath.touch()
            return filepath
        
        # 定义 CSV 列
        fieldnames = [
            "timestamp",
            "experiment_id",
            "experiment_name",
            "prompt",
            "seed",
            "num_steps",
            "guidance_scale",
            "resolution_w",
            "resolution_h",
            "latency_ms",
            "peak_vram_mb",
            "scheduler_type",
            "optimizations",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self._results:
                row = {
                    "timestamp": result["timestamp"],
                    "experiment_id": result["experiment_id"],
                    "experiment_name": result["experiment_name"],
                    "prompt": result["prompt"],
                    "seed": result["seed"],
                    "num_steps": result["num_steps"],
                    "guidance_scale": result["guidance_scale"],
                    "resolution_w": result["resolution"][0],
                    "resolution_h": result["resolution"][1],
                    "latency_ms": result["latency_ms"],
                    "peak_vram_mb": result["peak_vram_mb"],
                    "scheduler_type": result["scheduler_type"],
                    "optimizations": json.dumps(result["optimizations"]),
                }
                writer.writerow(row)
        
        logger.info(f"CSV 导出完成: {filepath} ({len(self._results)} 条记录)")
        return filepath
    
    def export_json(self, filename: str) -> Path:
        """
        导出 JSON 格式完整日志
        
        Args:
            filename: 输出文件名 (不含路径)
        
        Returns:
            导出文件的完整路径
        
        Requirements: 13.4
        """
        filepath = self.log_dir / filename
        
        data = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "export_time": datetime.now().isoformat(),
            "configs": self._configs,
            "results": self._results,
            "metrics": self._metrics,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON 导出完成: {filepath}")
        return filepath

    def generate_summary(self) -> ExperimentSummary:
        """
        生成实验摘要统计
        
        计算均值、标准差、最优配置等聚合统计信息
        
        Returns:
            ExperimentSummary 实例
        
        Requirements: 13.3
        """
        # 按配置名称分组结果
        results_by_config: Dict[str, List[Dict[str, Any]]] = {}
        for result in self._results:
            config_name = f"{result['scheduler_type']}_{result['num_steps']}"
            if config_name not in results_by_config:
                results_by_config[config_name] = []
            results_by_config[config_name].append(result)
        
        # 计算统计信息
        latency_stats: Dict[str, Dict[str, float]] = {}
        vram_stats: Dict[str, Dict[str, float]] = {}
        
        for config_name, results in results_by_config.items():
            latencies = [r["latency_ms"] for r in results]
            vrams = [r["peak_vram_mb"] for r in results]
            
            latency_stats[config_name] = self._compute_stats(latencies)
            vram_stats[config_name] = self._compute_stats(vrams)
        
        # 计算质量统计 (从 metrics 中提取)
        quality_stats: Dict[str, Dict[str, float]] = {}
        metrics_by_config: Dict[str, List[float]] = {}
        
        for metric_entry in self._metrics:
            metrics = metric_entry.get("metrics", {})
            if "clip_score" in metrics and "config_name" in metrics:
                config_name = metrics["config_name"]
                if config_name not in metrics_by_config:
                    metrics_by_config[config_name] = []
                metrics_by_config[config_name].append(metrics["clip_score"])
        
        for config_name, scores in metrics_by_config.items():
            quality_stats[config_name] = self._compute_stats(scores)
        
        # 确定最优配置
        best_speed_config = ""
        best_quality_config = ""
        best_tradeoff_config = ""
        
        if latency_stats:
            # 最快配置
            best_speed_config = min(
                latency_stats.keys(),
                key=lambda k: latency_stats[k].get("mean", float("inf"))
            )
        
        if quality_stats:
            # 最高质量配置
            best_quality_config = max(
                quality_stats.keys(),
                key=lambda k: quality_stats[k].get("mean", 0)
            )
        
        # 最佳权衡: 速度/质量比 (简化计算)
        if latency_stats and quality_stats:
            tradeoff_scores = {}
            for config_name in latency_stats:
                if config_name in quality_stats:
                    latency = latency_stats[config_name].get("mean", float("inf"))
                    quality = quality_stats[config_name].get("mean", 0)
                    if latency > 0:
                        tradeoff_scores[config_name] = quality / latency * 1000
            if tradeoff_scores:
                best_tradeoff_config = max(
                    tradeoff_scores.keys(),
                    key=lambda k: tradeoff_scores[k]
                )
        elif latency_stats:
            best_tradeoff_config = best_speed_config
        
        # 收集环境信息
        gpu_info, cuda_version, pytorch_version = self._get_environment_info()
        
        # 构建配置列表
        configs = []
        for config_entry in self._configs:
            config_data = config_entry.get("config", {})
            if isinstance(config_data, dict) and "name" in config_data:
                configs.append(ExperimentConfig.from_dict(config_data))
        
        summary = ExperimentSummary(
            experiment_name=self.experiment_name,
            total_runs=len(self._results),
            configs=configs,
            latency_stats=latency_stats,
            vram_stats=vram_stats,
            quality_stats=quality_stats,
            best_speed_config=best_speed_config,
            best_quality_config=best_quality_config,
            best_tradeoff_config=best_tradeoff_config,
            gpu_info=gpu_info,
            cuda_version=cuda_version,
            pytorch_version=pytorch_version,
        )
        
        logger.info(
            f"生成实验摘要: {len(self._results)} 次运行, "
            f"{len(latency_stats)} 个配置"
        )
        
        return summary
    
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
    
    def _get_environment_info(self) -> tuple:
        """获取环境信息"""
        gpu_info = "Unknown"
        cuda_version = "Unknown"
        pytorch_version = "Unknown"
        
        try:
            import torch
            pytorch_version = torch.__version__
            
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda or "Unknown"
                gpu_info = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        return gpu_info, cuda_version, pytorch_version
