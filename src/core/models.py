"""
Core data models for the LCM-LoRA acceleration system.

Implements dataclasses for experiment configuration, generation results,
and metrics collection with serialization/deserialization support.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import json
import base64
from io import BytesIO


@dataclass
class RuntimeMetrics:
    """运行时指标"""
    latency_ms: float
    peak_vram_mb: float
    throughput: float  # images/second
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeMetrics":
        """从字典反序列化"""
        return cls(
            latency_ms=float(data["latency_ms"]),
            peak_vram_mb=float(data["peak_vram_mb"]),
            throughput=float(data["throughput"])
        )


@dataclass
class QualityMetrics:
    """质量评估指标"""
    clip_score: float
    fid: Optional[float] = None
    lpips: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetrics":
        """从字典反序列化"""
        return cls(
            clip_score=float(data["clip_score"]),
            fid=float(data["fid"]) if data.get("fid") is not None else None,
            lpips=float(data["lpips"]) if data.get("lpips") is not None else None
        )


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str  # e.g., "Euler_20", "LCM_4"
    scheduler_type: str  # "euler", "dpm_solver", "lcm"
    num_steps: int
    guidance_scale: float
    use_lcm_lora: bool
    optimizations: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """从字典反序列化"""
        return cls(
            name=str(data["name"]),
            scheduler_type=str(data["scheduler_type"]),
            num_steps=int(data["num_steps"]),
            guidance_scale=float(data["guidance_scale"]),
            use_lcm_lora=bool(data["use_lcm_lora"]),
            optimizations=dict(data.get("optimizations", {}))
        )


@dataclass
class GenerationResult:
    """单次生成的完整结果"""
    image: Optional[Image.Image]
    prompt: str
    seed: int
    num_steps: int
    guidance_scale: float
    resolution: Tuple[int, int]
    latency_ms: float
    peak_vram_mb: float
    scheduler_type: str
    optimizations: Dict[str, bool]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        """
        序列化为字典
        
        Args:
            include_image: 是否包含图像的 base64 编码
        """
        result = {
            "prompt": self.prompt,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
            "resolution": list(self.resolution),
            "latency_ms": self.latency_ms,
            "peak_vram_mb": self.peak_vram_mb,
            "scheduler_type": self.scheduler_type,
            "optimizations": self.optimizations,
            "timestamp": self.timestamp.isoformat()
        }
        
        if include_image and self.image is not None:
            buffer = BytesIO()
            self.image.save(buffer, format="PNG")
            result["image_base64"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """从字典反序列化"""
        image = None
        if "image_base64" in data and data["image_base64"]:
            image_data = base64.b64decode(data["image_base64"])
            image = Image.open(BytesIO(image_data))
        
        return cls(
            image=image,
            prompt=str(data["prompt"]),
            seed=int(data["seed"]),
            num_steps=int(data["num_steps"]),
            guidance_scale=float(data["guidance_scale"]),
            resolution=tuple(data["resolution"]),
            latency_ms=float(data["latency_ms"]),
            peak_vram_mb=float(data["peak_vram_mb"]),
            scheduler_type=str(data["scheduler_type"]),
            optimizations=dict(data.get("optimizations", {})),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ExperimentSummary:
    """实验摘要"""
    experiment_name: str
    total_runs: int
    configs: List[ExperimentConfig]
    
    # 聚合统计: config_name -> {mean, std, min, max}
    latency_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vram_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 最优配置
    best_speed_config: str = ""
    best_quality_config: str = ""
    best_tradeoff_config: str = ""
    
    # 环境信息
    gpu_info: str = ""
    cuda_version: str = ""
    pytorch_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "experiment_name": self.experiment_name,
            "total_runs": self.total_runs,
            "configs": [c.to_dict() for c in self.configs],
            "latency_stats": self.latency_stats,
            "vram_stats": self.vram_stats,
            "quality_stats": self.quality_stats,
            "best_speed_config": self.best_speed_config,
            "best_quality_config": self.best_quality_config,
            "best_tradeoff_config": self.best_tradeoff_config,
            "gpu_info": self.gpu_info,
            "cuda_version": self.cuda_version,
            "pytorch_version": self.pytorch_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentSummary":
        """从字典反序列化"""
        return cls(
            experiment_name=str(data["experiment_name"]),
            total_runs=int(data["total_runs"]),
            configs=[ExperimentConfig.from_dict(c) for c in data.get("configs", [])],
            latency_stats=dict(data.get("latency_stats", {})),
            vram_stats=dict(data.get("vram_stats", {})),
            quality_stats=dict(data.get("quality_stats", {})),
            best_speed_config=str(data.get("best_speed_config", "")),
            best_quality_config=str(data.get("best_quality_config", "")),
            best_tradeoff_config=str(data.get("best_tradeoff_config", "")),
            gpu_info=str(data.get("gpu_info", "")),
            cuda_version=str(data.get("cuda_version", "")),
            pytorch_version=str(data.get("pytorch_version", ""))
        )
    
    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentSummary":
        """从 JSON 字符串反序列化"""
        return cls.from_dict(json.loads(json_str))
