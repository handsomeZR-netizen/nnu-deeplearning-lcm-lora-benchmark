"""
PipelineManager - Core inference pipeline management for LCM-LoRA acceleration system.

Manages Stable Diffusion pipeline loading, configuration, optimization, and generation.
Supports baseline (Euler/DPM-Solver/DDIM) and LCM-LoRA accelerated pipelines.
"""

import time
import logging
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional, Dict, Tuple, Literal

import torch
from PIL import Image

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LCMScheduler,
)

from .models import GenerationResult

logger = logging.getLogger(__name__)

# Type aliases
SchedulerType = Literal["euler", "dpm_solver", "ddim", "lcm"]


class VRAMError(Exception):
    """显存不足异常"""
    pass


class ModelLoadError(Exception):
    """模型加载异常"""
    pass


def handle_oom(func):
    """OOM 错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM detected, clearing cache...")
            raise VRAMError("显存不足，请尝试降低分辨率或启用更多优化")
    return wrapper



class PipelineManager:
    """
    管理 Stable Diffusion 推理管线的加载、配置和切换
    
    Supports:
    - Baseline pipelines with Euler/DPM-Solver/DDIM schedulers
    - LCM-LoRA accelerated pipeline with LCM scheduler
    - Memory optimizations (attention slicing, VAE slicing/tiling, xFormers, SDPA)
    - Deterministic generation with seed control
    """
    
    # Scheduler type mapping
    SCHEDULER_MAP = {
        "euler": EulerDiscreteScheduler,
        "dpm_solver": DPMSolverMultistepScheduler,
        "ddim": DDIMScheduler,
        "lcm": LCMScheduler,
    }
    
    def __init__(
        self,
        model_dir: str,
        lcm_lora_dir: str,
        device: str = "cuda"
    ):
        """
        初始化管线管理器
        
        Args:
            model_dir: 基础模型目录 (DreamShaper-7)
            lcm_lora_dir: LCM-LoRA 权重目录
            device: 推理设备 ("cuda" or "cpu")
        """
        self.model_dir = Path(model_dir)
        self.lcm_lora_dir = Path(lcm_lora_dir)
        self.device = device
        
        self._pipeline: Optional[StableDiffusionPipeline] = None
        self._current_scheduler_type: Optional[str] = None
        self._lcm_lora_loaded: bool = False
        self._lora_fused: bool = False
        self._optimizations: Dict[str, bool] = {
            "attention_slicing": False,
            "vae_slicing": False,
            "vae_tiling": False,
            "xformers": False,
            "sdpa": False,
            "cpu_offload": False,
        }
        
        # Validate paths
        if not self.model_dir.exists():
            raise ModelLoadError(f"模型目录不存在: {self.model_dir}")
        if not self.lcm_lora_dir.exists():
            raise ModelLoadError(f"LCM-LoRA 目录不存在: {self.lcm_lora_dir}")
    
    @property
    def pipeline(self) -> Optional[StableDiffusionPipeline]:
        """获取当前管线"""
        return self._pipeline
    
    @property
    def is_loaded(self) -> bool:
        """检查管线是否已加载"""
        return self._pipeline is not None
    
    @property
    def current_scheduler_type(self) -> Optional[str]:
        """获取当前调度器类型"""
        return self._current_scheduler_type
    
    @property
    def optimizations(self) -> Dict[str, bool]:
        """获取当前优化配置"""
        return self._optimizations.copy()
    
    def _load_pipeline_fp16(self) -> StableDiffusionPipeline:
        """
        加载管线，优先使用 FP16 variant，失败则回退到普通加载
        
        Returns:
            加载的管线实例
        """
        kwargs = {"torch_dtype": torch.float16}
        
        try:
            # 尝试 FP16 variant
            logger.info(f"尝试加载 FP16 variant 模型: {self.model_dir}")
            pipe = AutoPipelineForText2Image.from_pretrained(
                str(self.model_dir),
                variant="fp16",
                **kwargs
            )
            logger.info("FP16 variant 加载成功")
        except OSError as e:
            if "variant" in str(e).lower() or "fp16" in str(e).lower():
                # 回退到普通加载
                logger.warning(f"FP16 variant 不可用，回退到普通加载: {e}")
                pipe = AutoPipelineForText2Image.from_pretrained(
                    str(self.model_dir),
                    **kwargs
                )
                logger.info("普通模式加载成功")
            else:
                raise ModelLoadError(f"模型加载失败: {e}")
        except Exception as e:
            raise ModelLoadError(f"模型加载失败: {e}")
        
        return pipe
    
    def _set_scheduler(self, scheduler_type: SchedulerType) -> None:
        """
        设置调度器类型
        
        Args:
            scheduler_type: 调度器类型 ("euler", "dpm_solver", "ddim", "lcm")
        """
        if self._pipeline is None:
            raise RuntimeError("管线未加载，请先调用 load_baseline_pipeline 或 load_lcm_pipeline")
        
        if scheduler_type not in self.SCHEDULER_MAP:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
        
        scheduler_cls = self.SCHEDULER_MAP[scheduler_type]
        
        # Get base config and handle DPM-Solver specific settings
        scheduler_config = dict(self._pipeline.scheduler.config)
        
        if scheduler_type == "dpm_solver":
            # DPM-Solver doesn't support final_sigmas_type="zero" with algorithm_type="deis"
            # Use default algorithm_type and final_sigmas_type for compatibility
            scheduler_config.pop("final_sigmas_type", None)
            scheduler_config.pop("algorithm_type", None)
        
        self._pipeline.scheduler = scheduler_cls.from_config(scheduler_config)
        self._current_scheduler_type = scheduler_type
        logger.info(f"调度器已切换为: {scheduler_type}")

    def load_baseline_pipeline(
        self,
        scheduler_type: SchedulerType = "euler"
    ) -> StableDiffusionPipeline:
        """
        加载基线推理管线
        
        Args:
            scheduler_type: 调度器类型 ("euler", "dpm_solver", "ddim")
        
        Returns:
            配置好的推理管线
        
        Requirements: 1.1, 1.2, 1.3, 1.5
        """
        if scheduler_type == "lcm":
            raise ValueError("基线管线不支持 LCM 调度器，请使用 load_lcm_pipeline")
        
        # 加载管线 (FP16)
        self._pipeline = self._load_pipeline_fp16()
        self._pipeline = self._pipeline.to(self.device)
        
        # 设置调度器
        self._set_scheduler(scheduler_type)
        
        # 重置 LCM-LoRA 状态
        self._lcm_lora_loaded = False
        self._lora_fused = False
        
        logger.info(f"基线管线加载完成: scheduler={scheduler_type}, device={self.device}")
        return self._pipeline
    
    def load_lcm_pipeline(
        self,
        fuse_lora: bool = True
    ) -> StableDiffusionPipeline:
        """
        加载 LCM-LoRA 加速管线
        
        Args:
            fuse_lora: 是否融合 LoRA 权重以减少推理开销
        
        Returns:
            配置好的 LCM 管线
        
        Requirements: 2.1, 2.2, 2.5
        """
        # 加载基础管线 (FP16)
        self._pipeline = self._load_pipeline_fp16()
        self._pipeline = self._pipeline.to(self.device)
        
        # 切换到 LCM Scheduler
        self._set_scheduler("lcm")
        
        # 加载 LCM-LoRA 权重
        logger.info(f"加载 LCM-LoRA 权重: {self.lcm_lora_dir}")
        self._pipeline.load_lora_weights(str(self.lcm_lora_dir))
        self._lcm_lora_loaded = True
        
        # 融合 LoRA 权重
        if fuse_lora and hasattr(self._pipeline, "fuse_lora"):
            logger.info("融合 LoRA 权重...")
            self._pipeline.fuse_lora()
            self._lora_fused = True
        
        logger.info(f"LCM 管线加载完成: fuse_lora={fuse_lora}, device={self.device}")
        return self._pipeline
    
    def apply_optimizations(
        self,
        attention_slicing: bool = True,
        vae_slicing: bool = True,
        vae_tiling: bool = False,
        xformers: bool = False,
        sdpa: bool = True,
        cpu_offload: bool = False
    ) -> None:
        """
        应用显存优化配置
        
        Args:
            attention_slicing: 启用注意力切片以降低峰值显存
            vae_slicing: 启用 VAE 切片以降低 VAE 显存占用
            vae_tiling: 启用 VAE 平铺以支持更高分辨率
            xformers: 使用 xFormers memory-efficient attention (如果可用)
            sdpa: 使用 PyTorch SDPA (如果可用)
            cpu_offload: 启用 CPU offload 作为备选方案
        
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
        """
        if self._pipeline is None:
            raise RuntimeError("管线未加载，请先调用 load_baseline_pipeline 或 load_lcm_pipeline")
        
        # Attention slicing
        if attention_slicing:
            self._pipeline.enable_attention_slicing()
            self._optimizations["attention_slicing"] = True
            logger.info("已启用 attention_slicing")
        else:
            self._pipeline.disable_attention_slicing()
            self._optimizations["attention_slicing"] = False
        
        # VAE slicing
        if vae_slicing and hasattr(self._pipeline, "enable_vae_slicing"):
            self._pipeline.enable_vae_slicing()
            self._optimizations["vae_slicing"] = True
            logger.info("已启用 vae_slicing")
        elif hasattr(self._pipeline, "disable_vae_slicing"):
            self._pipeline.disable_vae_slicing()
            self._optimizations["vae_slicing"] = False
        
        # VAE tiling
        if vae_tiling and hasattr(self._pipeline, "enable_vae_tiling"):
            self._pipeline.enable_vae_tiling()
            self._optimizations["vae_tiling"] = True
            logger.info("已启用 vae_tiling")
        elif hasattr(self._pipeline, "disable_vae_tiling"):
            self._pipeline.disable_vae_tiling()
            self._optimizations["vae_tiling"] = False
        
        # xFormers
        if xformers:
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
                self._optimizations["xformers"] = True
                logger.info("已启用 xFormers memory-efficient attention")
            except Exception as e:
                logger.warning(f"xFormers 不可用: {e}")
                self._optimizations["xformers"] = False
        
        # SDPA (Scaled Dot Product Attention)
        if sdpa and not xformers:  # xFormers 和 SDPA 互斥
            # PyTorch 2.0+ 默认启用 SDPA
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                self._optimizations["sdpa"] = True
                logger.info("SDPA 已可用 (PyTorch 2.0+)")
            else:
                logger.warning("SDPA 不可用 (需要 PyTorch 2.0+)")
                self._optimizations["sdpa"] = False
        
        # CPU offload (备选方案)
        if cpu_offload:
            try:
                self._pipeline.enable_model_cpu_offload()
                self._optimizations["cpu_offload"] = True
                logger.info("已启用 CPU offload")
            except Exception as e:
                logger.warning(f"CPU offload 不可用: {e}")
                self._optimizations["cpu_offload"] = False

    @handle_oom
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        width: int = 512,
        height: int = 512
    ) -> GenerationResult:
        """
        执行图像生成
        
        Args:
            prompt: 文本提示
            num_steps: 推理步数
            guidance_scale: 引导强度 (LCM 建议 0-2)
            seed: 随机种子
            width: 图像宽度
            height: 图像高度
        
        Returns:
            GenerationResult: 包含图像、延迟、显存等信息
        
        Requirements: 1.4, 2.3, 2.4, 4.1, 4.2
        """
        if self._pipeline is None:
            raise RuntimeError("管线未加载，请先调用 load_baseline_pipeline 或 load_lcm_pipeline")
        
        # 设置随机种子
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 重置显存统计
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # 计时开始
        start_time = time.perf_counter()
        
        # 执行生成
        output = self._pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        )
        
        # 计时结束
        if self.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 收集指标
        latency_ms = (end_time - start_time) * 1000
        
        if self.device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram_mb = 0.0
        
        # 获取生成的图像
        image = output.images[0]
        
        # 构建结果
        result = GenerationResult(
            image=image,
            prompt=prompt,
            seed=seed,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            resolution=(width, height),
            latency_ms=latency_ms,
            peak_vram_mb=peak_vram_mb,
            scheduler_type=self._current_scheduler_type or "unknown",
            optimizations=self._optimizations.copy(),
            timestamp=datetime.now()
        )
        
        logger.info(
            f"生成完成: steps={num_steps}, guidance={guidance_scale}, "
            f"latency={latency_ms:.1f}ms, vram={peak_vram_mb:.0f}MB"
        )
        
        return result
    
    def warmup(self, num_steps: int = 2) -> None:
        """
        预热管线，避免首次编译/加载影响计时
        
        Args:
            num_steps: 预热步数
        """
        if self._pipeline is None:
            raise RuntimeError("管线未加载")
        
        logger.info("预热管线...")
        with torch.inference_mode():
            _ = self._pipeline(
                prompt="warmup",
                num_inference_steps=num_steps,
                guidance_scale=0.0
            ).images[0]
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        logger.info("预热完成")
    
    def unload(self) -> None:
        """卸载管线并释放显存"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        
        self._current_scheduler_type = None
        self._lcm_lora_loaded = False
        self._lora_fused = False
        self._optimizations = {k: False for k in self._optimizations}
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("管线已卸载")
    
    def get_vram_usage(self) -> Dict[str, float]:
        """
        获取当前显存使用情况
        
        Returns:
            包含 allocated_mb 和 reserved_mb 的字典
        """
        if self.device != "cuda":
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        }
