"""
MetricsCollector - Quality and runtime metrics collection for LCM-LoRA acceleration system.

Implements quality evaluation metrics including CLIPScore, LPIPS, and FID.
Ensures evaluation models are decoupled from inference pipeline for accurate VRAM measurement.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from ..core.models import QualityMetrics, RuntimeMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    收集和计算各类评估指标
    
    Supports:
    - CLIPScore: 图文相似度评估
    - LPIPS: 感知相似度评估
    - FID: 分布距离评估 (可选)
    - Runtime metrics: 延迟、显存、吞吐量
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    
    def __init__(
        self,
        device: str = "cuda",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        lpips_net: str = "alex",
    ):
        """
        初始化评估模型
        
        Args:
            device: 评估设备 ("cuda" or "cpu")
            clip_model_name: CLIP 模型名称
            lpips_net: LPIPS 网络类型 ("alex", "vgg", "squeeze")
        """
        self.device = device
        self.clip_model_name = clip_model_name
        self.lpips_net = lpips_net
        
        # Lazy loading - models loaded on first use
        self._clip_model = None
        self._clip_processor = None
        self._lpips_model = None
        self._inception_model = None
        
        logger.info(f"MetricsCollector initialized: device={device}")
    
    def _ensure_clip_loaded(self) -> None:
        """确保 CLIP 模型已加载"""
        if self._clip_model is not None:
            return
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info(f"Loading CLIP model: {self.clip_model_name}")
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"无法加载 CLIP 模型: {e}")
    
    def _ensure_lpips_loaded(self) -> None:
        """确保 LPIPS 模型已加载"""
        if self._lpips_model is not None:
            return
        
        try:
            import lpips
            
            logger.info(f"Loading LPIPS model: {self.lpips_net}")
            self._lpips_model = lpips.LPIPS(net=self.lpips_net)
            self._lpips_model = self._lpips_model.to(self.device)
            self._lpips_model.eval()
            logger.info("LPIPS model loaded successfully")
        except ImportError:
            logger.error("lpips package not installed. Install with: pip install lpips")
            raise RuntimeError("lpips 包未安装，请运行: pip install lpips")
        except Exception as e:
            logger.error(f"Failed to load LPIPS model: {e}")
            raise RuntimeError(f"无法加载 LPIPS 模型: {e}")

    @torch.inference_mode()
    def compute_clip_score(self, image: Image.Image, prompt: str) -> float:
        """
        计算 CLIPScore 图文相似度
        
        Args:
            image: PIL 图像
            prompt: 文本提示
        
        Returns:
            CLIPScore 值，范围 [0, 1]
        
        Requirements: 5.1, 5.4
        """
        self._ensure_clip_loaded()
        
        # Process inputs
        inputs = self._clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self._clip_model(**inputs)
        
        # Compute cosine similarity
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = torch.sum(image_embeds * text_embeds, dim=-1)
        
        # Convert to [0, 1] range (cosine similarity is in [-1, 1])
        clip_score = (similarity.item() + 1.0) / 2.0
        
        # Clamp to ensure valid range
        clip_score = max(0.0, min(1.0, clip_score))
        
        logger.debug(f"CLIPScore: {clip_score:.4f} for prompt: {prompt[:50]}...")
        return clip_score
    
    def compute_clip_score_batch(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> List[float]:
        """
        批量计算 CLIPScore
        
        Args:
            images: PIL 图像列表
            prompts: 文本提示列表
        
        Returns:
            CLIPScore 值列表
        
        Requirements: 5.5
        """
        if len(images) != len(prompts):
            raise ValueError("图像数量和提示数量必须相同")
        
        scores = []
        for image, prompt in zip(images, prompts):
            score = self.compute_clip_score(image, prompt)
            scores.append(score)
        
        return scores

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        将 PIL 图像转换为 LPIPS 所需的张量格式
        
        Args:
            image: PIL 图像
        
        Returns:
            张量，形状 [1, 3, H, W]，范围 [-1, 1]
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        arr = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        
        # Normalize to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    @torch.inference_mode()
    def compute_lpips(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        计算 LPIPS 感知相似度
        
        Args:
            image1: 第一张 PIL 图像
            image2: 第二张 PIL 图像
        
        Returns:
            LPIPS 值，越小表示越相似
        
        Requirements: 5.3
        """
        self._ensure_lpips_loaded()
        
        # Resize images to same size if needed
        if image1.size != image2.size:
            # Resize to smaller image's size
            target_size = (
                min(image1.size[0], image2.size[0]),
                min(image1.size[1], image2.size[1])
            )
            image1 = image1.resize(target_size, Image.Resampling.LANCZOS)
            image2 = image2.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensors
        tensor1 = self._image_to_tensor(image1)
        tensor2 = self._image_to_tensor(image2)
        
        # Compute LPIPS
        lpips_value = self._lpips_model(tensor1, tensor2)
        
        result = lpips_value.item()
        logger.debug(f"LPIPS: {result:.4f}")
        return result
    
    def compute_lpips_batch(
        self,
        images1: List[Image.Image],
        images2: List[Image.Image]
    ) -> List[float]:
        """
        批量计算 LPIPS
        
        Args:
            images1: 第一组 PIL 图像列表
            images2: 第二组 PIL 图像列表
        
        Returns:
            LPIPS 值列表
        
        Requirements: 5.5
        """
        if len(images1) != len(images2):
            raise ValueError("两组图像数量必须相同")
        
        scores = []
        for img1, img2 in zip(images1, images2):
            score = self.compute_lpips(img1, img2)
            scores.append(score)
        
        return scores

    def _ensure_inception_loaded(self) -> None:
        """确保 Inception 模型已加载 (用于 FID 计算)"""
        if self._inception_model is not None:
            return
        
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            
            logger.info("Loading Inception model for FID computation")
            self._inception_model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )
            # Remove final classification layer
            self._inception_model.fc = torch.nn.Identity()
            self._inception_model = self._inception_model.to(self.device)
            self._inception_model.eval()
            logger.info("Inception model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Inception model: {e}")
            raise RuntimeError(f"无法加载 Inception 模型: {e}")
    
    def _preprocess_for_inception(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像用于 Inception 模型
        
        Args:
            image: PIL 图像
        
        Returns:
            预处理后的张量
        """
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return preprocess(image).unsqueeze(0).to(self.device)
    
    @torch.inference_mode()
    def _get_inception_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        获取图像的 Inception 特征
        
        Args:
            images: PIL 图像列表
        
        Returns:
            特征数组，形状 [N, 2048]
        """
        self._ensure_inception_loaded()
        
        features_list = []
        for image in images:
            tensor = self._preprocess_for_inception(image)
            features = self._inception_model(tensor)
            features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def _calculate_fid_from_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        从特征计算 FID
        
        Args:
            features1: 第一组特征
            features2: 第二组特征
        
        Returns:
            FID 值
        """
        from scipy import linalg
        
        # Calculate mean and covariance
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
    
    def compute_fid(
        self,
        generated_images: List[Image.Image],
        reference_images: List[Image.Image]
    ) -> float:
        """
        计算 FID 分布距离
        
        Args:
            generated_images: 生成的图像列表
            reference_images: 参考图像列表
        
        Returns:
            FID 值，越小表示分布越接近
        
        Requirements: 5.2, 5.5
        
        Note: FID 需要足够数量的图像才能准确计算 (建议 >= 50)
        """
        if len(generated_images) < 2 or len(reference_images) < 2:
            raise ValueError("FID 计算需要至少 2 张图像")
        
        logger.info(f"Computing FID: {len(generated_images)} generated vs {len(reference_images)} reference images")
        
        # Get features
        gen_features = self._get_inception_features(generated_images)
        ref_features = self._get_inception_features(reference_images)
        
        # Calculate FID
        fid = self._calculate_fid_from_features(gen_features, ref_features)
        
        logger.info(f"FID: {fid:.4f}")
        return fid

    def collect_runtime_metrics(
        self,
        latency_ms: float,
        peak_vram_mb: float,
        batch_size: int = 1
    ) -> RuntimeMetrics:
        """
        收集运行时指标
        
        Args:
            latency_ms: 延迟 (毫秒)
            peak_vram_mb: 峰值显存 (MB)
            batch_size: 批次大小
        
        Returns:
            RuntimeMetrics 对象
        """
        throughput = (batch_size / latency_ms) * 1000.0  # images/second
        
        return RuntimeMetrics(
            latency_ms=latency_ms,
            peak_vram_mb=peak_vram_mb,
            throughput=throughput
        )
    
    def collect_quality_metrics(
        self,
        image: Image.Image,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        compute_fid: bool = False,
        reference_images: Optional[List[Image.Image]] = None,
        generated_images: Optional[List[Image.Image]] = None,
    ) -> QualityMetrics:
        """
        收集质量评估指标
        
        Args:
            image: 生成的图像
            prompt: 文本提示
            reference_image: 参考图像 (用于 LPIPS)
            compute_fid: 是否计算 FID
            reference_images: 参考图像列表 (用于 FID)
            generated_images: 生成图像列表 (用于 FID)
        
        Returns:
            QualityMetrics 对象
        """
        # Compute CLIPScore
        clip_score = self.compute_clip_score(image, prompt)
        
        # Compute LPIPS if reference provided
        lpips_value = None
        if reference_image is not None:
            lpips_value = self.compute_lpips(image, reference_image)
        
        # Compute FID if requested and images provided
        fid_value = None
        if compute_fid and reference_images is not None and generated_images is not None:
            fid_value = self.compute_fid(generated_images, reference_images)
        
        return QualityMetrics(
            clip_score=clip_score,
            fid=fid_value,
            lpips=lpips_value
        )
    
    def get_vram_usage(self) -> Dict[str, float]:
        """
        获取当前评估模型的显存使用情况
        
        Returns:
            包含 allocated_mb 和 reserved_mb 的字典
        
        Requirements: 5.4 (确保评估与推理显存分离)
        """
        if self.device != "cuda":
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        }
    
    def unload_models(self) -> None:
        """卸载所有评估模型并释放显存"""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_processor
            self._clip_model = None
            self._clip_processor = None
        
        if self._lpips_model is not None:
            del self._lpips_model
            self._lpips_model = None
        
        if self._inception_model is not None:
            del self._inception_model
            self._inception_model = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("评估模型已卸载")
    
    def __del__(self):
        """析构时释放资源"""
        try:
            self.unload_models()
        except Exception:
            pass
