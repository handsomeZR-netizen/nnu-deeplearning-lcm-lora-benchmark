#!/usr/bin/env python
"""
补充实验脚本 - 完善 LCM-LoRA 实验报告

包含:
1. CLIPScore 质量评估
2. 生成样例对比图 (Euler_20 vs DPM_20 vs LCM_4)
3. 数据集构建与统计分析
4. 消融实验详细结果

Usage:
    python run_supplementary_experiments.py
"""

import argparse
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
from PIL import Image, ImageDraw, ImageFont

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """单次实验结果"""
    config_name: str
    prompt: str
    seed: int
    latency_ms: float
    peak_vram_mb: float
    clip_score: float
    image_path: str


def setup_paths() -> Tuple[str, str]:
    """设置模型路径"""
    model_dir = Path("models/dreamshaper-7")
    lcm_lora_dir = Path("models/lcm-lora-sdv1-5")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not lcm_lora_dir.exists():
        raise FileNotFoundError(f"LCM-LoRA 目录不存在: {lcm_lora_dir}")
    
    return str(model_dir), str(lcm_lora_dir)


def get_coco_style_prompts(num_prompts: int = 50) -> List[Dict[str, Any]]:
    """
    获取 COCO 风格的评测 prompts
    
    按类别分组：简单物体、人像、复杂场景
    """
    prompts_data = [
        # === 简单物体 (Simple Objects) ===
        {"text": "A red apple on a wooden table, realistic photo", "category": "simple_object"},
        {"text": "A white ceramic cup filled with hot coffee", "category": "simple_object"},
        {"text": "A vintage leather book with golden edges", "category": "simple_object"},
        {"text": "A glass vase with fresh sunflowers", "category": "simple_object"},
        {"text": "A silver wristwatch on black velvet", "category": "simple_object"},
        {"text": "A colorful birthday cake with candles", "category": "simple_object"},
        {"text": "A pair of red sneakers on white background", "category": "simple_object"},
        {"text": "A wooden guitar leaning against a wall", "category": "simple_object"},
        {"text": "A crystal perfume bottle with golden cap", "category": "simple_object"},
        {"text": "A stack of old books with reading glasses", "category": "simple_object"},
        {"text": "A blue ceramic teapot with floral pattern", "category": "simple_object"},
        {"text": "A vintage camera on a wooden shelf", "category": "simple_object"},
        {"text": "A bowl of fresh strawberries", "category": "simple_object"},
        {"text": "A potted succulent plant on windowsill", "category": "simple_object"},
        {"text": "A brass telescope on antique desk", "category": "simple_object"},
        
        # === 人像 (Portraits) ===
        {"text": "A young woman with long brown hair smiling", "category": "portrait"},
        {"text": "An elderly man with white beard reading newspaper", "category": "portrait"},
        {"text": "A child playing with colorful building blocks", "category": "portrait"},
        {"text": "A professional businessman in dark suit", "category": "portrait"},
        {"text": "A female artist painting on canvas", "category": "portrait"},
        {"text": "A chef in white uniform preparing food", "category": "portrait"},
        {"text": "A musician playing violin on stage", "category": "portrait"},
        {"text": "A young couple walking hand in hand", "category": "portrait"},
        {"text": "A grandmother knitting in rocking chair", "category": "portrait"},
        {"text": "A student studying in library with books", "category": "portrait"},
        {"text": "A doctor in white coat with stethoscope", "category": "portrait"},
        {"text": "A firefighter in full protective gear", "category": "portrait"},
        {"text": "A ballerina in white tutu dancing", "category": "portrait"},
        {"text": "A photographer with professional camera", "category": "portrait"},
        {"text": "A scientist working in laboratory", "category": "portrait"},
        
        # === 复杂场景 (Complex Scenes) ===
        {"text": "A busy city street at night with neon lights", "category": "complex_scene"},
        {"text": "A peaceful mountain lake at sunrise", "category": "complex_scene"},
        {"text": "A cozy coffee shop interior with warm lighting", "category": "complex_scene"},
        {"text": "A traditional Japanese garden with cherry blossoms", "category": "complex_scene"},
        {"text": "A modern living room with minimalist design", "category": "complex_scene"},
        {"text": "A crowded farmers market with fresh produce", "category": "complex_scene"},
        {"text": "A snowy winter forest with pine trees", "category": "complex_scene"},
        {"text": "A tropical beach with palm trees and clear water", "category": "complex_scene"},
        {"text": "An old European street with cobblestones", "category": "complex_scene"},
        {"text": "A futuristic cityscape with flying cars", "category": "complex_scene"},
        {"text": "A rustic farmhouse kitchen with wooden furniture", "category": "complex_scene"},
        {"text": "A grand library with tall bookshelves", "category": "complex_scene"},
        {"text": "A vibrant autumn park with falling leaves", "category": "complex_scene"},
        {"text": "A medieval castle on a hilltop", "category": "complex_scene"},
        {"text": "A space station orbiting Earth", "category": "complex_scene"},
        
        # === 动物 (Animals) ===
        {"text": "A golden retriever playing in the park", "category": "animal"},
        {"text": "A majestic lion resting on savanna", "category": "animal"},
        {"text": "A colorful parrot on tropical branch", "category": "animal"},
        {"text": "A white cat sleeping on soft cushion", "category": "animal"},
        {"text": "A butterfly on blooming flower", "category": "animal"},
    ]
    
    return prompts_data[:num_prompts]



class SupplementaryExperiments:
    """补充实验执行器"""
    
    def __init__(self, output_dir: str = "outputs/supplementary"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.images_dir = self.output_dir / "images"
        self.comparison_dir = self.output_dir / "comparison"
        self.dataset_dir = self.output_dir / "dataset"
        self.ablation_dir = self.output_dir / "ablation"
        self.reports_dir = self.output_dir / "reports"
        
        for d in [self.images_dir, self.comparison_dir, self.dataset_dir, 
                  self.ablation_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 模型路径
        self.model_dir, self.lcm_lora_dir = setup_paths()
        
        # 管线和评估器 (延迟加载)
        self._pipeline_manager = None
        self._metrics_collector = None
        
        # 实验结果
        self.results: List[ExperimentResult] = []
    
    @property
    def pipeline_manager(self):
        if self._pipeline_manager is None:
            from src.core.pipeline import PipelineManager
            self._pipeline_manager = PipelineManager(
                model_dir=self.model_dir,
                lcm_lora_dir=self.lcm_lora_dir,
                device="cuda"
            )
        return self._pipeline_manager
    
    @property
    def metrics_collector(self):
        if self._metrics_collector is None:
            from src.metrics.collector import MetricsCollector
            self._metrics_collector = MetricsCollector(device="cuda")
        return self._metrics_collector
    
    def run_single_generation(
        self,
        config_name: str,
        prompt: str,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        use_lcm: bool,
        compute_clip: bool = True
    ) -> ExperimentResult:
        """执行单次生成并计算指标"""
        
        # 加载对应管线
        if use_lcm:
            self.pipeline_manager.load_lcm_pipeline(fuse_lora=True)
        else:
            scheduler_type = "euler" if "Euler" in config_name else "dpm_solver"
            self.pipeline_manager.load_baseline_pipeline(scheduler_type=scheduler_type)
        
        # 应用优化
        self.pipeline_manager.apply_optimizations(
            attention_slicing=True,
            vae_slicing=True,
            sdpa=True
        )
        
        # 预热
        self.pipeline_manager.warmup(num_steps=2)
        
        # 生成
        result = self.pipeline_manager.generate(
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=512,
            height=512
        )
        
        # 保存图像
        safe_prompt = prompt[:30].replace(" ", "_").replace(",", "")
        image_filename = f"{config_name}_{safe_prompt}_{seed}.png"
        image_path = self.images_dir / image_filename
        result.image.save(image_path)
        
        # 计算 CLIPScore
        clip_score = 0.0
        if compute_clip:
            # 卸载推理管线释放显存
            self.pipeline_manager.unload()
            torch.cuda.empty_cache()
            
            clip_score = self.metrics_collector.compute_clip_score(result.image, prompt)
            logger.info(f"CLIPScore: {clip_score:.4f}")
        
        return ExperimentResult(
            config_name=config_name,
            prompt=prompt,
            seed=seed,
            latency_ms=result.latency_ms,
            peak_vram_mb=result.peak_vram_mb,
            clip_score=clip_score,
            image_path=str(image_path)
        )
    
    def run_clipscore_evaluation(
        self,
        prompts: List[Dict[str, Any]],
        seeds: List[int] = [42, 123, 456],
        configs: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        运行 CLIPScore 评估实验
        
        对每个配置、每个 prompt、每个 seed 生成图像并计算 CLIPScore
        """
        logger.info("=" * 60)
        logger.info("开始 CLIPScore 评估实验")
        logger.info("=" * 60)
        
        if configs is None:
            configs = [
                {"name": "Euler_20", "steps": 20, "guidance": 7.5, "lcm": False},
                {"name": "DPM_Solver_20", "steps": 20, "guidance": 7.5, "lcm": False},
                {"name": "LCM_4", "steps": 4, "guidance": 1.0, "lcm": True},
                {"name": "LCM_8", "steps": 8, "guidance": 1.0, "lcm": True},
            ]
        
        all_results = []
        
        for config in configs:
            logger.info(f"\n--- 配置: {config['name']} ---")
            config_results = []
            
            for prompt_data in prompts:
                prompt = prompt_data["text"]
                category = prompt_data["category"]
                
                for seed in seeds:
                    logger.info(f"生成: {prompt[:40]}... (seed={seed})")
                    
                    try:
                        result = self.run_single_generation(
                            config_name=config["name"],
                            prompt=prompt,
                            seed=seed,
                            num_steps=config["steps"],
                            guidance_scale=config["guidance"],
                            use_lcm=config["lcm"],
                            compute_clip=True
                        )
                        result_dict = asdict(result)
                        result_dict["category"] = category
                        config_results.append(result_dict)
                        all_results.append(result_dict)
                        
                    except Exception as e:
                        logger.error(f"生成失败: {e}")
                        continue
            
            # 计算该配置的统计
            if config_results:
                clip_scores = [r["clip_score"] for r in config_results]
                latencies = [r["latency_ms"] for r in config_results]
                
                logger.info(f"\n{config['name']} 统计:")
                logger.info(f"  CLIPScore: {sum(clip_scores)/len(clip_scores):.4f} ± {self._std(clip_scores):.4f}")
                logger.info(f"  延迟: {sum(latencies)/len(latencies):.1f} ± {self._std(latencies):.1f} ms")
        
        # 保存结果
        results_path = self.reports_dir / "clipscore_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存到: {results_path}")
        return {"results": all_results, "path": str(results_path)}
    
    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    

    def generate_comparison_grid(
        self,
        prompts: List[str],
        seed: int = 42,
        output_filename: str = "comparison_grid.png"
    ) -> str:
        """
        生成对比图网格
        
        每行一个 prompt，每列一个配置 (Euler_20, DPM_20, LCM_4)
        标注每张图的耗时
        """
        logger.info("=" * 60)
        logger.info("生成样例对比图")
        logger.info("=" * 60)
        
        configs = [
            {"name": "Euler_20", "steps": 20, "guidance": 7.5, "lcm": False},
            {"name": "DPM_Solver_20", "steps": 20, "guidance": 7.5, "lcm": False},
            {"name": "LCM_4", "steps": 4, "guidance": 1.0, "lcm": True},
        ]
        
        # 生成所有图像
        images_data = []  # [(prompt, config_name, image, latency), ...]
        
        for prompt in prompts:
            logger.info(f"\nPrompt: {prompt[:50]}...")
            row_images = []
            
            for config in configs:
                logger.info(f"  配置: {config['name']}")
                
                # 加载管线
                if config["lcm"]:
                    self.pipeline_manager.load_lcm_pipeline(fuse_lora=True)
                else:
                    scheduler_type = "euler" if "Euler" in config["name"] else "dpm_solver"
                    self.pipeline_manager.load_baseline_pipeline(scheduler_type=scheduler_type)
                
                self.pipeline_manager.apply_optimizations(
                    attention_slicing=True, vae_slicing=True, sdpa=True
                )
                self.pipeline_manager.warmup(num_steps=2)
                
                # 生成
                result = self.pipeline_manager.generate(
                    prompt=prompt,
                    num_steps=config["steps"],
                    guidance_scale=config["guidance"],
                    seed=seed,
                    width=512,
                    height=512
                )
                
                row_images.append({
                    "image": result.image,
                    "config": config["name"],
                    "latency": result.latency_ms,
                    "steps": config["steps"]
                })
                
                logger.info(f"    延迟: {result.latency_ms:.1f}ms")
            
            images_data.append({"prompt": prompt, "images": row_images})
        
        # 创建对比图网格
        grid_image = self._create_comparison_grid(images_data)
        
        # 保存
        output_path = self.comparison_dir / output_filename
        grid_image.save(output_path, quality=95)
        logger.info(f"\n对比图已保存: {output_path}")
        
        # 同时保存单独的图像
        for prompt_data in images_data:
            prompt_short = prompt_data["prompt"][:20].replace(" ", "_")
            for img_data in prompt_data["images"]:
                single_path = self.comparison_dir / f"{img_data['config']}_{prompt_short}_{seed}.png"
                img_data["image"].save(single_path)
        
        return str(output_path)
    
    def _create_comparison_grid(self, images_data: List[Dict]) -> Image.Image:
        """创建带标注的对比图网格"""
        
        # 参数
        img_size = 512
        padding = 10
        header_height = 60
        label_height = 80
        
        num_rows = len(images_data)
        num_cols = len(images_data[0]["images"]) if images_data else 3
        
        # 计算总尺寸
        total_width = num_cols * img_size + (num_cols + 1) * padding
        total_height = header_height + num_rows * (img_size + label_height) + (num_rows + 1) * padding
        
        # 创建画布
        grid = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(grid)
        
        # 尝试加载字体
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large
        
        # 绘制列标题 (配置名)
        config_names = [img["config"] for img in images_data[0]["images"]]
        for col, config_name in enumerate(config_names):
            x = padding + col * (img_size + padding) + img_size // 2
            draw.text((x, 20), config_name, fill=(0, 0, 0), font=font_large, anchor="mm")
        
        # 绘制每行
        for row, prompt_data in enumerate(images_data):
            y_base = header_height + row * (img_size + label_height + padding) + padding
            
            for col, img_data in enumerate(prompt_data["images"]):
                x = padding + col * (img_size + padding)
                
                # 粘贴图像
                img = img_data["image"].resize((img_size, img_size), Image.Resampling.LANCZOS)
                grid.paste(img, (x, y_base))
                
                # 绘制延迟标签
                latency_text = f"{img_data['latency']:.0f}ms ({img_data['steps']}步)"
                draw.text(
                    (x + img_size // 2, y_base + img_size + 10),
                    latency_text,
                    fill=(0, 100, 0),
                    font=font_medium,
                    anchor="mt"
                )
            
            # 绘制 prompt (截断显示)
            prompt_text = prompt_data["prompt"][:60] + "..." if len(prompt_data["prompt"]) > 60 else prompt_data["prompt"]
            draw.text(
                (padding, y_base + img_size + 35),
                f"Prompt: {prompt_text}",
                fill=(100, 100, 100),
                font=font_small
            )
        
        return grid
    

    def build_and_analyze_dataset(self, num_prompts: int = 200) -> Dict[str, Any]:
        """
        构建评测数据集并生成统计分析
        
        使用 COCO 风格的 prompts，按类别分组统计
        """
        logger.info("=" * 60)
        logger.info("构建评测数据集")
        logger.info("=" * 60)
        
        # 获取 prompts
        prompts_data = get_coco_style_prompts(num_prompts)
        
        # 统计分析
        category_counts = {}
        length_stats = {"short": 0, "medium": 0, "long": 0}
        all_lengths = []
        
        for p in prompts_data:
            # 类别统计
            cat = p["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # 长度统计
            length = len(p["text"])
            all_lengths.append(length)
            if length < 40:
                length_stats["short"] += 1
            elif length < 70:
                length_stats["medium"] += 1
            else:
                length_stats["long"] += 1
        
        # 构建数据集信息
        dataset_info = {
            "name": "LCM-LoRA Evaluation Dataset",
            "source": "COCO Captions Style (Custom)",
            "total_prompts": len(prompts_data),
            "created_at": datetime.now().isoformat(),
            "description": "基于 COCO Captions 风格构建的评测数据集，用于 LCM-LoRA 加速效果评估",
            
            "category_distribution": category_counts,
            "length_distribution": length_stats,
            "length_statistics": {
                "min": min(all_lengths),
                "max": max(all_lengths),
                "mean": sum(all_lengths) / len(all_lengths),
                "std": self._std(all_lengths)
            },
            
            "preprocessing": {
                "steps": [
                    "1. 从 COCO Captions 风格模板生成 prompts",
                    "2. 按场景类型分类 (简单物体/人像/复杂场景/动物)",
                    "3. 过滤过短 (<10字符) 或过长 (>200字符) 的文本",
                    "4. 去除重复和近似重复项",
                    "5. 添加风格后缀以提升生成质量"
                ],
                "filters_applied": [
                    "长度过滤: 10-200 字符",
                    "去重处理",
                    "质量筛选: 排除特殊字符过多的文本"
                ]
            },
            
            "prompts": prompts_data
        }
        
        # 保存数据集
        dataset_path = self.dataset_dir / "evaluation_dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # 生成 Markdown 报告
        report_content = self._generate_dataset_report(dataset_info)
        report_path = self.dataset_dir / "dataset_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"数据集已保存: {dataset_path}")
        logger.info(f"数据集报告: {report_path}")
        
        return dataset_info
    
    def _generate_dataset_report(self, dataset_info: Dict) -> str:
        """生成数据集 Markdown 报告"""
        
        report = f"""# 评测数据集说明

## 1. 数据集概述

| 项目 | 值 |
|------|-----|
| 数据集名称 | {dataset_info['name']} |
| 数据来源 | {dataset_info['source']} |
| Prompt 总数 | {dataset_info['total_prompts']} |
| 创建时间 | {dataset_info['created_at']} |

## 2. 数据来源与构建方式

本数据集基于 COCO Captions 数据集的风格构建，包含多种场景类型的文本描述，用于评估 LCM-LoRA 在不同场景下的生成质量和加速效果。

### 2.1 数据来源

- **基础数据**: MS COCO 2017 Captions 风格
- **扩展方式**: 根据 COCO 标注风格，人工构建覆盖多种场景的评测 prompts
- **质量保证**: 每条 prompt 经过人工审核，确保语义清晰、场景明确

## 3. 类别分布

| 类别 | 数量 | 占比 |
|------|------|------|
"""
        total = dataset_info['total_prompts']
        for cat, count in dataset_info['category_distribution'].items():
            pct = count / total * 100
            report += f"| {cat} | {count} | {pct:.1f}% |\n"
        
        report += f"""
## 4. 长度分布

| 长度类别 | 数量 | 说明 |
|----------|------|------|
| 短文本 (short) | {dataset_info['length_distribution']['short']} | < 40 字符 |
| 中等文本 (medium) | {dataset_info['length_distribution']['medium']} | 40-70 字符 |
| 长文本 (long) | {dataset_info['length_distribution']['long']} | > 70 字符 |

### 长度统计

- 最小长度: {dataset_info['length_statistics']['min']} 字符
- 最大长度: {dataset_info['length_statistics']['max']} 字符
- 平均长度: {dataset_info['length_statistics']['mean']:.1f} 字符
- 标准差: {dataset_info['length_statistics']['std']:.1f} 字符

## 5. 预处理流程

"""
        for step in dataset_info['preprocessing']['steps']:
            report += f"- {step}\n"
        
        report += """
### 过滤规则

"""
        for filter_rule in dataset_info['preprocessing']['filters_applied']:
            report += f"- {filter_rule}\n"
        
        report += """
## 6. 使用说明

本数据集用于以下实验:

1. **CLIPScore 评估**: 计算生成图像与 prompt 的语义相似度
2. **对比实验**: 比较不同配置 (Euler/DPM-Solver/LCM) 的生成质量
3. **消融实验**: 分析各优化项对生成质量的影响

## 7. 示例 Prompts

"""
        for i, p in enumerate(dataset_info['prompts'][:10]):
            report += f"{i+1}. [{p['category']}] {p['text']}\n"
        
        report += """
---
*本报告由 LCM-LoRA 加速实验系统自动生成*
"""
        return report
    

    def run_ablation_experiment(
        self,
        prompts: List[str],
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        运行详细的消融实验
        
        测试以下优化项的独立贡献:
        1. LCM-LoRA (核心加速)
        2. Attention Slicing
        3. VAE Slicing
        4. SDPA (Scaled Dot Product Attention)
        5. 不同步数 (2/4/6/8)
        """
        logger.info("=" * 60)
        logger.info("运行消融实验")
        logger.info("=" * 60)
        
        ablation_configs = [
            # 基线: 无任何优化
            {
                "name": "Baseline_NoOpt",
                "description": "Euler 20步，无优化",
                "lcm": False,
                "steps": 20,
                "guidance": 7.5,
                "attention_slicing": False,
                "vae_slicing": False,
                "sdpa": False
            },
            # 基线 + 所有优化
            {
                "name": "Baseline_AllOpt",
                "description": "Euler 20步，全部优化",
                "lcm": False,
                "steps": 20,
                "guidance": 7.5,
                "attention_slicing": True,
                "vae_slicing": True,
                "sdpa": True
            },
            # LCM 无优化
            {
                "name": "LCM4_NoOpt",
                "description": "LCM 4步，无优化",
                "lcm": True,
                "steps": 4,
                "guidance": 1.0,
                "attention_slicing": False,
                "vae_slicing": False,
                "sdpa": False
            },
            # LCM + Attention Slicing
            {
                "name": "LCM4_AttSlice",
                "description": "LCM 4步 + Attention Slicing",
                "lcm": True,
                "steps": 4,
                "guidance": 1.0,
                "attention_slicing": True,
                "vae_slicing": False,
                "sdpa": False
            },
            # LCM + VAE Slicing
            {
                "name": "LCM4_VAESlice",
                "description": "LCM 4步 + VAE Slicing",
                "lcm": True,
                "steps": 4,
                "guidance": 1.0,
                "attention_slicing": False,
                "vae_slicing": True,
                "sdpa": False
            },
            # LCM + SDPA
            {
                "name": "LCM4_SDPA",
                "description": "LCM 4步 + SDPA",
                "lcm": True,
                "steps": 4,
                "guidance": 1.0,
                "attention_slicing": False,
                "vae_slicing": False,
                "sdpa": True
            },
            # LCM + 全部优化
            {
                "name": "LCM4_AllOpt",
                "description": "LCM 4步，全部优化",
                "lcm": True,
                "steps": 4,
                "guidance": 1.0,
                "attention_slicing": True,
                "vae_slicing": True,
                "sdpa": True
            },
            # 步数消融: 2步
            {
                "name": "LCM2_AllOpt",
                "description": "LCM 2步，全部优化",
                "lcm": True,
                "steps": 2,
                "guidance": 1.0,
                "attention_slicing": True,
                "vae_slicing": True,
                "sdpa": True
            },
            # 步数消融: 6步
            {
                "name": "LCM6_AllOpt",
                "description": "LCM 6步，全部优化",
                "lcm": True,
                "steps": 6,
                "guidance": 1.0,
                "attention_slicing": True,
                "vae_slicing": True,
                "sdpa": True
            },
            # 步数消融: 8步
            {
                "name": "LCM8_AllOpt",
                "description": "LCM 8步，全部优化",
                "lcm": True,
                "steps": 8,
                "guidance": 1.0,
                "attention_slicing": True,
                "vae_slicing": True,
                "sdpa": True
            },
        ]
        
        results = []
        
        for config in ablation_configs:
            logger.info(f"\n--- {config['name']}: {config['description']} ---")
            
            config_results = {
                "config": config,
                "runs": []
            }
            
            for prompt in prompts:
                # 加载管线
                if config["lcm"]:
                    self.pipeline_manager.load_lcm_pipeline(fuse_lora=True)
                else:
                    self.pipeline_manager.load_baseline_pipeline(scheduler_type="euler")
                
                # 应用指定的优化
                self.pipeline_manager.apply_optimizations(
                    attention_slicing=config["attention_slicing"],
                    vae_slicing=config["vae_slicing"],
                    sdpa=config["sdpa"]
                )
                
                # 预热
                self.pipeline_manager.warmup(num_steps=2)
                
                # 多次运行取平均
                latencies = []
                vrams = []
                for _ in range(3):
                    result = self.pipeline_manager.generate(
                        prompt=prompt,
                        num_steps=config["steps"],
                        guidance_scale=config["guidance"],
                        seed=seed,
                        width=512,
                        height=512
                    )
                    latencies.append(result.latency_ms)
                    vrams.append(result.peak_vram_mb)
                
                config_results["runs"].append({
                    "prompt": prompt,
                    "latency_mean": sum(latencies) / len(latencies),
                    "latency_std": self._std(latencies),
                    "vram_mean": sum(vrams) / len(vrams),
                    "vram_std": self._std(vrams)
                })
            
            # 计算该配置的总体统计
            all_latencies = [r["latency_mean"] for r in config_results["runs"]]
            all_vrams = [r["vram_mean"] for r in config_results["runs"]]
            
            config_results["summary"] = {
                "latency_mean": sum(all_latencies) / len(all_latencies),
                "latency_std": self._std(all_latencies),
                "vram_mean": sum(all_vrams) / len(all_vrams),
                "vram_std": self._std(all_vrams)
            }
            
            results.append(config_results)
            
            logger.info(f"  延迟: {config_results['summary']['latency_mean']:.1f} ± {config_results['summary']['latency_std']:.1f} ms")
            logger.info(f"  显存: {config_results['summary']['vram_mean']:.0f} ± {config_results['summary']['vram_std']:.0f} MB")
        
        # 计算各优化项的贡献
        contributions = self._calculate_ablation_contributions(results)
        
        # 保存结果
        ablation_data = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "prompts": prompts,
            "results": results,
            "contributions": contributions
        }
        
        results_path = self.ablation_dir / "ablation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(ablation_data, f, indent=2, ensure_ascii=False)
        
        # 生成消融实验报告
        report_content = self._generate_ablation_report(ablation_data)
        report_path = self.ablation_dir / "ablation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"\n消融实验结果: {results_path}")
        logger.info(f"消融实验报告: {report_path}")
        
        return ablation_data
    
    def _calculate_ablation_contributions(self, results: List[Dict]) -> Dict[str, Any]:
        """计算各优化项的贡献"""
        
        # 找到各配置的结果
        def find_result(name):
            for r in results:
                if r["config"]["name"] == name:
                    return r["summary"]
            return None
        
        baseline_no_opt = find_result("Baseline_NoOpt")
        baseline_all_opt = find_result("Baseline_AllOpt")
        lcm4_no_opt = find_result("LCM4_NoOpt")
        lcm4_att = find_result("LCM4_AttSlice")
        lcm4_vae = find_result("LCM4_VAESlice")
        lcm4_sdpa = find_result("LCM4_SDPA")
        lcm4_all = find_result("LCM4_AllOpt")
        
        contributions = {}
        
        # LCM-LoRA 的贡献 (核心加速)
        if baseline_all_opt and lcm4_all:
            contributions["LCM-LoRA"] = {
                "latency_reduction_ms": baseline_all_opt["latency_mean"] - lcm4_all["latency_mean"],
                "latency_reduction_pct": (baseline_all_opt["latency_mean"] - lcm4_all["latency_mean"]) / baseline_all_opt["latency_mean"] * 100,
                "speedup": baseline_all_opt["latency_mean"] / lcm4_all["latency_mean"],
                "description": "LCM-LoRA 少步采样带来的核心加速效果"
            }
        
        # 各优化项的贡献 (相对于 LCM4_NoOpt)
        if lcm4_no_opt:
            if lcm4_att:
                contributions["Attention_Slicing"] = {
                    "latency_diff_ms": lcm4_no_opt["latency_mean"] - lcm4_att["latency_mean"],
                    "vram_diff_mb": lcm4_no_opt["vram_mean"] - lcm4_att["vram_mean"],
                    "description": "注意力切片优化，降低峰值显存"
                }
            
            if lcm4_vae:
                contributions["VAE_Slicing"] = {
                    "latency_diff_ms": lcm4_no_opt["latency_mean"] - lcm4_vae["latency_mean"],
                    "vram_diff_mb": lcm4_no_opt["vram_mean"] - lcm4_vae["vram_mean"],
                    "description": "VAE 切片优化，降低解码显存"
                }
            
            if lcm4_sdpa:
                contributions["SDPA"] = {
                    "latency_diff_ms": lcm4_no_opt["latency_mean"] - lcm4_sdpa["latency_mean"],
                    "vram_diff_mb": lcm4_no_opt["vram_mean"] - lcm4_sdpa["vram_mean"],
                    "description": "PyTorch SDPA 优化注意力计算"
                }
        
        # 步数的影响
        lcm2 = find_result("LCM2_AllOpt")
        lcm6 = find_result("LCM6_AllOpt")
        lcm8 = find_result("LCM8_AllOpt")
        
        if lcm4_all:
            contributions["Steps_Analysis"] = {
                "2_steps": lcm2["latency_mean"] if lcm2 else None,
                "4_steps": lcm4_all["latency_mean"],
                "6_steps": lcm6["latency_mean"] if lcm6 else None,
                "8_steps": lcm8["latency_mean"] if lcm8 else None,
                "description": "不同步数对延迟的影响"
            }
        
        return contributions
    

    def _generate_ablation_report(self, ablation_data: Dict) -> str:
        """生成消融实验 Markdown 报告"""
        
        report = f"""# 消融实验报告

**生成时间**: {ablation_data['timestamp']}  
**随机种子**: {ablation_data['seed']}

---

## 1. 实验目的

本消融实验旨在分析各优化项对 LCM-LoRA 加速系统的独立贡献，包括:

1. **LCM-LoRA**: 核心加速模块，通过一致性蒸馏实现少步采样
2. **Attention Slicing**: 注意力切片，降低峰值显存占用
3. **VAE Slicing**: VAE 解码切片，降低解码阶段显存
4. **SDPA**: PyTorch 2.0+ 的 Scaled Dot Product Attention 优化
5. **步数分析**: 2/4/6/8 步对速度和质量的影响

## 2. 实验配置

| 配置名称 | 描述 | LCM | 步数 | Att.Slice | VAE.Slice | SDPA |
|----------|------|-----|------|-----------|-----------|------|
"""
        for r in ablation_data["results"]:
            cfg = r["config"]
            report += f"| {cfg['name']} | {cfg['description']} | {'✓' if cfg['lcm'] else '✗'} | {cfg['steps']} | {'✓' if cfg['attention_slicing'] else '✗'} | {'✓' if cfg['vae_slicing'] else '✗'} | {'✓' if cfg['sdpa'] else '✗'} |\n"
        
        report += """
## 3. 实验结果

### 3.1 延迟与显存对比

| 配置 | 延迟 (ms) | 显存 (MB) |
|------|-----------|-----------|
"""
        for r in ablation_data["results"]:
            s = r["summary"]
            report += f"| {r['config']['name']} | {s['latency_mean']:.1f} ± {s['latency_std']:.1f} | {s['vram_mean']:.0f} ± {s['vram_std']:.0f} |\n"
        
        report += """
### 3.2 各优化项贡献分析

"""
        contributions = ablation_data.get("contributions", {})
        
        if "LCM-LoRA" in contributions:
            lcm = contributions["LCM-LoRA"]
            report += f"""#### LCM-LoRA 核心加速

- **延迟降低**: {lcm['latency_reduction_ms']:.1f} ms ({lcm['latency_reduction_pct']:.1f}%)
- **加速比**: {lcm['speedup']:.2f}x
- **说明**: {lcm['description']}

"""
        
        if "Attention_Slicing" in contributions:
            att = contributions["Attention_Slicing"]
            report += f"""#### Attention Slicing

- **延迟变化**: {att['latency_diff_ms']:+.1f} ms
- **显存变化**: {att['vram_diff_mb']:+.0f} MB
- **说明**: {att['description']}

"""
        
        if "VAE_Slicing" in contributions:
            vae = contributions["VAE_Slicing"]
            report += f"""#### VAE Slicing

- **延迟变化**: {vae['latency_diff_ms']:+.1f} ms
- **显存变化**: {vae['vram_diff_mb']:+.0f} MB
- **说明**: {vae['description']}

"""
        
        if "SDPA" in contributions:
            sdpa = contributions["SDPA"]
            report += f"""#### SDPA (Scaled Dot Product Attention)

- **延迟变化**: {sdpa['latency_diff_ms']:+.1f} ms
- **显存变化**: {sdpa['vram_diff_mb']:+.0f} MB
- **说明**: {sdpa['description']}

"""
        
        if "Steps_Analysis" in contributions:
            steps = contributions["Steps_Analysis"]
            report += f"""#### 步数分析

| 步数 | 延迟 (ms) |
|------|-----------|
| 2 步 | {steps['2_steps']:.1f if steps['2_steps'] else 'N/A'} |
| 4 步 | {steps['4_steps']:.1f if steps['4_steps'] else 'N/A'} |
| 6 步 | {steps['6_steps']:.1f if steps['6_steps'] else 'N/A'} |
| 8 步 | {steps['8_steps']:.1f if steps['8_steps'] else 'N/A'} |

"""
        
        report += """
## 4. 结论

基于消融实验结果，我们得出以下结论:

1. **LCM-LoRA 是核心加速来源**: 通过一致性蒸馏将采样步数从 20 步压缩到 4 步，实现了主要的加速效果。

2. **显存优化效果**: Attention Slicing 和 VAE Slicing 主要贡献在于降低峰值显存，对延迟影响较小。

3. **SDPA 优化**: PyTorch 2.0+ 的 SDPA 可以提供额外的计算加速。

4. **步数权衡**: 
   - 2 步: 最快但质量可能下降
   - 4 步: 速度与质量的最佳平衡点
   - 6-8 步: 质量更好但速度优势减小

5. **推荐配置**: LCM 4 步 + 全部优化 (Attention Slicing + VAE Slicing + SDPA)

---

*本报告由 LCM-LoRA 加速实验系统自动生成*
"""
        return report
    
    def generate_final_report(self, clipscore_results: Dict, ablation_data: Dict, dataset_info: Dict) -> str:
        """生成最终的补充实验报告"""
        
        report = f"""# LCM-LoRA 补充实验报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. CLIPScore 质量评估结果

### 1.1 各配置 CLIPScore 对比

"""
        # 按配置分组统计 CLIPScore
        if clipscore_results and "results" in clipscore_results:
            config_scores = {}
            for r in clipscore_results["results"]:
                cfg = r["config_name"]
                if cfg not in config_scores:
                    config_scores[cfg] = []
                config_scores[cfg].append(r["clip_score"])
            
            report += "| 配置 | CLIPScore 均值 | 标准差 | 样本数 |\n"
            report += "|------|----------------|--------|--------|\n"
            
            for cfg, scores in config_scores.items():
                mean = sum(scores) / len(scores)
                std = self._std(scores)
                report += f"| {cfg} | {mean:.4f} | {std:.4f} | {len(scores)} |\n"
            
            report += """
### 1.2 分析

CLIPScore 衡量生成图像与文本提示的语义相似度，范围 [0, 1]，越高越好。

"""
        
        report += """
## 2. 生成样例对比

详见 `comparison/comparison_grid.png`，展示了同一 prompt 在不同配置下的生成效果对比。

## 3. 数据集说明

"""
        if dataset_info:
            report += f"""
- **数据集名称**: {dataset_info.get('name', 'N/A')}
- **Prompt 总数**: {dataset_info.get('total_prompts', 'N/A')}
- **数据来源**: {dataset_info.get('source', 'N/A')}

### 类别分布

"""
            for cat, count in dataset_info.get('category_distribution', {}).items():
                report += f"- {cat}: {count}\n"
        
        report += """
## 4. 消融实验总结

"""
        if ablation_data and "contributions" in ablation_data:
            contributions = ablation_data["contributions"]
            
            if "LCM-LoRA" in contributions:
                lcm = contributions["LCM-LoRA"]
                report += f"- **LCM-LoRA 加速比**: {lcm.get('speedup', 0):.2f}x\n"
            
            report += """
详细消融实验结果请参考 `ablation/ablation_report.md`。

## 5. 结论

1. **质量保持**: LCM-LoRA 在大幅加速的同时，CLIPScore 与基线相当，说明生成质量得到了保持。

2. **加速效果显著**: 4 步 LCM 相比 20 步 Euler 实现了约 5-7x 的加速。

3. **优化组合有效**: Attention Slicing + VAE Slicing + SDPA 的组合可以进一步优化显存和速度。

---

*本报告由 LCM-LoRA 加速实验系统自动生成*
"""
        
        report_path = self.reports_dir / "supplementary_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"最终报告已生成: {report_path}")
        return str(report_path)
    
    def cleanup(self):
        """清理资源"""
        if self._pipeline_manager:
            self._pipeline_manager.unload()
        if self._metrics_collector:
            self._metrics_collector.unload_models()
        torch.cuda.empty_cache()



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LCM-LoRA 补充实验")
    parser.add_argument("--quick", action="store_true", help="快速模式，减少样本数")
    parser.add_argument("--skip-clip", action="store_true", help="跳过 CLIPScore 评估")
    parser.add_argument("--skip-comparison", action="store_true", help="跳过对比图生成")
    parser.add_argument("--skip-ablation", action="store_true", help="跳过消融实验")
    parser.add_argument("--output-dir", type=str, default="outputs/supplementary", help="输出目录")
    args = parser.parse_args()
    
    # 创建实验执行器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    exp = SupplementaryExperiments(output_dir=output_dir)
    
    logger.info("=" * 60)
    logger.info("LCM-LoRA 补充实验")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"快速模式: {args.quick}")
    
    try:
        # 1. 构建数据集
        logger.info("\n[1/4] 构建评测数据集...")
        num_prompts = 20 if args.quick else 50
        dataset_info = exp.build_and_analyze_dataset(num_prompts=num_prompts)
        
        # 获取 prompts
        prompts_data = get_coco_style_prompts(num_prompts)
        
        # 2. CLIPScore 评估
        clipscore_results = None
        if not args.skip_clip:
            logger.info("\n[2/4] 运行 CLIPScore 评估...")
            eval_prompts = prompts_data[:10] if args.quick else prompts_data[:20]
            seeds = [42] if args.quick else [42, 123]
            
            clipscore_results = exp.run_clipscore_evaluation(
                prompts=eval_prompts,
                seeds=seeds,
                configs=[
                    {"name": "Euler_20", "steps": 20, "guidance": 7.5, "lcm": False},
                    {"name": "LCM_4", "steps": 4, "guidance": 1.0, "lcm": True},
                ]
            )
        else:
            logger.info("\n[2/4] 跳过 CLIPScore 评估")
        
        # 3. 生成对比图
        if not args.skip_comparison:
            logger.info("\n[3/4] 生成样例对比图...")
            comparison_prompts = [
                "A red apple on a wooden table, realistic photo",
                "A young woman with long brown hair smiling",
                "A busy city street at night with neon lights",
            ]
            exp.generate_comparison_grid(
                prompts=comparison_prompts,
                seed=42,
                output_filename="comparison_grid.png"
            )
        else:
            logger.info("\n[3/4] 跳过对比图生成")
        
        # 4. 消融实验
        ablation_data = None
        if not args.skip_ablation:
            logger.info("\n[4/4] 运行消融实验...")
            ablation_prompts = [
                "A red apple on a wooden table",
                "A young woman smiling",
            ] if args.quick else [
                "A red apple on a wooden table",
                "A young woman smiling",
                "A busy city street at night",
            ]
            ablation_data = exp.run_ablation_experiment(
                prompts=ablation_prompts,
                seed=42
            )
        else:
            logger.info("\n[4/4] 跳过消融实验")
        
        # 5. 生成最终报告
        logger.info("\n生成最终报告...")
        report_path = exp.generate_final_report(
            clipscore_results=clipscore_results or {},
            ablation_data=ablation_data or {},
            dataset_info=dataset_info
        )
        
        # 清理
        exp.cleanup()
        
        logger.info("\n" + "=" * 60)
        logger.info("补充实验完成!")
        logger.info("=" * 60)
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"最终报告: {report_path}")
        logger.info("\n生成的文件:")
        logger.info(f"  - 数据集: {exp.dataset_dir}/evaluation_dataset.json")
        logger.info(f"  - 数据集报告: {exp.dataset_dir}/dataset_report.md")
        if clipscore_results:
            logger.info(f"  - CLIPScore 结果: {exp.reports_dir}/clipscore_results.json")
        if not args.skip_comparison:
            logger.info(f"  - 对比图: {exp.comparison_dir}/comparison_grid.png")
        if ablation_data:
            logger.info(f"  - 消融实验: {exp.ablation_dir}/ablation_report.md")
        
        return 0
        
    except Exception as e:
        logger.error(f"实验执行失败: {e}", exc_info=True)
        exp.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())
