"""
FID/LPIPS 指标评估脚本
需要真实图像分布来计算 FID，需要参考图像来计算 LPIPS

使用方式:
1. 准备真实图像: 从 COCO 数据集下载或使用本地图像
2. 生成图像: 使用不同方法生成图像
3. 计算指标: 运行本脚本

FID 计算: 生成图像 vs 真实图像分布
LPIPS 计算: LCM 生成图像 vs Baseline 生成图像 (同 prompt)
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random

import torch
from PIL import Image
from tqdm import tqdm

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.collector import MetricsCollector


def download_coco_images(
    output_dir: str = "outputs/coco_real_images",
    num_images: int = 200,
    seed: int = 42
) -> List[str]:
    """
    下载 COCO 真实图像用于 FID 计算
    
    由于 COCO 数据集较大，这里提供两种方案:
    1. 使用 COCO API 下载指定数量的图像
    2. 使用本地已有的真实图像
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 尝试使用 pycocotools
        from pycocotools.coco import COCO
        import urllib.request
        
        # COCO 2017 验证集 annotations
        ann_file = "annotations/captions_val2017.json"
        
        if not os.path.exists(ann_file):
            print("COCO annotations 文件不存在，请下载:")
            print("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            print("unzip annotations_trainval2017.zip")
            return []
        
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        
        random.seed(seed)
        selected_ids = random.sample(img_ids, min(num_images, len(img_ids)))
        
        image_paths = []
        for img_id in tqdm(selected_ids, desc="下载 COCO 图像"):
            img_info = coco.loadImgs(img_id)[0]
            img_url = img_info['coco_url']
            img_path = os.path.join(output_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                urllib.request.urlretrieve(img_url, img_path)
            
            image_paths.append(img_path)
        
        return image_paths
        
    except ImportError:
        print("pycocotools 未安装，使用备选方案")
        print("安装命令: pip install pycocotools")
        return []


def load_images_from_directory(
    directory: str,
    max_images: Optional[int] = None,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')
) -> List[Image.Image]:
    """从目录加载图像"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).glob(f"*{ext}"))
        image_paths.extend(Path(directory).glob(f"*{ext.upper()}"))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    images = []
    for path in tqdm(image_paths, desc=f"加载图像 from {directory}"):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"无法加载图像 {path}: {e}")
    
    return images


def generate_images_for_evaluation(
    prompts: List[Dict],
    output_dir: str,
    use_lcm: bool = True,
    num_steps: int = 4,
    guidance_scale: float = 1.5,
    resolution: int = 512
) -> List[str]:
    """
    生成图像用于评估
    
    Args:
        prompts: prompt 列表，每个包含 id, prompt, seed
        output_dir: 输出目录
        use_lcm: 是否使用 LCM-LoRA
        num_steps: 推理步数
        guidance_scale: CFG scale
        resolution: 分辨率
    
    Returns:
        生成图像的路径列表
    """
    from src.core.pipeline import LCMPipeline
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 pipeline
    pipeline = LCMPipeline()
    pipeline.load_model()
    
    if use_lcm:
        pipeline.load_lcm_lora()
    
    image_paths = []
    
    for item in tqdm(prompts, desc=f"生成图像 ({'LCM' if use_lcm else 'Baseline'})"):
        prompt = item["prompt"]
        seed = item.get("seed", 42)
        img_id = item["id"]
        
        # 生成图像
        result = pipeline.generate(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=resolution,
            height=resolution,
            seed=seed
        )
        
        # 保存图像
        method = "lcm" if use_lcm else "baseline"
        img_path = os.path.join(output_dir, f"{method}_{img_id:04d}.png")
        result.image.save(img_path)
        image_paths.append(img_path)
    
    pipeline.unload()
    return image_paths


def compute_fid_score(
    generated_images: List[Image.Image],
    reference_images: List[Image.Image],
    device: str = "cuda"
) -> float:
    """计算 FID 分数"""
    collector = MetricsCollector(device=device)
    
    try:
        fid = collector.compute_fid(generated_images, reference_images)
        return fid
    finally:
        collector.unload_models()


def compute_lpips_scores(
    images1: List[Image.Image],
    images2: List[Image.Image],
    device: str = "cuda"
) -> List[float]:
    """计算 LPIPS 分数列表"""
    collector = MetricsCollector(device=device)
    
    try:
        scores = collector.compute_lpips_batch(images1, images2)
        return scores
    finally:
        collector.unload_models()


def run_full_evaluation(
    generated_dir: str,
    reference_dir: str,
    baseline_dir: Optional[str] = None,
    output_dir: str = "outputs/fid_lpips_results",
    device: str = "cuda",
    max_images: int = 200
):
    """
    运行完整的 FID/LPIPS 评估
    
    Args:
        generated_dir: 生成图像目录 (LCM 生成)
        reference_dir: 真实图像目录 (用于 FID)
        baseline_dir: Baseline 生成图像目录 (用于 LPIPS，可选)
        output_dir: 结果输出目录
        device: 计算设备
        max_images: 最大图像数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("FID/LPIPS 指标评估")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "generated_dir": generated_dir,
            "reference_dir": reference_dir,
            "baseline_dir": baseline_dir,
            "max_images": max_images,
            "device": device
        },
        "metrics": {}
    }
    
    # 加载生成图像
    print(f"\n加载生成图像: {generated_dir}")
    generated_images = load_images_from_directory(generated_dir, max_images)
    print(f"  加载了 {len(generated_images)} 张生成图像")
    
    if len(generated_images) == 0:
        print("错误: 没有找到生成图像!")
        return
    
    # 加载真实图像并计算 FID
    print(f"\n加载真实图像: {reference_dir}")
    reference_images = load_images_from_directory(reference_dir, max_images)
    print(f"  加载了 {len(reference_images)} 张真实图像")
    
    if len(reference_images) >= 50:
        print("\n计算 FID...")
        fid_score = compute_fid_score(generated_images, reference_images, device)
        print(f"  FID Score: {fid_score:.4f}")
        results["metrics"]["fid"] = {
            "score": fid_score,
            "num_generated": len(generated_images),
            "num_reference": len(reference_images),
            "note": "FID 越低越好，表示生成图像分布与真实图像分布越接近"
        }
    else:
        print(f"  警告: 真实图像数量不足 ({len(reference_images)} < 50)，跳过 FID 计算")
        print("  FID 需要至少 50 张图像才能准确计算")
        results["metrics"]["fid"] = {
            "score": None,
            "error": f"真实图像数量不足: {len(reference_images)} < 50"
        }
    
    # 计算 LPIPS (如果提供了 baseline)
    if baseline_dir and os.path.exists(baseline_dir):
        print(f"\n加载 Baseline 图像: {baseline_dir}")
        baseline_images = load_images_from_directory(baseline_dir, max_images)
        print(f"  加载了 {len(baseline_images)} 张 Baseline 图像")
        
        if len(baseline_images) > 0:
            # 确保数量匹配
            min_count = min(len(generated_images), len(baseline_images))
            gen_subset = generated_images[:min_count]
            base_subset = baseline_images[:min_count]
            
            print(f"\n计算 LPIPS (对比 {min_count} 对图像)...")
            lpips_scores = compute_lpips_scores(gen_subset, base_subset, device)
            
            avg_lpips = sum(lpips_scores) / len(lpips_scores)
            min_lpips = min(lpips_scores)
            max_lpips = max(lpips_scores)
            
            print(f"  平均 LPIPS: {avg_lpips:.4f}")
            print(f"  最小 LPIPS: {min_lpips:.4f}")
            print(f"  最大 LPIPS: {max_lpips:.4f}")
            
            results["metrics"]["lpips"] = {
                "average": avg_lpips,
                "min": min_lpips,
                "max": max_lpips,
                "num_pairs": min_count,
                "all_scores": lpips_scores,
                "note": "LPIPS 越低表示两种方法生成的图像越相似"
            }
    else:
        print("\n未提供 Baseline 目录，跳过 LPIPS 计算")
        results["metrics"]["lpips"] = {
            "score": None,
            "note": "需要提供 baseline_dir 参数来计算 LPIPS"
        }
    
    # 保存结果
    result_path = os.path.join(output_dir, "fid_lpips_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")
    
    # 生成报告
    generate_report(results, output_dir)
    
    return results


def generate_report(results: Dict, output_dir: str):
    """生成评估报告"""
    report = f"""# FID/LPIPS 指标评估报告

**评估时间**: {results['timestamp']}

## 1. 评估配置

| 配置项 | 值 |
|--------|-----|
| 生成图像目录 | {results['config']['generated_dir']} |
| 真实图像目录 | {results['config']['reference_dir']} |
| Baseline 目录 | {results['config'].get('baseline_dir', 'N/A')} |
| 最大图像数 | {results['config']['max_images']} |

## 2. FID 评估结果

"""
    
    fid_metrics = results['metrics'].get('fid', {})
    if fid_metrics.get('score') is not None:
        report += f"""| 指标 | 值 |
|------|-----|
| FID Score | {fid_metrics['score']:.4f} |
| 生成图像数 | {fid_metrics['num_generated']} |
| 真实图像数 | {fid_metrics['num_reference']} |

**说明**: {fid_metrics.get('note', '')}

### FID 分数解读

| FID 范围 | 质量评价 |
|----------|----------|
| < 10 | 优秀 (接近真实图像分布) |
| 10-50 | 良好 |
| 50-100 | 一般 |
| > 100 | 较差 |

"""
    else:
        report += f"**未计算**: {fid_metrics.get('error', '未知原因')}\n\n"
    
    report += "## 3. LPIPS 评估结果\n\n"
    
    lpips_metrics = results['metrics'].get('lpips', {})
    if lpips_metrics.get('average') is not None:
        report += f"""| 指标 | 值 |
|------|-----|
| 平均 LPIPS | {lpips_metrics['average']:.4f} |
| 最小 LPIPS | {lpips_metrics['min']:.4f} |
| 最大 LPIPS | {lpips_metrics['max']:.4f} |
| 对比图像对数 | {lpips_metrics['num_pairs']} |

**说明**: {lpips_metrics.get('note', '')}

### LPIPS 分数解读

| LPIPS 范围 | 相似度评价 |
|------------|------------|
| < 0.1 | 非常相似 |
| 0.1-0.3 | 相似 |
| 0.3-0.5 | 有差异 |
| > 0.5 | 差异较大 |

"""
    else:
        report += f"**未计算**: {lpips_metrics.get('note', '未知原因')}\n\n"
    
    report += """## 4. 指标说明

### 4.1 FID (Fréchet Inception Distance)

FID 通过比较生成图像和真实图像在 Inception 网络特征空间中的分布距离来评估生成质量：
- 使用 Inception-v3 提取图像特征
- 计算两组特征的均值和协方差
- 计算 Fréchet 距离

**优点**: 能够同时评估生成质量和多样性
**缺点**: 需要大量图像 (建议 >= 50 张)

### 4.2 LPIPS (Learned Perceptual Image Patch Similarity)

LPIPS 使用深度网络特征计算两张图像的感知相似度：
- 使用预训练网络 (AlexNet/VGG) 提取多层特征
- 计算特征空间中的加权距离

**用途**: 评估加速方法 (LCM) 与原始方法 (Baseline) 生成图像的差异

---
*本报告由 FID/LPIPS 评估脚本自动生成*
"""
    
    report_path = os.path.join(output_dir, "fid_lpips_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {report_path}")


def prepare_real_images_guide():
    """打印准备真实图像的指南"""
    guide = """
================================================================================
                        准备真实图像指南 (用于 FID 计算)
================================================================================

FID 需要真实图像分布作为参考。以下是几种获取方式:

【方案 1】使用 COCO 数据集 (推荐)
---------------------------------
1. 下载 COCO 2017 验证集图像:
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip val2017.zip -d outputs/coco_real_images/

2. 或者只下载部分图像 (需要 pycocotools):
   pip install pycocotools
   然后运行本脚本的 --download_coco 选项

【方案 2】使用本地真实照片
--------------------------
1. 收集 200+ 张真实照片 (非 AI 生成)
2. 放入目录: outputs/real_images/
3. 建议涵盖多种场景: 人物、物体、风景等

【方案 3】使用其他数据集
------------------------
- ImageNet 验证集
- LAION 子集
- 任何真实图像数据集

【注意事项】
-----------
- FID 需要至少 50 张图像，建议 200+ 张
- 图像应该与生成图像的主题相关
- 分辨率不需要完全一致 (会自动 resize 到 299x299)

================================================================================
"""
    print(guide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID/LPIPS 指标评估")
    
    parser.add_argument("--generated_dir", type=str, 
                        default="outputs/generated_images",
                        help="生成图像目录")
    parser.add_argument("--reference_dir", type=str,
                        default="outputs/real_images",
                        help="真实图像目录 (用于 FID)")
    parser.add_argument("--baseline_dir", type=str,
                        default=None,
                        help="Baseline 生成图像目录 (用于 LPIPS)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/fid_lpips_results",
                        help="结果输出目录")
    parser.add_argument("--max_images", type=int, default=200,
                        help="最大图像数量")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--guide", action="store_true",
                        help="显示准备真实图像的指南")
    parser.add_argument("--download_coco", action="store_true",
                        help="下载 COCO 图像")
    
    args = parser.parse_args()
    
    if args.guide:
        prepare_real_images_guide()
    elif args.download_coco:
        download_coco_images(
            output_dir="outputs/coco_real_images",
            num_images=args.max_images
        )
    else:
        run_full_evaluation(
            generated_dir=args.generated_dir,
            reference_dir=args.reference_dir,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir,
            device=args.device,
            max_images=args.max_images
        )
