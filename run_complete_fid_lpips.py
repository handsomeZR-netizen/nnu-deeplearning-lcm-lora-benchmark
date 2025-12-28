"""
完整的 FID/LPIPS 评估流程
自动完成: 生成图像 -> 下载/准备真实图像 -> 计算指标

使用方式:
python run_complete_fid_lpips.py --num_samples 100 --use_unsplash
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random

import torch
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))


# Unsplash 免费图像 API (无需 API key 的方式)
UNSPLASH_CATEGORIES = {
    "nature": "https://source.unsplash.com/512x512/?nature",
    "people": "https://source.unsplash.com/512x512/?people,portrait",
    "food": "https://source.unsplash.com/512x512/?food",
    "animals": "https://source.unsplash.com/512x512/?animals",
    "city": "https://source.unsplash.com/512x512/?city,street",
    "objects": "https://source.unsplash.com/512x512/?objects,still-life",
}


def download_unsplash_images(
    output_dir: str = "outputs/real_images",
    num_images: int = 200,
    seed: int = 42
) -> List[str]:
    """
    从 Unsplash 下载真实图像
    注意: Unsplash source API 每次返回随机图像
    """
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    
    categories = list(UNSPLASH_CATEGORIES.keys())
    images_per_category = num_images // len(categories)
    
    image_paths = []
    
    print(f"从 Unsplash 下载 {num_images} 张真实图像...")
    
    for category in categories:
        url_template = UNSPLASH_CATEGORIES[category]
        
        for i in tqdm(range(images_per_category), desc=f"下载 {category}"):
            img_path = os.path.join(output_dir, f"{category}_{i:03d}.jpg")
            
            if os.path.exists(img_path):
                image_paths.append(img_path)
                continue
            
            try:
                # 添加随机参数避免缓存
                url = f"{url_template}&sig={random.randint(1, 100000)}"
                urllib.request.urlretrieve(url, img_path)
                image_paths.append(img_path)
                time.sleep(0.5)  # 避免请求过快
            except Exception as e:
                print(f"下载失败: {e}")
    
    print(f"下载完成: {len(image_paths)} 张图像")
    return image_paths


def download_picsum_images(
    output_dir: str = "outputs/real_images",
    num_images: int = 200,
    seed: int = 42
) -> List[str]:
    """
    从 Lorem Picsum 下载真实图像 (更稳定的备选方案)
    https://picsum.photos/
    """
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    
    print(f"从 Lorem Picsum 下载 {num_images} 张真实图像...")
    
    for i in tqdm(range(num_images), desc="下载图像"):
        img_path = os.path.join(output_dir, f"picsum_{i:04d}.jpg")
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            continue
        
        try:
            # Picsum 提供随机图像，seed 参数可以获取特定图像
            url = f"https://picsum.photos/seed/{seed + i}/512/512"
            urllib.request.urlretrieve(url, img_path)
            image_paths.append(img_path)
            time.sleep(0.3)
        except Exception as e:
            print(f"下载失败 {i}: {e}")
    
    print(f"下载完成: {len(image_paths)} 张图像")
    return image_paths


def load_evaluation_prompts(
    dataset_path: str = "outputs/coco_eval_dataset/evaluation_dataset.json",
    num_samples: int = 100
) -> List[Dict]:
    """加载评测 prompts"""
    
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = data["samples"][:num_samples]
        print(f"从 {dataset_path} 加载了 {len(prompts)} 个 prompts")
        return prompts
    
    # 如果数据集不存在，使用默认 prompts
    print("评测数据集不存在，使用默认 prompts")
    default_prompts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A cup of coffee on a wooden table",
        "A mountain landscape with snow",
        "A person walking in the rain",
        "A colorful flower garden",
        "A modern city skyline at night",
        "A dog playing in the park",
        "A plate of delicious food",
        "A peaceful forest path",
    ]
    
    return [{"id": i, "prompt": p, "seed": 42 + i} for i, p in enumerate(default_prompts)]


def generate_evaluation_images(
    prompts: List[Dict],
    output_dir: str,
    use_lcm: bool = True,
    num_steps: int = 4,
    guidance_scale: float = 1.5,
    resolution: int = 512,
    model_dir: str = "models/dreamshaper-7",
    lcm_lora_dir: str = "models/lcm-lora-sdv1-5"
) -> List[str]:
    """生成评估用图像"""
    from src.core.pipeline import PipelineManager
    
    method = "lcm" if use_lcm else "baseline"
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # 检查是否已有生成的图像
    existing = list(Path(method_dir).glob("*.png"))
    if len(existing) >= len(prompts):
        print(f"已存在 {len(existing)} 张 {method} 图像，跳过生成")
        return [str(p) for p in sorted(existing)[:len(prompts)]]
    
    print(f"\n生成 {method.upper()} 图像 ({len(prompts)} 张)...")
    
    # 初始化 PipelineManager
    pipeline = PipelineManager(
        model_dir=model_dir,
        lcm_lora_dir=lcm_lora_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if use_lcm:
        pipeline.load_lcm_pipeline(fuse_lora=True)
    else:
        pipeline.load_baseline_pipeline(scheduler_type="euler")
    
    # 应用优化
    pipeline.apply_optimizations(
        attention_slicing=True,
        vae_slicing=True,
        sdpa=True
    )
    
    # 预热
    pipeline.warmup(num_steps=2)
    
    image_paths = []
    
    for item in tqdm(prompts, desc=f"生成 {method}"):
        img_path = os.path.join(method_dir, f"{item['id']:04d}.png")
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            continue
        
        result = pipeline.generate(
            prompt=item["prompt"],
            num_steps=num_steps if use_lcm else 20,
            guidance_scale=guidance_scale if use_lcm else 7.5,
            width=resolution,
            height=resolution,
            seed=item.get("seed", 42)
        )
        
        result.image.save(img_path)
        image_paths.append(img_path)
    
    pipeline.unload()
    torch.cuda.empty_cache()
    
    return image_paths


def compute_metrics(
    generated_dir: str,
    reference_dir: str,
    baseline_dir: str = None,
    device: str = "cuda"
) -> Dict:
    """计算 FID 和 LPIPS 指标"""
    from src.metrics.collector import MetricsCollector
    
    results = {}
    
    # 加载图像
    def load_images(directory: str, max_count: int = 200) -> List[Image.Image]:
        paths = sorted(Path(directory).glob("*.png")) + sorted(Path(directory).glob("*.jpg"))
        images = []
        for p in paths[:max_count]:
            try:
                images.append(Image.open(p).convert("RGB"))
            except:
                pass
        return images
    
    print("\n加载图像...")
    generated_images = load_images(generated_dir)
    reference_images = load_images(reference_dir)
    
    print(f"  生成图像: {len(generated_images)} 张")
    print(f"  真实图像: {len(reference_images)} 张")
    
    collector = MetricsCollector(device=device)
    
    # 计算 FID
    if len(generated_images) >= 50 and len(reference_images) >= 50:
        print("\n计算 FID...")
        try:
            fid = collector.compute_fid(generated_images, reference_images)
            results["fid"] = fid
            print(f"  FID: {fid:.4f}")
        except Exception as e:
            print(f"  FID 计算失败: {e}")
            results["fid"] = None
    else:
        print(f"\n图像数量不足，跳过 FID (需要 >= 50)")
        results["fid"] = None
    
    # 计算 LPIPS
    if baseline_dir and os.path.exists(baseline_dir):
        baseline_images = load_images(baseline_dir)
        print(f"  Baseline 图像: {len(baseline_images)} 张")
        
        if len(baseline_images) > 0:
            min_count = min(len(generated_images), len(baseline_images))
            print(f"\n计算 LPIPS ({min_count} 对)...")
            
            try:
                lpips_scores = collector.compute_lpips_batch(
                    generated_images[:min_count],
                    baseline_images[:min_count]
                )
                results["lpips_mean"] = sum(lpips_scores) / len(lpips_scores)
                results["lpips_std"] = (sum((x - results["lpips_mean"])**2 for x in lpips_scores) / len(lpips_scores)) ** 0.5
                results["lpips_min"] = min(lpips_scores)
                results["lpips_max"] = max(lpips_scores)
                print(f"  LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
            except Exception as e:
                print(f"  LPIPS 计算失败: {e}")
    
    collector.unload_models()
    torch.cuda.empty_cache()
    
    return results


def run_complete_evaluation(
    num_samples: int = 100,
    output_dir: str = "outputs/fid_lpips_evaluation",
    download_real: bool = True,
    generate_baseline: bool = True,
    device: str = "cuda",
    model_dir: str = "models/dreamshaper-7",
    lcm_lora_dir: str = "models/lcm-lora-sdv1-5"
):
    """运行完整评估流程"""
    
    print("=" * 70)
    print("FID/LPIPS 完整评估流程")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: 准备真实图像
    real_images_dir = os.path.join(output_dir, "real_images")
    if download_real:
        existing_real = list(Path(real_images_dir).glob("*.jpg")) if os.path.exists(real_images_dir) else []
        if len(existing_real) < num_samples:
            print("\n[Step 1] 下载真实图像...")
            download_picsum_images(real_images_dir, num_samples)
        else:
            print(f"\n[Step 1] 已有 {len(existing_real)} 张真实图像")
    
    # Step 2: 加载评测 prompts
    print("\n[Step 2] 加载评测 prompts...")
    prompts = load_evaluation_prompts(num_samples=num_samples)
    
    # Step 3: 生成 LCM 图像
    print("\n[Step 3] 生成 LCM 图像...")
    lcm_dir = os.path.join(output_dir, "generated", "lcm")
    lcm_paths = generate_evaluation_images(
        prompts, 
        os.path.join(output_dir, "generated"),
        use_lcm=True,
        num_steps=4,
        model_dir=model_dir,
        lcm_lora_dir=lcm_lora_dir
    )
    
    # Step 4: 生成 Baseline 图像 (可选)
    baseline_dir = None
    if generate_baseline:
        print("\n[Step 4] 生成 Baseline 图像...")
        baseline_dir = os.path.join(output_dir, "generated", "baseline")
        baseline_paths = generate_evaluation_images(
            prompts,
            os.path.join(output_dir, "generated"),
            use_lcm=False,
            num_steps=20,
            model_dir=model_dir,
            lcm_lora_dir=lcm_lora_dir
        )
    
    # Step 5: 计算指标
    print("\n[Step 5] 计算 FID/LPIPS 指标...")
    metrics = compute_metrics(
        generated_dir=lcm_dir,
        reference_dir=real_images_dir,
        baseline_dir=baseline_dir,
        device=device
    )
    
    # Step 6: 保存结果
    print("\n[Step 6] 保存结果...")
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_samples": num_samples,
            "lcm_steps": 4,
            "baseline_steps": 20,
        },
        "metrics": metrics
    }
    
    result_path = os.path.join(output_dir, "evaluation_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    report = generate_evaluation_report(results, output_dir)
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)
    print(f"结果文件: {result_path}")
    print(f"报告文件: {os.path.join(output_dir, 'evaluation_report.md')}")
    
    if metrics.get("fid") is not None:
        print(f"\nFID Score: {metrics['fid']:.4f}")
    if metrics.get("lpips_mean") is not None:
        print(f"LPIPS (LCM vs Baseline): {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
    
    return results


def generate_evaluation_report(results: Dict, output_dir: str) -> str:
    """生成评估报告"""
    metrics = results["metrics"]
    
    report = f"""# FID/LPIPS 指标评估报告

**评估时间**: {results['timestamp']}

## 1. 评估配置

| 配置项 | 值 |
|--------|-----|
| 样本数量 | {results['config']['num_samples']} |
| LCM 步数 | {results['config']['lcm_steps']} |
| Baseline 步数 | {results['config']['baseline_steps']} |

## 2. 评估结果

### 2.1 FID (Fréchet Inception Distance)

"""
    
    if metrics.get("fid") is not None:
        fid = metrics["fid"]
        quality = "优秀" if fid < 10 else "良好" if fid < 50 else "一般" if fid < 100 else "较差"
        report += f"""| 指标 | 值 | 评价 |
|------|-----|------|
| FID Score | {fid:.4f} | {quality} |

**解读**: FID 越低表示生成图像分布与真实图像分布越接近。
- < 10: 优秀
- 10-50: 良好  
- 50-100: 一般
- > 100: 较差

"""
    else:
        report += "**未计算** (图像数量不足或计算失败)\n\n"
    
    report += "### 2.2 LPIPS (Learned Perceptual Image Patch Similarity)\n\n"
    
    if metrics.get("lpips_mean") is not None:
        lpips = metrics["lpips_mean"]
        similarity = "非常相似" if lpips < 0.1 else "相似" if lpips < 0.3 else "有差异" if lpips < 0.5 else "差异较大"
        report += f"""| 指标 | 值 |
|------|-----|
| 平均 LPIPS | {metrics['lpips_mean']:.4f} |
| 标准差 | {metrics['lpips_std']:.4f} |
| 最小值 | {metrics['lpips_min']:.4f} |
| 最大值 | {metrics['lpips_max']:.4f} |

**解读**: LPIPS 衡量 LCM (4步) 与 Baseline (20步) 生成图像的感知差异。
- 当前结果: {similarity}
- < 0.1: 非常相似
- 0.1-0.3: 相似
- 0.3-0.5: 有差异
- > 0.5: 差异较大

"""
    else:
        report += "**未计算** (未生成 Baseline 图像)\n\n"
    
    report += """## 3. 结论

"""
    
    if metrics.get("fid") is not None and metrics.get("lpips_mean") is not None:
        report += f"""基于 FID 和 LPIPS 指标评估:

1. **生成质量 (FID={metrics['fid']:.2f})**: LCM 生成的图像与真实图像分布的距离
2. **加速代价 (LPIPS={metrics['lpips_mean']:.4f})**: 4步 LCM 与 20步 Baseline 的感知差异

LCM-LoRA 在大幅减少推理步数 (20→4) 的同时，保持了较好的生成质量。
"""
    
    report += """
---
*本报告由 FID/LPIPS 评估脚本自动生成*
"""
    
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整 FID/LPIPS 评估")
    parser.add_argument("--num_samples", type=int, default=100, help="样本数量")
    parser.add_argument("--output_dir", type=str, default="outputs/fid_lpips_evaluation", help="输出目录")
    parser.add_argument("--no_download", action="store_true", help="不下载真实图像")
    parser.add_argument("--no_baseline", action="store_true", help="不生成 Baseline")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--model_dir", type=str, default="models/dreamshaper-7", help="基础模型目录")
    parser.add_argument("--lcm_lora_dir", type=str, default="models/lcm-lora-sdv1-5", help="LCM-LoRA 目录")
    
    args = parser.parse_args()
    
    run_complete_evaluation(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        download_real=not args.no_download,
        generate_baseline=not args.no_baseline,
        device=args.device,
        model_dir=args.model_dir,
        lcm_lora_dir=args.lcm_lora_dir
    )
