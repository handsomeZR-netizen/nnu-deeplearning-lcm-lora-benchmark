#!/usr/bin/env python
"""
简化版补充实验脚本 - 分步执行，避免显存问题

Usage:
    python run_simple_supplement.py --step 1  # 只生成对比图
    python run_simple_supplement.py --step 2  # 只计算 CLIPScore
    python run_simple_supplement.py --step 3  # 只运行消融实验
    python run_simple_supplement.py --all     # 运行全部
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 输出目录
OUTPUT_DIR = Path("outputs/supplement_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_gpu_memory():
    """清理 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_pipeline_manager():
    """获取 PipelineManager 实例"""
    from src.core.pipeline import PipelineManager
    return PipelineManager(
        model_dir="models/dreamshaper-7",
        lcm_lora_dir="models/lcm-lora-sdv1-5",
        device="cuda"
    )


# ============================================================
# Step 1: 生成对比图
# ============================================================
def run_comparison_generation():
    """生成 Euler vs DPM vs LCM 对比图"""
    logger.info("=" * 60)
    logger.info("Step 1: 生成样例对比图")
    logger.info("=" * 60)
    
    comparison_dir = OUTPUT_DIR / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    prompts = [
        "A red apple on a wooden table, realistic photo",
        "A young woman with long brown hair smiling, portrait",
        "A busy city street at night with neon lights",
    ]
    
    configs = [
        {"name": "Euler_20", "steps": 20, "guidance": 7.5, "lcm": False, "scheduler": "euler"},
        {"name": "DPM_20", "steps": 20, "guidance": 7.5, "lcm": False, "scheduler": "dpm_solver"},
        {"name": "LCM_4", "steps": 4, "guidance": 1.0, "lcm": True, "scheduler": "lcm"},
    ]
    
    seed = 42
    all_images = []  # [(prompt, config_name, image, latency), ...]
    
    manager = get_pipeline_manager()
    
    for prompt in prompts:
        logger.info(f"\nPrompt: {prompt[:50]}...")
        row_data = {"prompt": prompt, "images": []}
        
        for config in configs:
            logger.info(f"  配置: {config['name']}")
            
            # 清理显存
            manager.unload()
            clear_gpu_memory()
            
            # 加载管线
            if config["lcm"]:
                manager.load_lcm_pipeline(fuse_lora=True)
            else:
                manager.load_baseline_pipeline(scheduler_type=config["scheduler"])
            
            manager.apply_optimizations(attention_slicing=True, vae_slicing=True, sdpa=True)
            manager.warmup(num_steps=2)
            
            # 生成
            result = manager.generate(
                prompt=prompt,
                num_steps=config["steps"],
                guidance_scale=config["guidance"],
                seed=seed,
                width=512,
                height=512
            )
            
            # 保存单张图
            safe_prompt = prompt[:20].replace(" ", "_").replace(",", "")
            img_path = comparison_dir / f"{config['name']}_{safe_prompt}_{seed}.png"
            result.image.save(img_path)
            
            row_data["images"].append({
                "config": config["name"],
                "image": result.image,
                "latency": result.latency_ms,
                "steps": config["steps"],
                "path": str(img_path)
            })
            
            logger.info(f"    延迟: {result.latency_ms:.1f}ms, 保存: {img_path.name}")
        
        all_images.append(row_data)
    
    # 创建对比网格图
    grid_path = create_comparison_grid(all_images, comparison_dir / "comparison_grid.png")
    
    # 清理
    manager.unload()
    clear_gpu_memory()
    
    logger.info(f"\n对比图已保存: {grid_path}")
    return str(grid_path)


def create_comparison_grid(images_data: List[Dict], output_path: Path) -> str:
    """创建对比网格图"""
    img_size = 512
    padding = 10
    header_height = 50
    label_height = 70
    
    num_rows = len(images_data)
    num_cols = len(images_data[0]["images"])
    
    total_width = num_cols * img_size + (num_cols + 1) * padding
    total_height = header_height + num_rows * (img_size + label_height) + (num_rows + 1) * padding
    
    grid = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 22)
        font_medium = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # 列标题
    config_names = [img["config"] for img in images_data[0]["images"]]
    for col, name in enumerate(config_names):
        x = padding + col * (img_size + padding) + img_size // 2
        draw.text((x, 20), name, fill=(0, 0, 0), font=font_large, anchor="mm")
    
    # 每行
    for row, row_data in enumerate(images_data):
        y_base = header_height + row * (img_size + label_height + padding) + padding
        
        for col, img_data in enumerate(row_data["images"]):
            x = padding + col * (img_size + padding)
            
            img = img_data["image"].resize((img_size, img_size), Image.Resampling.LANCZOS)
            grid.paste(img, (x, y_base))
            
            # 延迟标签
            label = f"{img_data['latency']:.0f}ms ({img_data['steps']}步)"
            draw.text((x + img_size // 2, y_base + img_size + 8), label, 
                     fill=(0, 128, 0), font=font_medium, anchor="mt")
        
        # Prompt
        prompt_short = row_data["prompt"][:55] + "..." if len(row_data["prompt"]) > 55 else row_data["prompt"]
        draw.text((padding, y_base + img_size + 30), f"Prompt: {prompt_short}",
                 fill=(100, 100, 100), font=font_small)
    
    grid.save(output_path, quality=95)
    return str(output_path)


# ============================================================
# Step 2: CLIPScore 评估
# ============================================================
def run_clipscore_evaluation():
    """计算 CLIPScore"""
    logger.info("=" * 60)
    logger.info("Step 2: CLIPScore 评估")
    logger.info("=" * 60)
    
    clipscore_dir = OUTPUT_DIR / "clipscore"
    clipscore_dir.mkdir(exist_ok=True)
    
    prompts = [
        "A red apple on a wooden table, realistic photo",
        "A white ceramic cup filled with hot coffee",
        "A young woman with long brown hair smiling",
        "A busy city street at night with neon lights",
        "A golden retriever playing in the park",
    ]
    
    configs = [
        {"name": "Euler_20", "steps": 20, "guidance": 7.5, "lcm": False, "scheduler": "euler"},
        {"name": "LCM_4", "steps": 4, "guidance": 1.0, "lcm": True, "scheduler": "lcm"},
    ]
    
    seed = 42
    results = []
    
    manager = get_pipeline_manager()
    
    # 先生成所有图像
    logger.info("\n生成图像...")
    generated_images = {}  # {(config_name, prompt): (image, latency)}
    
    for config in configs:
        logger.info(f"\n配置: {config['name']}")
        
        manager.unload()
        clear_gpu_memory()
        
        if config["lcm"]:
            manager.load_lcm_pipeline(fuse_lora=True)
        else:
            manager.load_baseline_pipeline(scheduler_type=config["scheduler"])
        
        manager.apply_optimizations(attention_slicing=True, vae_slicing=True, sdpa=True)
        manager.warmup(num_steps=2)
        
        for prompt in prompts:
            result = manager.generate(
                prompt=prompt,
                num_steps=config["steps"],
                guidance_scale=config["guidance"],
                seed=seed,
                width=512,
                height=512
            )
            generated_images[(config["name"], prompt)] = (result.image, result.latency_ms)
            logger.info(f"  {prompt[:30]}... -> {result.latency_ms:.1f}ms")
    
    # 卸载推理管线
    manager.unload()
    clear_gpu_memory()
    
    # 加载 CLIP 模型计算分数
    logger.info("\n计算 CLIPScore...")
    from src.metrics.collector import MetricsCollector
    metrics = MetricsCollector(device="cuda")
    
    for (config_name, prompt), (image, latency) in generated_images.items():
        clip_score = metrics.compute_clip_score(image, prompt)
        results.append({
            "config": config_name,
            "prompt": prompt,
            "clip_score": clip_score,
            "latency_ms": latency
        })
        logger.info(f"  [{config_name}] {prompt[:30]}... -> CLIPScore: {clip_score:.4f}")
    
    metrics.unload_models()
    clear_gpu_memory()
    
    # 统计
    logger.info("\n" + "=" * 40)
    logger.info("CLIPScore 统计")
    logger.info("=" * 40)
    
    for config in configs:
        config_results = [r for r in results if r["config"] == config["name"]]
        scores = [r["clip_score"] for r in config_results]
        latencies = [r["latency_ms"] for r in config_results]
        
        mean_score = sum(scores) / len(scores)
        mean_latency = sum(latencies) / len(latencies)
        
        logger.info(f"{config['name']}:")
        logger.info(f"  CLIPScore: {mean_score:.4f}")
        logger.info(f"  延迟: {mean_latency:.1f}ms")
    
    # 保存结果
    results_path = clipscore_dir / "clipscore_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存: {results_path}")
    return results


# ============================================================
# Step 3: 消融实验
# ============================================================
def run_ablation_experiment():
    """运行消融实验"""
    logger.info("=" * 60)
    logger.info("Step 3: 消融实验")
    logger.info("=" * 60)
    
    ablation_dir = OUTPUT_DIR / "ablation"
    ablation_dir.mkdir(exist_ok=True)
    
    prompt = "A red apple on a wooden table, realistic photo"
    seed = 42
    num_runs = 3
    
    ablation_configs = [
        {"name": "Baseline_Euler20", "lcm": False, "steps": 20, "guidance": 7.5,
         "att_slice": True, "vae_slice": True, "sdpa": True},
        {"name": "LCM4_NoOpt", "lcm": True, "steps": 4, "guidance": 1.0,
         "att_slice": False, "vae_slice": False, "sdpa": False},
        {"name": "LCM4_AttSlice", "lcm": True, "steps": 4, "guidance": 1.0,
         "att_slice": True, "vae_slice": False, "sdpa": False},
        {"name": "LCM4_VAESlice", "lcm": True, "steps": 4, "guidance": 1.0,
         "att_slice": False, "vae_slice": True, "sdpa": False},
        {"name": "LCM4_SDPA", "lcm": True, "steps": 4, "guidance": 1.0,
         "att_slice": False, "vae_slice": False, "sdpa": True},
        {"name": "LCM4_AllOpt", "lcm": True, "steps": 4, "guidance": 1.0,
         "att_slice": True, "vae_slice": True, "sdpa": True},
        {"name": "LCM2_AllOpt", "lcm": True, "steps": 2, "guidance": 1.0,
         "att_slice": True, "vae_slice": True, "sdpa": True},
        {"name": "LCM6_AllOpt", "lcm": True, "steps": 6, "guidance": 1.0,
         "att_slice": True, "vae_slice": True, "sdpa": True},
        {"name": "LCM8_AllOpt", "lcm": True, "steps": 8, "guidance": 1.0,
         "att_slice": True, "vae_slice": True, "sdpa": True},
    ]
    
    results = []
    manager = get_pipeline_manager()
    
    for config in ablation_configs:
        logger.info(f"\n配置: {config['name']}")
        
        manager.unload()
        clear_gpu_memory()
        
        if config["lcm"]:
            manager.load_lcm_pipeline(fuse_lora=True)
        else:
            manager.load_baseline_pipeline(scheduler_type="euler")
        
        manager.apply_optimizations(
            attention_slicing=config["att_slice"],
            vae_slicing=config["vae_slice"],
            sdpa=config["sdpa"]
        )
        manager.warmup(num_steps=2)
        
        latencies = []
        vrams = []
        
        for i in range(num_runs):
            result = manager.generate(
                prompt=prompt,
                num_steps=config["steps"],
                guidance_scale=config["guidance"],
                seed=seed + i,
                width=512,
                height=512
            )
            latencies.append(result.latency_ms)
            vrams.append(result.peak_vram_mb)
        
        mean_latency = sum(latencies) / len(latencies)
        mean_vram = sum(vrams) / len(vrams)
        
        results.append({
            "config": config["name"],
            "steps": config["steps"],
            "lcm": config["lcm"],
            "optimizations": {
                "attention_slicing": config["att_slice"],
                "vae_slicing": config["vae_slice"],
                "sdpa": config["sdpa"]
            },
            "latency_mean_ms": mean_latency,
            "vram_mean_mb": mean_vram
        })
        
        logger.info(f"  延迟: {mean_latency:.1f}ms, 显存: {mean_vram:.0f}MB")
    
    manager.unload()
    clear_gpu_memory()
    
    # 保存结果
    results_path = ablation_dir / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成消融报告
    report = generate_ablation_report(results)
    report_path = ablation_dir / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"\n结果已保存: {results_path}")
    logger.info(f"报告已保存: {report_path}")
    return results


def generate_ablation_report(results: List[Dict]) -> str:
    """生成消融实验报告"""
    report = f"""# 消融实验报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 实验配置

| 配置 | LCM | 步数 | Att.Slice | VAE.Slice | SDPA |
|------|-----|------|-----------|-----------|------|
"""
    for r in results:
        opt = r["optimizations"]
        report += f"| {r['config']} | {'✓' if r['lcm'] else '✗'} | {r['steps']} | "
        report += f"{'✓' if opt['attention_slicing'] else '✗'} | "
        report += f"{'✓' if opt['vae_slicing'] else '✗'} | "
        report += f"{'✓' if opt['sdpa'] else '✗'} |\n"
    
    report += """
## 2. 实验结果

| 配置 | 延迟 (ms) | 显存 (MB) |
|------|-----------|-----------|
"""
    for r in results:
        report += f"| {r['config']} | {r['latency_mean_ms']:.1f} | {r['vram_mean_mb']:.0f} |\n"
    
    # 计算加速比
    baseline = next((r for r in results if r["config"] == "Baseline_Euler20"), None)
    lcm4_all = next((r for r in results if r["config"] == "LCM4_AllOpt"), None)
    
    if baseline and lcm4_all:
        speedup = baseline["latency_mean_ms"] / lcm4_all["latency_mean_ms"]
        report += f"""
## 3. 关键发现

1. **LCM-LoRA 加速比**: {speedup:.2f}x (Euler 20步 vs LCM 4步)
2. **基线延迟**: {baseline['latency_mean_ms']:.1f}ms
3. **LCM 4步延迟**: {lcm4_all['latency_mean_ms']:.1f}ms
"""
    
    report += """
## 4. 结论

- LCM-LoRA 是主要加速来源，通过一致性蒸馏将步数从 20 压缩到 4
- Attention Slicing 和 VAE Slicing 主要降低峰值显存
- SDPA 提供额外的计算优化
- 推荐配置: LCM 4步 + 全部优化

---
*本报告由 LCM-LoRA 加速实验系统自动生成*
"""
    return report


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="简化版补充实验")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="运行指定步骤")
    parser.add_argument("--all", action="store_true", help="运行全部步骤")
    args = parser.parse_args()
    
    if not args.step and not args.all:
        parser.print_help()
        print("\n请指定 --step 1/2/3 或 --all")
        return 1
    
    logger.info("=" * 60)
    logger.info("LCM-LoRA 补充实验 (简化版)")
    logger.info("=" * 60)
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    try:
        if args.all or args.step == 1:
            run_comparison_generation()
        
        if args.all or args.step == 2:
            run_clipscore_evaluation()
        
        if args.all or args.step == 3:
            run_ablation_experiment()
        
        logger.info("\n" + "=" * 60)
        logger.info("实验完成!")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"实验失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
