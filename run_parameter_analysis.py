"""
参数分析实验脚本
测试 guidance_scale 和分辨率对性能的影响
"""

import torch
import time
import json
import os
import gc
import logging
from datetime import datetime
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "models/dreamshaper-7"
LCM_LORA_PATH = "models/lcm-lora-sdv1-5"
OUTPUT_DIR = "outputs/parameter_analysis"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_vram():
    """获取显存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def clear_memory():
    """清理显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_lcm_pipeline():
    """加载 LCM 管线"""
    from diffusers import StableDiffusionPipeline, LCMScheduler
    
    logger.info("加载 LCM-LoRA 管线...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA_PATH)
    pipe.fuse_lora()
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    # 预热
    logger.info("预热管线...")
    with torch.no_grad():
        _ = pipe("test", num_inference_steps=2, guidance_scale=1.0, output_type="latent")
    
    return pipe


def run_guidance_scale_analysis(pipe, output_dir):
    """Guidance Scale 参数分析"""
    logger.info("\n" + "=" * 60)
    logger.info("Guidance Scale 参数分析")
    logger.info("=" * 60)
    
    prompt = "A beautiful landscape with mountains and lake, realistic photo"
    guidance_scales = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5]
    steps = 4
    seed = 42
    
    results = []
    images = []
    
    for cfg in guidance_scales:
        logger.info(f"\n测试 guidance_scale={cfg}")
        clear_memory()
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # 多次测试取平均
        latencies = []
        for i in range(3):
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = time.time()
            
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    width=512,
                    height=512,
                )
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        results.append({
            "guidance_scale": cfg,
            "latency_ms": avg_latency,
            "vram_mb": peak_vram,
        })
        
        # 保存图像
        img_path = os.path.join(output_dir, f"cfg_{cfg}.png")
        result.images[0].save(img_path)
        images.append((cfg, result.images[0]))
        
        logger.info(f"  延迟: {avg_latency:.1f}ms, 显存: {peak_vram:.0f}MB")
    
    # 保存结果
    with open(os.path.join(output_dir, "guidance_scale_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results, images


def run_resolution_analysis(pipe, output_dir):
    """分辨率参数分析"""
    logger.info("\n" + "=" * 60)
    logger.info("分辨率参数分析")
    logger.info("=" * 60)
    
    prompt = "A beautiful landscape with mountains and lake, realistic photo"
    resolutions = [(384, 384), (512, 512), (576, 576), (640, 640), (768, 768)]
    steps = 4
    cfg = 1.0
    seed = 42
    
    results = []
    
    for width, height in resolutions:
        logger.info(f"\n测试分辨率 {width}x{height}")
        clear_memory()
        
        # 多次测试取平均
        latencies = []
        for i in range(3):
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = time.time()
            
            try:
                with torch.no_grad():
                    result = pipe(
                        prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        width=width,
                        height=height,
                    )
                
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"  显存不足，跳过 {width}x{height}")
                    clear_memory()
                    break
                raise
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            results.append({
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height,
                "pixels": width * height,
                "latency_ms": avg_latency,
                "vram_mb": peak_vram,
            })
            
            # 保存图像
            img_path = os.path.join(output_dir, f"res_{width}x{height}.png")
            result.images[0].save(img_path)
            
            logger.info(f"  延迟: {avg_latency:.1f}ms, 显存: {peak_vram:.0f}MB")
    
    # 保存结果
    with open(os.path.join(output_dir, "resolution_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_analysis_report(cfg_results, res_results, output_dir):
    """生成参数分析报告"""
    
    report = f"""# 参数分析实验报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**实验配置**: LCM-LoRA 4步, DreamShaper-7

---

## 1. Guidance Scale 分析

### 1.1 实验设置
- 固定步数: 4
- 固定分辨率: 512x512
- 测试 CFG: 1.0, 1.5, 2.0, 3.0, 5.0, 7.5

### 1.2 结果

| Guidance Scale | 延迟 (ms) | 显存 (MB) |
|----------------|-----------|-----------|
"""
    
    for r in cfg_results:
        report += f"| {r['guidance_scale']} | {r['latency_ms']:.1f} | {r['vram_mb']:.0f} |\n"
    
    report += """
### 1.3 分析

- **LCM 推荐 CFG**: 1.0-2.0（一致性模型特性）
- **CFG 对延迟影响**: 较小（主要影响生成质量）
- **高 CFG 问题**: LCM 在高 CFG 下可能出现过饱和

---

## 2. 分辨率分析

### 2.1 实验设置
- 固定步数: 4
- 固定 CFG: 1.0
- 测试分辨率: 384x384, 512x512, 576x576, 640x640, 768x768

### 2.2 结果

| 分辨率 | 像素数 | 延迟 (ms) | 显存 (MB) | 相对 512 延迟 |
|--------|--------|-----------|-----------|---------------|
"""
    
    base_latency = None
    for r in res_results:
        if r['resolution'] == '512x512':
            base_latency = r['latency_ms']
            break
    
    for r in res_results:
        relative = r['latency_ms'] / base_latency if base_latency else 1.0
        report += f"| {r['resolution']} | {r['pixels']:,} | {r['latency_ms']:.1f} | {r['vram_mb']:.0f} | {relative:.2f}x |\n"
    
    report += """
### 2.3 分析

- **延迟与分辨率关系**: 近似与像素数成正比
- **显存增长**: 分辨率翻倍，显存约增加 2-3x
- **推荐分辨率**: 512x512（平衡质量与速度）

---

## 3. 结论

1. **Guidance Scale**: LCM 模式下推荐使用 1.0-2.0，对延迟影响小
2. **分辨率**: 是影响延迟和显存的主要因素
3. **优化建议**: 
   - 实时预览使用 384x384 或 512x512
   - 最终输出可使用 768x768

---

## 4. 生成样例

### CFG 对比 (512x512, 4步)
"""
    
    for r in cfg_results:
        report += f"- CFG {r['guidance_scale']}: `cfg_{r['guidance_scale']}.png`\n"
    
    report += """
### 分辨率对比 (CFG 1.0, 4步)
"""
    
    for r in res_results:
        report += f"- {r['resolution']}: `res_{r['resolution']}.png`\n"
    
    report += """
---
*本报告由参数分析实验脚本自动生成*
"""
    
    report_path = os.path.join(output_dir, "parameter_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"\n报告已保存: {report_path}")


def main():
    logger.info("=" * 60)
    logger.info("LCM-LoRA 参数分析实验")
    logger.info("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载管线
    pipe = load_lcm_pipeline()
    
    # 运行实验
    cfg_results, _ = run_guidance_scale_analysis(pipe, OUTPUT_DIR)
    res_results = run_resolution_analysis(pipe, OUTPUT_DIR)
    
    # 生成报告
    generate_analysis_report(cfg_results, res_results, OUTPUT_DIR)
    
    # 清理
    del pipe
    clear_memory()
    
    logger.info("\n" + "=" * 60)
    logger.info("参数分析实验完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
