"""
LCM-LoRA 实时图像生成系统
Gradio 交互式演示界面
"""

import gradio as gr
import torch
import time
import os
import json
from datetime import datetime
from PIL import Image

# 全局变量
pipe_euler = None
pipe_lcm = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 配置
MODEL_PATH = "models/dreamshaper-7"
LCM_LORA_PATH = "models/lcm-lora-sdv1-5"
OUTPUT_DIR = "outputs/gradio_demo"


def get_vram_usage():
    """获取当前显存使用量"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0


def load_euler_pipeline():
    """加载 Euler 基线管线"""
    global pipe_euler
    
    if pipe_euler is not None:
        return pipe_euler
    
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    
    print("加载 Euler 基线管线...")
    pipe_euler = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe_euler.scheduler = EulerDiscreteScheduler.from_config(pipe_euler.scheduler.config)
    pipe_euler = pipe_euler.to(device)
    pipe_euler.enable_attention_slicing()
    
    # 预热
    with torch.no_grad():
        _ = pipe_euler("test", num_inference_steps=1, output_type="latent")
    
    print("Euler 管线加载完成")
    return pipe_euler


def load_lcm_pipeline():
    """加载 LCM-LoRA 加速管线"""
    global pipe_lcm
    
    if pipe_lcm is not None:
        return pipe_lcm
    
    from diffusers import StableDiffusionPipeline, LCMScheduler
    
    print("加载 LCM-LoRA 管线...")
    pipe_lcm = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe_lcm.scheduler = LCMScheduler.from_config(pipe_lcm.scheduler.config)
    pipe_lcm.load_lora_weights(LCM_LORA_PATH)
    pipe_lcm.fuse_lora()
    pipe_lcm = pipe_lcm.to(device)
    pipe_lcm.enable_attention_slicing()
    
    # 预热
    with torch.no_grad():
        _ = pipe_lcm("test", num_inference_steps=2, guidance_scale=1.0, output_type="latent")
    
    print("LCM 管线加载完成")
    return pipe_lcm


def generate_image(
    prompt: str,
    mode: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
):
    """生成图像"""
    
    if not prompt.strip():
        return None, "请输入提示词", "", ""
    
    # 设置随机种子
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 记录开始时间和显存
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_time = time.time()
    vram_before = get_vram_usage()
    
    try:
        if mode == "Euler (基线)":
            pipe = load_euler_pipeline()
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                )
        else:  # LCM-LoRA
            pipe = load_lcm_pipeline()
            # LCM 使用低 guidance scale
            lcm_guidance = min(guidance_scale, 2.0) if guidance_scale > 1.0 else 1.0
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=lcm_guidance,
                    generator=generator,
                    width=width,
                    height=height,
                )
        
        image = result.images[0]
        
    except Exception as e:
        return None, f"生成失败: {str(e)}", "", ""
    
    # 计算耗时和显存
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # ms
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # 保存图像
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{mode.split()[0]}_{steps}steps_{timestamp}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    
    # 生成状态信息
    status = f"✅ 生成成功！"
    
    metrics = f"""📊 性能指标:
• 延迟: {latency:.1f} ms
• 峰值显存: {peak_vram:.0f} MB
• 步数: {steps}
• 分辨率: {width}x{height}"""
    
    save_info = f"💾 已保存: {filepath}"
    
    return image, status, metrics, save_info


def compare_generate(
    prompt: str,
    euler_steps: int,
    lcm_steps: int,
    seed: int,
    width: int,
    height: int,
):
    """对比生成"""
    
    if not prompt.strip():
        return None, None, "请输入提示词", ""
    
    generator_euler = torch.Generator(device=device).manual_seed(seed)
    generator_lcm = torch.Generator(device=device).manual_seed(seed)
    
    results = {}
    
    # Euler 生成
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_euler = time.time()
    
    try:
        pipe_e = load_euler_pipeline()
        with torch.no_grad():
            result_euler = pipe_e(
                prompt,
                num_inference_steps=euler_steps,
                guidance_scale=7.5,
                generator=generator_euler,
                width=width,
                height=height,
            )
        euler_image = result_euler.images[0]
        euler_latency = (time.time() - start_euler) * 1000
        euler_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    except Exception as e:
        return None, None, f"Euler 生成失败: {str(e)}", ""
    
    # LCM 生成
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_lcm = time.time()
    
    try:
        pipe_l = load_lcm_pipeline()
        with torch.no_grad():
            result_lcm = pipe_l(
                prompt,
                num_inference_steps=lcm_steps,
                guidance_scale=1.0,
                generator=generator_lcm,
                width=width,
                height=height,
            )
        lcm_image = result_lcm.images[0]
        lcm_latency = (time.time() - start_lcm) * 1000
        lcm_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    except Exception as e:
        return euler_image, None, f"LCM 生成失败: {str(e)}", ""
    
    # 计算加速比
    speedup = euler_latency / lcm_latency if lcm_latency > 0 else 0
    
    comparison = f"""📊 对比结果:

| 指标 | Euler {euler_steps}步 | LCM {lcm_steps}步 | 改进 |
|------|----------|---------|------|
| 延迟 | {euler_latency:.0f} ms | {lcm_latency:.0f} ms | **{speedup:.1f}x 加速** |
| 显存 | {euler_vram:.0f} MB | {lcm_vram:.0f} MB | - |
"""
    
    # 保存对比图
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    euler_path = os.path.join(OUTPUT_DIR, f"compare_euler_{timestamp}.png")
    lcm_path = os.path.join(OUTPUT_DIR, f"compare_lcm_{timestamp}.png")
    euler_image.save(euler_path)
    lcm_image.save(lcm_path)
    
    save_info = f"💾 已保存对比图到 {OUTPUT_DIR}"
    
    return euler_image, lcm_image, comparison, save_info


def create_demo():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="LCM-LoRA 实时图像生成系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 LCM-LoRA 实时图像生成系统
        
        基于 Stable Diffusion + LCM-LoRA 的少步推理加速演示
        
        **实验环境**: RTX 4060 Laptop GPU | CUDA 12.8 | PyTorch 2.9.1
        """)
        
        with gr.Tabs():
            # Tab 1: 单图生成
            with gr.TabItem("🖼️ 单图生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_single = gr.Textbox(
                            label="提示词 (Prompt)",
                            placeholder="输入图像描述...",
                            value="A beautiful sunset over mountains, realistic photo",
                            lines=2
                        )
                        
                        mode = gr.Radio(
                            choices=["Euler (基线)", "LCM-LoRA (加速)"],
                            value="LCM-LoRA (加速)",
                            label="生成模式"
                        )
                        
                        with gr.Row():
                            steps_single = gr.Slider(
                                minimum=1, maximum=30, value=4, step=1,
                                label="推理步数"
                            )
                            guidance_single = gr.Slider(
                                minimum=1.0, maximum=15.0, value=1.0, step=0.5,
                                label="Guidance Scale"
                            )
                        
                        with gr.Row():
                            width_single = gr.Slider(
                                minimum=256, maximum=768, value=512, step=64,
                                label="宽度"
                            )
                            height_single = gr.Slider(
                                minimum=256, maximum=768, value=512, step=64,
                                label="高度"
                            )
                        
                        seed_single = gr.Number(value=42, label="随机种子", precision=0)
                        
                        btn_generate = gr.Button("🚀 生成图像", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="生成结果", type="pil")
                        status_single = gr.Textbox(label="状态", interactive=False)
                        metrics_single = gr.Textbox(label="性能指标", interactive=False, lines=5)
                        save_info_single = gr.Textbox(label="保存信息", interactive=False)
                
                btn_generate.click(
                    fn=generate_image,
                    inputs=[prompt_single, mode, steps_single, guidance_single, seed_single, width_single, height_single],
                    outputs=[output_image, status_single, metrics_single, save_info_single]
                )
            
            # Tab 2: 对比生成
            with gr.TabItem("⚖️ 对比生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_compare = gr.Textbox(
                            label="提示词 (Prompt)",
                            placeholder="输入图像描述...",
                            value="A golden retriever playing in the park, realistic photo",
                            lines=2
                        )
                        
                        with gr.Row():
                            euler_steps = gr.Slider(
                                minimum=10, maximum=30, value=20, step=1,
                                label="Euler 步数"
                            )
                            lcm_steps = gr.Slider(
                                minimum=2, maximum=8, value=4, step=1,
                                label="LCM 步数"
                            )
                        
                        with gr.Row():
                            width_compare = gr.Slider(
                                minimum=256, maximum=768, value=512, step=64,
                                label="宽度"
                            )
                            height_compare = gr.Slider(
                                minimum=256, maximum=768, value=512, step=64,
                                label="高度"
                            )
                        
                        seed_compare = gr.Number(value=42, label="随机种子", precision=0)
                        
                        btn_compare = gr.Button("⚡ 对比生成", variant="primary")
                
                with gr.Row():
                    euler_output = gr.Image(label="Euler 基线", type="pil")
                    lcm_output = gr.Image(label="LCM-LoRA 加速", type="pil")
                
                comparison_result = gr.Markdown(label="对比结果")
                save_info_compare = gr.Textbox(label="保存信息", interactive=False)
                
                btn_compare.click(
                    fn=compare_generate,
                    inputs=[prompt_compare, euler_steps, lcm_steps, seed_compare, width_compare, height_compare],
                    outputs=[euler_output, lcm_output, comparison_result, save_info_compare]
                )
            
            # Tab 3: 系统信息
            with gr.TabItem("ℹ️ 系统信息"):
                gr.Markdown(f"""
                ## 系统配置
                
                | 项目 | 值 |
                |------|-----|
                | GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
                | CUDA 版本 | 12.8 |
                | PyTorch 版本 | 2.9.1+cu128 |
                | 基础模型 | DreamShaper-7 |
                | 加速模块 | LCM-LoRA v1.5 |
                
                ## 方法说明
                
                ### Euler 基线
                - 传统扩散模型采样器
                - 需要 20-30 步才能获得高质量图像
                - Guidance Scale 通常设为 7.5
                
                ### LCM-LoRA 加速
                - 基于一致性蒸馏的少步采样
                - 仅需 2-8 步即可生成高质量图像
                - Guidance Scale 设为 1.0-2.0
                - 加速比可达 5-7x
                
                ## 推荐配置
                
                | 场景 | 模式 | 步数 | 预期延迟 |
                |------|------|------|----------|
                | 实时预览 | LCM | 2 | ~400ms |
                | 平衡质量 | LCM | 4 | ~550ms |
                | 高质量 | LCM | 6-8 | ~750-900ms |
                | 基线对照 | Euler | 20 | ~3000ms |
                """)
        
        gr.Markdown("""
        ---
        **LCM-LoRA 加速实验系统** | 深度学习课程项目 | 2025
        """)
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("LCM-LoRA 实时图像生成系统")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 启动 Gradio
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
