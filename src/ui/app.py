"""
GradioApp - Interactive image generation system with Gradio interface.

Provides a web-based UI for:
- Text-to-image generation with parameter controls
- Baseline vs LCM mode comparison
- Real-time metrics display (latency, VRAM)
- CSV log export functionality

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
from PIL import Image

from ..core.pipeline import PipelineManager, VRAMError, ModelLoadError
from ..core.models import GenerationResult
from ..benchmark.logger import ExperimentLogger

logger = logging.getLogger(__name__)


class GradioApp:
    """
    Gradio 交互式生成系统
    
    Features:
    - Text input with parameter controls (steps, guidance, seed, resolution)
    - Mode selection (Baseline vs LCM)
    - Side-by-side comparison display
    - Results display with metrics (image, latency, VRAM)
    - CSV log export
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """
    
    # Default parameter ranges
    STEPS_RANGE = (1, 50, 1)  # min, max, step
    GUIDANCE_RANGE = (0.0, 20.0, 0.5)
    RESOLUTION_OPTIONS = ["512x512", "768x768", "1024x1024"]
    SCHEDULER_OPTIONS = ["euler", "dpm_solver", "ddim"]
    
    def __init__(
        self,
        model_dir: str,
        lcm_lora_dir: str,
        output_dir: str = "outputs",
        device: str = "cuda"
    ):
        """
        初始化 Gradio 应用
        
        Args:
            model_dir: 基础模型目录
            lcm_lora_dir: LCM-LoRA 权重目录
            output_dir: 输出目录 (日志、图像)
            device: 推理设备
        """
        self.model_dir = model_dir
        self.lcm_lora_dir = lcm_lora_dir
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
        # Pipeline manager (延迟初始化)
        self._pipeline_manager: Optional[PipelineManager] = None
        self._current_mode: Optional[str] = None  # "baseline" or "lcm"
        
        # 实验日志器
        self._logger = ExperimentLogger(
            log_dir=str(self.output_dir / "logs"),
            experiment_name="gradio_session"
        )
        
        # 生成历史
        self._generation_history: List[GenerationResult] = []
        
        logger.info(f"GradioApp 初始化: model_dir={model_dir}, device={device}")
    
    @property
    def is_pipeline_loaded(self) -> bool:
        """检查管线是否已加载"""
        return self._pipeline_manager is not None and self._pipeline_manager.is_loaded
    
    def _ensure_pipeline(self, mode: str, scheduler: str = "euler") -> None:
        """
        确保管线已加载并处于正确模式
        
        Args:
            mode: "baseline" or "lcm"
            scheduler: 调度器类型 (仅 baseline 模式使用)
        """
        # 初始化 pipeline manager
        if self._pipeline_manager is None:
            self._pipeline_manager = PipelineManager(
                model_dir=self.model_dir,
                lcm_lora_dir=self.lcm_lora_dir,
                device=self.device
            )
        
        # 检查是否需要切换模式
        need_reload = (
            not self._pipeline_manager.is_loaded or
            self._current_mode != mode or
            (mode == "baseline" and self._pipeline_manager.current_scheduler_type != scheduler)
        )
        
        if need_reload:
            logger.info(f"加载管线: mode={mode}, scheduler={scheduler}")
            
            if mode == "lcm":
                self._pipeline_manager.load_lcm_pipeline(fuse_lora=True)
            else:
                self._pipeline_manager.load_baseline_pipeline(scheduler_type=scheduler)
            
            # 应用默认优化
            self._pipeline_manager.apply_optimizations(
                attention_slicing=True,
                vae_slicing=True,
                sdpa=True
            )
            
            self._current_mode = mode
    
    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """解析分辨率字符串"""
        parts = resolution_str.lower().replace(" ", "").split("x")
        return int(parts[0]), int(parts[1])

    def generate_single(
        self,
        prompt: str,
        mode: str,
        scheduler: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        resolution: str
    ) -> Tuple[Optional[Image.Image], str, str]:
        """
        单次图像生成
        
        Args:
            prompt: 文本提示
            mode: "baseline" or "lcm"
            scheduler: 调度器类型
            num_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            resolution: 分辨率字符串
        
        Returns:
            (image, metrics_text, status_text)
        
        Requirements: 10.1, 10.3, 10.4
        """
        if not prompt or not prompt.strip():
            return None, "", "⚠️ 请输入文本提示"
        
        try:
            # 确保管线已加载
            effective_scheduler = scheduler if mode == "baseline" else "lcm"
            self._ensure_pipeline(mode, effective_scheduler)
            
            # 解析分辨率
            width, height = self._parse_resolution(resolution)
            
            # 执行生成
            result = self._pipeline_manager.generate(
                prompt=prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                width=width,
                height=height
            )
            
            # 记录结果
            self._generation_history.append(result)
            self._logger.log_result(result)
            
            # 格式化指标文本
            metrics_text = self._format_metrics(result)
            status_text = f"✅ 生成成功 | {result.scheduler_type} | {result.num_steps} 步"
            
            return result.image, metrics_text, status_text
            
        except VRAMError as e:
            logger.error(f"显存不足: {e}")
            return None, "", f"❌ 显存不足: {e}"
        except ModelLoadError as e:
            logger.error(f"模型加载失败: {e}")
            return None, "", f"❌ 模型加载失败: {e}"
        except Exception as e:
            logger.exception(f"生成失败: {e}")
            return None, "", f"❌ 生成失败: {e}"
    
    def generate_comparison(
        self,
        prompt: str,
        baseline_scheduler: str,
        baseline_steps: int,
        baseline_guidance: float,
        lcm_steps: int,
        lcm_guidance: float,
        seed: int,
        resolution: str
    ) -> Tuple[
        Optional[Image.Image], str, str,
        Optional[Image.Image], str, str,
        str
    ]:
        """
        并排对比生成 (Baseline vs LCM)
        
        Args:
            prompt: 文本提示
            baseline_scheduler: 基线调度器
            baseline_steps: 基线步数
            baseline_guidance: 基线引导强度
            lcm_steps: LCM 步数
            lcm_guidance: LCM 引导强度
            seed: 随机种子
            resolution: 分辨率
        
        Returns:
            (baseline_image, baseline_metrics, baseline_status,
             lcm_image, lcm_metrics, lcm_status,
             comparison_summary)
        
        Requirements: 10.2, 10.6
        """
        if not prompt or not prompt.strip():
            empty = (None, "", "⚠️ 请输入文本提示")
            return (*empty, *empty, "")
        
        results = []
        
        # 生成基线图像
        baseline_result = self.generate_single(
            prompt=prompt,
            mode="baseline",
            scheduler=baseline_scheduler,
            num_steps=baseline_steps,
            guidance_scale=baseline_guidance,
            seed=seed,
            resolution=resolution
        )
        results.append(("Baseline", baseline_result))
        
        # 生成 LCM 图像
        lcm_result = self.generate_single(
            prompt=prompt,
            mode="lcm",
            scheduler="lcm",
            num_steps=lcm_steps,
            guidance_scale=lcm_guidance,
            seed=seed,
            resolution=resolution
        )
        results.append(("LCM", lcm_result))
        
        # 生成对比摘要
        comparison_summary = self._generate_comparison_summary(
            baseline_result, lcm_result,
            baseline_steps, lcm_steps
        )
        
        return (
            baseline_result[0], baseline_result[1], baseline_result[2],
            lcm_result[0], lcm_result[1], lcm_result[2],
            comparison_summary
        )
    
    def _format_metrics(self, result: GenerationResult) -> str:
        """格式化指标显示文本"""
        return (
            f"**延迟**: {result.latency_ms:.1f} ms\n"
            f"**显存**: {result.peak_vram_mb:.0f} MB\n"
            f"**步数**: {result.num_steps}\n"
            f"**引导强度**: {result.guidance_scale}\n"
            f"**种子**: {result.seed}\n"
            f"**分辨率**: {result.resolution[0]}x{result.resolution[1]}\n"
            f"**调度器**: {result.scheduler_type}"
        )
    
    def _generate_comparison_summary(
        self,
        baseline_result: Tuple,
        lcm_result: Tuple,
        baseline_steps: int,
        lcm_steps: int
    ) -> str:
        """生成对比摘要"""
        # 解析结果 (image, metrics_text, status_text)
        baseline_ok = baseline_result[0] is not None
        lcm_ok = lcm_result[0] is not None
        
        if not baseline_ok or not lcm_ok:
            return "⚠️ 对比不完整，部分生成失败"
        
        # 从历史记录获取最近两次结果
        if len(self._generation_history) < 2:
            return "⚠️ 历史记录不足"
        
        baseline_gen = self._generation_history[-2]
        lcm_gen = self._generation_history[-1]
        
        # 计算加速比
        speedup = baseline_gen.latency_ms / lcm_gen.latency_ms if lcm_gen.latency_ms > 0 else 0
        step_reduction = (baseline_steps - lcm_steps) / baseline_steps * 100
        
        summary = (
            f"## 📊 对比摘要\n\n"
            f"| 指标 | Baseline ({baseline_gen.scheduler_type}) | LCM |\n"
            f"|------|----------|-----|\n"
            f"| 步数 | {baseline_steps} | {lcm_steps} |\n"
            f"| 延迟 | {baseline_gen.latency_ms:.1f} ms | {lcm_gen.latency_ms:.1f} ms |\n"
            f"| 显存 | {baseline_gen.peak_vram_mb:.0f} MB | {lcm_gen.peak_vram_mb:.0f} MB |\n\n"
            f"**🚀 加速比**: {speedup:.2f}x\n"
            f"**📉 步数减少**: {step_reduction:.0f}%"
        )
        
        return summary

    def export_logs_csv(self) -> Tuple[Optional[str], str]:
        """
        导出生成日志为 CSV
        
        Returns:
            (file_path, status_message)
        
        Requirements: 10.5
        """
        if not self._generation_history:
            return None, "⚠️ 没有生成记录可导出"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generation_log_{timestamp}.csv"
            filepath = self._logger.export_csv(filename)
            
            return str(filepath), f"✅ 日志已导出: {filepath}"
        except Exception as e:
            logger.exception(f"导出失败: {e}")
            return None, f"❌ 导出失败: {e}"
    
    def get_generation_count(self) -> int:
        """获取生成次数"""
        return len(self._generation_history)
    
    def clear_history(self) -> str:
        """清空生成历史"""
        count = len(self._generation_history)
        self._generation_history.clear()
        return f"✅ 已清空 {count} 条记录"
    
    def build_interface(self) -> gr.Blocks:
        """
        构建 Gradio 界面
        
        包含:
        - 文本输入框
        - 参数控制面板 (steps, guidance, seed, resolution)
        - 模式选择 (Baseline vs LCM)
        - 结果展示 (图像 + 指标)
        - 对比展示面板
        - 日志导出按钮
        
        Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
        """
        with gr.Blocks(
            title="LCM-LoRA 图像生成系统",
            theme=gr.themes.Soft()
        ) as interface:
            
            # 标题
            gr.Markdown(
                """
                # 🎨 LCM-LoRA 扩散模型加速系统
                
                基于 Stable Diffusion + LCM-LoRA 的实时图像生成系统，支持 2-8 步快速采样。
                """
            )
            
            with gr.Tabs():
                # ==================== 单图生成标签页 ====================
                with gr.TabItem("🖼️ 单图生成"):
                    with gr.Row():
                        # 左侧: 输入和参数
                        with gr.Column(scale=1):
                            # 文本输入
                            prompt_input = gr.Textbox(
                                label="文本提示 (Prompt)",
                                placeholder="输入图像描述，例如: a beautiful sunset over mountains",
                                lines=3
                            )
                            
                            # 模式选择
                            mode_select = gr.Radio(
                                choices=["baseline", "lcm"],
                                value="lcm",
                                label="生成模式"
                            )
                            
                            # 调度器选择 (仅 baseline 模式)
                            scheduler_select = gr.Dropdown(
                                choices=self.SCHEDULER_OPTIONS,
                                value="euler",
                                label="调度器 (Baseline 模式)",
                                interactive=True
                            )
                            
                            # 参数控制
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=1, maximum=50, value=4, step=1,
                                    label="推理步数"
                                )
                                guidance_slider = gr.Slider(
                                    minimum=0.0, maximum=20.0, value=1.0, step=0.5,
                                    label="引导强度 (Guidance Scale)"
                                )
                            
                            with gr.Row():
                                seed_input = gr.Number(
                                    value=42,
                                    label="随机种子",
                                    precision=0
                                )
                                resolution_select = gr.Dropdown(
                                    choices=self.RESOLUTION_OPTIONS,
                                    value="512x512",
                                    label="分辨率"
                                )
                            
                            # 生成按钮
                            generate_btn = gr.Button(
                                "🎨 生成图像",
                                variant="primary",
                                size="lg"
                            )
                        
                        # 右侧: 结果展示
                        with gr.Column(scale=1):
                            output_image = gr.Image(
                                label="生成结果",
                                type="pil"
                            )
                            output_metrics = gr.Markdown(
                                label="指标信息"
                            )
                            output_status = gr.Textbox(
                                label="状态",
                                interactive=False
                            )
                    
                    # 绑定生成事件
                    generate_btn.click(
                        fn=self.generate_single,
                        inputs=[
                            prompt_input, mode_select, scheduler_select,
                            steps_slider, guidance_slider, seed_input,
                            resolution_select
                        ],
                        outputs=[output_image, output_metrics, output_status]
                    )
                
                # ==================== 对比生成标签页 ====================
                with gr.TabItem("⚖️ 对比生成"):
                    gr.Markdown("### Baseline vs LCM 并排对比")
                    
                    # 共享参数
                    with gr.Row():
                        compare_prompt = gr.Textbox(
                            label="文本提示 (Prompt)",
                            placeholder="输入图像描述",
                            lines=2,
                            scale=3
                        )
                        compare_seed = gr.Number(
                            value=42,
                            label="随机种子",
                            precision=0,
                            scale=1
                        )
                        compare_resolution = gr.Dropdown(
                            choices=self.RESOLUTION_OPTIONS,
                            value="512x512",
                            label="分辨率",
                            scale=1
                        )
                    
                    with gr.Row():
                        # Baseline 参数
                        with gr.Column():
                            gr.Markdown("#### Baseline 配置")
                            compare_baseline_scheduler = gr.Dropdown(
                                choices=self.SCHEDULER_OPTIONS,
                                value="euler",
                                label="调度器"
                            )
                            compare_baseline_steps = gr.Slider(
                                minimum=1, maximum=50, value=20, step=1,
                                label="步数"
                            )
                            compare_baseline_guidance = gr.Slider(
                                minimum=0.0, maximum=20.0, value=7.5, step=0.5,
                                label="引导强度"
                            )
                        
                        # LCM 参数
                        with gr.Column():
                            gr.Markdown("#### LCM 配置")
                            compare_lcm_steps = gr.Slider(
                                minimum=1, maximum=8, value=4, step=1,
                                label="步数"
                            )
                            compare_lcm_guidance = gr.Slider(
                                minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                label="引导强度"
                            )
                    
                    # 对比生成按钮
                    compare_btn = gr.Button(
                        "⚖️ 开始对比生成",
                        variant="primary",
                        size="lg"
                    )
                    
                    # 结果展示
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Baseline 结果")
                            compare_baseline_image = gr.Image(
                                label="Baseline",
                                type="pil"
                            )
                            compare_baseline_metrics = gr.Markdown()
                            compare_baseline_status = gr.Textbox(
                                label="状态",
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### LCM 结果")
                            compare_lcm_image = gr.Image(
                                label="LCM",
                                type="pil"
                            )
                            compare_lcm_metrics = gr.Markdown()
                            compare_lcm_status = gr.Textbox(
                                label="状态",
                                interactive=False
                            )
                    
                    # 对比摘要
                    compare_summary = gr.Markdown(
                        label="对比摘要"
                    )
                    
                    # 绑定对比生成事件
                    compare_btn.click(
                        fn=self.generate_comparison,
                        inputs=[
                            compare_prompt,
                            compare_baseline_scheduler,
                            compare_baseline_steps,
                            compare_baseline_guidance,
                            compare_lcm_steps,
                            compare_lcm_guidance,
                            compare_seed,
                            compare_resolution
                        ],
                        outputs=[
                            compare_baseline_image, compare_baseline_metrics,
                            compare_baseline_status,
                            compare_lcm_image, compare_lcm_metrics,
                            compare_lcm_status,
                            compare_summary
                        ]
                    )
                
                # ==================== 日志导出标签页 ====================
                with gr.TabItem("📊 日志导出"):
                    gr.Markdown(
                        """
                        ### 生成日志管理
                        
                        导出生成记录为 CSV 格式，包含所有生成参数和性能指标。
                        """
                    )
                    
                    with gr.Row():
                        export_btn = gr.Button(
                            "📥 导出 CSV 日志",
                            variant="primary"
                        )
                        clear_btn = gr.Button(
                            "🗑️ 清空历史",
                            variant="secondary"
                        )
                    
                    export_file = gr.File(
                        label="导出文件"
                    )
                    export_status = gr.Textbox(
                        label="状态",
                        interactive=False
                    )
                    
                    # 绑定导出事件
                    def export_and_return():
                        filepath, status = self.export_logs_csv()
                        return filepath, status
                    
                    export_btn.click(
                        fn=export_and_return,
                        inputs=[],
                        outputs=[export_file, export_status]
                    )
                    
                    clear_btn.click(
                        fn=self.clear_history,
                        inputs=[],
                        outputs=[export_status]
                    )
            
            # 页脚
            gr.Markdown(
                """
                ---
                **LCM-LoRA 扩散模型加速系统** | 
                基于 Stable Diffusion + LCM-LoRA | 
                支持 2-8 步快速采样
                """
            )
        
        return interface
    
    def launch(
        self,
        share: bool = False,
        port: int = 7860,
        server_name: str = "127.0.0.1"
    ) -> None:
        """
        启动 Gradio 服务
        
        Args:
            share: 是否创建公共链接
            port: 服务端口
            server_name: 服务器地址
        """
        interface = self.build_interface()
        
        logger.info(f"启动 Gradio 服务: port={port}, share={share}")
        
        interface.launch(
            share=share,
            server_port=port,
            server_name=server_name
        )


def create_app(
    model_dir: str = "models/dreamshaper-7",
    lcm_lora_dir: str = "models/lcm-lora-sdv1-5",
    output_dir: str = "outputs",
    device: str = "cuda"
) -> GradioApp:
    """
    创建 GradioApp 实例的工厂函数
    
    Args:
        model_dir: 基础模型目录
        lcm_lora_dir: LCM-LoRA 权重目录
        output_dir: 输出目录
        device: 推理设备
    
    Returns:
        GradioApp 实例
    """
    return GradioApp(
        model_dir=model_dir,
        lcm_lora_dir=lcm_lora_dir,
        output_dir=output_dir,
        device=device
    )


# 便捷启动函数
def main():
    """命令行启动入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LCM-LoRA Gradio 界面")
    parser.add_argument("--model-dir", default="models/dreamshaper-7", help="基础模型目录")
    parser.add_argument("--lcm-lora-dir", default="models/lcm-lora-sdv1-5", help="LCM-LoRA 目录")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--device", default="cuda", help="推理设备")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    app = create_app(
        model_dir=args.model_dir,
        lcm_lora_dir=args.lcm_lora_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    app.launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
