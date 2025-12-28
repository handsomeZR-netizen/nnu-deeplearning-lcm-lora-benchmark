"""
Visualizer - 实验结果可视化模块

生成对比实验图表、步数曲线、消融实验表格、参数敏感性图和样例对比网格图。

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 使用 Agg 后端，避免需要显示设备
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class Visualizer:
    """
    生成实验结果图表
    
    支持:
    - 对比实验柱状图 (延迟、显存、CLIPScore)
    - 步数-质量-速度曲线
    - 消融实验表格
    - 参数敏感性分析图
    - 生成样例对比网格图
    
    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
    """
    
    # 配色方案
    COLORS = {
        'euler': '#3498db',      # 蓝色
        'dpm_solver': '#2ecc71', # 绿色
        'lcm': '#e74c3c',        # 红色
        'default': '#9b59b6',    # 紫色
    }
    
    # 图表样式
    STYLES = {
        'paper': {
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 150,
        },
        'presentation': {
            'figure.figsize': (12, 8),
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 100,
        },
    }
    
    def __init__(self, output_dir: str, style: str = "paper"):
        """
        初始化可视化器
        
        Args:
            output_dir: 图表输出目录
            style: 图表样式 ("paper" 或 "presentation")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self._apply_style()
        
        logger.info(f"Visualizer initialized: output_dir={self.output_dir}, style={style}")
    
    def _apply_style(self) -> None:
        """应用图表样式"""
        style_config = self.STYLES.get(self.style, self.STYLES['paper'])
        for key, value in style_config.items():
            plt.rcParams[key] = value
    
    def _get_color(self, config_name: str) -> str:
        """根据配置名称获取颜色"""
        config_lower = config_name.lower()
        if 'euler' in config_lower:
            return self.COLORS['euler']
        elif 'dpm' in config_lower:
            return self.COLORS['dpm_solver']
        elif 'lcm' in config_lower:
            return self.COLORS['lcm']
        return self.COLORS['default']
    
    def _save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = None) -> List[str]:
        """
        保存图表为多种格式
        
        Args:
            fig: matplotlib Figure 对象
            filename: 文件名 (不含扩展名)
            formats: 保存格式列表，默认 ['png', 'pdf']
        
        Returns:
            保存的文件路径列表
        """
        if formats is None:
            formats = ['png', 'pdf']
        
        saved_paths = []
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=150)
            saved_paths.append(str(filepath))
            logger.debug(f"Saved figure: {filepath}")
        
        return saved_paths


    def plot_comparison_bars(
        self,
        results: Any,  # ExperimentResults
        metrics: List[str] = None,
        title: str = "对比实验结果",
        filename: str = "comparison_bars",
    ) -> List[str]:
        """
        生成对比实验柱状图
        
        支持延迟、显存、CLIPScore 指标的柱状图对比
        
        Args:
            results: ExperimentResults 对象，包含 runtime_stats, vram_stats, quality_stats
            metrics: 要绘制的指标列表，默认 ['latency', 'vram', 'clip_score']
            title: 图表标题
            filename: 输出文件名 (不含扩展名)
        
        Returns:
            保存的文件路径列表
        
        Requirements: 11.1
        """
        if metrics is None:
            metrics = ['latency', 'vram', 'clip_score']
        
        # 获取配置名称
        config_names = list(results.runtime_stats.keys())
        if not config_names:
            logger.warning("No results to plot")
            return []
        
        # 准备数据
        data = {metric: [] for metric in metrics}
        errors = {metric: [] for metric in metrics}
        
        for config_name in config_names:
            if 'latency' in metrics:
                stats = results.runtime_stats.get(config_name, {})
                data['latency'].append(stats.get('mean', 0))
                errors['latency'].append(stats.get('std', 0))
            
            if 'vram' in metrics:
                stats = results.vram_stats.get(config_name, {})
                data['vram'].append(stats.get('mean', 0))
                errors['vram'].append(stats.get('std', 0))
            
            if 'clip_score' in metrics:
                stats = results.quality_stats.get(config_name, {})
                data['clip_score'].append(stats.get('mean', 0))
                errors['clip_score'].append(stats.get('std', 0))
        
        # 创建子图
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        if num_metrics == 1:
            axes = [axes]
        
        # 指标配置
        metric_config = {
            'latency': {'label': '延迟 (ms)', 'color': '#3498db'},
            'vram': {'label': '显存 (MB)', 'color': '#e74c3c'},
            'clip_score': {'label': 'CLIPScore', 'color': '#2ecc71'},
        }
        
        x = np.arange(len(config_names))
        bar_width = 0.6
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            config = metric_config.get(metric, {'label': metric, 'color': '#9b59b6'})
            
            # 为每个配置使用不同颜色
            colors = [self._get_color(name) for name in config_names]
            
            bars = ax.bar(
                x, data[metric], bar_width,
                yerr=errors[metric],
                color=colors,
                capsize=5,
                edgecolor='black',
                linewidth=0.5,
            )
            
            ax.set_xlabel('配置')
            ax.set_ylabel(config['label'])
            ax.set_title(config['label'])
            ax.set_xticks(x)
            ax.set_xticklabels(config_names, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, val in zip(bars, data[metric]):
                height = bar.get_height()
                ax.annotate(
                    f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                )
            
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        saved_paths = self._save_figure(fig, filename)
        plt.close(fig)
        
        logger.info(f"Generated comparison bars: {filename}")
        return saved_paths


    def plot_steps_curve(
        self,
        results: Any,  # ExperimentResults
        title: str = "步数-质量-速度曲线",
        filename: str = "steps_curve",
    ) -> List[str]:
        """
        生成步数-质量-速度折线图
        
        展示不同步数配置下的质量和速度关系
        
        Args:
            results: ExperimentResults 对象
            title: 图表标题
            filename: 输出文件名 (不含扩展名)
        
        Returns:
            保存的文件路径列表
        
        Requirements: 11.3
        """
        # 提取数据并按步数排序
        data_points = []
        
        for config in results.configs:
            config_name = config.name
            if config_name in results.runtime_stats:
                latency = results.runtime_stats[config_name].get('mean', 0)
                quality = results.quality_stats.get(config_name, {}).get('mean', 0)
                
                data_points.append({
                    'name': config_name,
                    'steps': config.num_steps,
                    'latency': latency,
                    'quality': quality,
                    'scheduler': config.scheduler_type,
                    'use_lcm': config.use_lcm_lora,
                })
        
        if not data_points:
            logger.warning("No data points to plot")
            return []
        
        # 按调度器类型分组
        lcm_points = [p for p in data_points if p['use_lcm']]
        baseline_points = [p for p in data_points if not p['use_lcm']]
        
        # 按步数排序
        lcm_points.sort(key=lambda x: x['steps'])
        baseline_points.sort(key=lambda x: x['steps'])
        
        # 创建双 Y 轴图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # 绘制延迟曲线 (左 Y 轴)
        if lcm_points:
            steps_lcm = [p['steps'] for p in lcm_points]
            latency_lcm = [p['latency'] for p in lcm_points]
            line1, = ax1.plot(
                steps_lcm, latency_lcm, 'o-',
                color=self.COLORS['lcm'],
                linewidth=2, markersize=8,
                label='LCM 延迟'
            )
        
        if baseline_points:
            steps_base = [p['steps'] for p in baseline_points]
            latency_base = [p['latency'] for p in baseline_points]
            line2, = ax1.plot(
                steps_base, latency_base, 's--',
                color=self.COLORS['euler'],
                linewidth=2, markersize=8,
                label='Baseline 延迟'
            )
        
        ax1.set_xlabel('采样步数')
        ax1.set_ylabel('延迟 (ms)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # 绘制质量曲线 (右 Y 轴)
        if lcm_points:
            quality_lcm = [p['quality'] for p in lcm_points]
            line3, = ax2.plot(
                steps_lcm, quality_lcm, '^-',
                color='#27ae60',
                linewidth=2, markersize=8,
                label='LCM CLIPScore'
            )
        
        if baseline_points:
            quality_base = [p['quality'] for p in baseline_points]
            line4, = ax2.plot(
                steps_base, quality_base, 'd--',
                color='#16a085',
                linewidth=2, markersize=8,
                label='Baseline CLIPScore'
            )
        
        ax2.set_ylabel('CLIPScore', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # 合并图例
        lines = []
        labels = []
        for ax in [ax1, ax2]:
            for line, label in zip(*ax.get_legend_handles_labels()):
                lines.append(line)
                labels.append(label)
        ax1.legend(lines, labels, loc='upper right')
        
        # 添加数据点标签
        for points in [lcm_points, baseline_points]:
            for p in points:
                ax1.annotate(
                    p['name'],
                    xy=(p['steps'], p['latency']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7,
                )
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        saved_paths = self._save_figure(fig, filename)
        plt.close(fig)
        
        logger.info(f"Generated steps curve: {filename}")
        return saved_paths


    def plot_ablation_table(
        self,
        results: Any,  # AblationResults
        title: str = "消融实验结果",
        filename: str = "ablation_table",
    ) -> List[str]:
        """
        生成消融实验表格图
        
        展示各优化项的独立贡献
        
        Args:
            results: AblationResults 对象
            title: 图表标题
            filename: 输出文件名 (不含扩展名)
        
        Returns:
            保存的文件路径列表
        
        Requirements: 11.2
        """
        # 准备表格数据
        config_names = list(results.runtime_stats.keys())
        if not config_names:
            logger.warning("No ablation results to plot")
            return []
        
        # 表格列
        columns = ['配置', '延迟 (ms)', '显存 (MB)', '延迟变化', '显存变化']
        
        # 获取基线数据
        baseline_name = 'full_optimization'
        baseline_latency = results.runtime_stats.get(baseline_name, {}).get('mean', 0)
        baseline_vram = results.vram_stats.get(baseline_name, {}).get('mean', 0)
        
        # 准备行数据
        rows = []
        cell_colors = []
        
        for config_name in config_names:
            latency = results.runtime_stats.get(config_name, {}).get('mean', 0)
            vram = results.vram_stats.get(config_name, {}).get('mean', 0)
            
            # 计算变化
            latency_diff = latency - baseline_latency
            vram_diff = vram - baseline_vram
            
            latency_diff_str = f"+{latency_diff:.1f}" if latency_diff > 0 else f"{latency_diff:.1f}"
            vram_diff_str = f"+{vram_diff:.0f}" if vram_diff > 0 else f"{vram_diff:.0f}"
            
            if config_name == baseline_name:
                latency_diff_str = "-"
                vram_diff_str = "-"
            
            rows.append([
                config_name,
                f"{latency:.1f}",
                f"{vram:.0f}",
                latency_diff_str,
                vram_diff_str,
            ])
            
            # 设置单元格颜色
            row_colors = ['white'] * 5
            if config_name == baseline_name:
                row_colors = ['#e8f5e9'] * 5  # 浅绿色
            elif latency_diff > 0:
                row_colors[3] = '#ffebee'  # 浅红色 (性能下降)
            elif latency_diff < 0:
                row_colors[3] = '#e8f5e9'  # 浅绿色 (性能提升)
            
            if vram_diff > 0:
                row_colors[4] = '#ffebee'
            elif vram_diff < 0:
                row_colors[4] = '#e8f5e9'
            
            cell_colors.append(row_colors)
        
        # 创建表格图
        fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.6 + 2)))
        ax.axis('off')
        
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellColours=cell_colors,
            colColours=['#bbdefb'] * len(columns),
            loc='center',
            cellLoc='center',
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_text_props(fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        saved_paths = self._save_figure(fig, filename)
        plt.close(fig)
        
        logger.info(f"Generated ablation table: {filename}")
        return saved_paths


    def plot_parameter_sensitivity(
        self,
        results: Any,  # ParameterResults or Dict[str, ParameterResults]
        title: str = "参数敏感性分析",
        filename: str = "parameter_sensitivity",
    ) -> List[str]:
        """
        生成参数敏感性分析图
        
        展示不同参数值对性能和质量的影响
        当没有质量数据时，只显示延迟图
        
        Args:
            results: ParameterResults 对象或 Dict[str, ParameterResults]
            title: 图表标题
            filename: 输出文件名 (不含扩展名)
        
        Returns:
            保存的文件路径列表
        
        Requirements: 11.4
        """
        # 处理输入格式
        if hasattr(results, 'parameter_name'):
            # 单个 ParameterResults
            param_results = {results.parameter_name: results}
        else:
            # Dict[str, ParameterResults]
            param_results = results
        
        num_params = len(param_results)
        if num_params == 0:
            logger.warning("No parameter results to plot")
            return []
        
        # 预先检查是否有任何质量数据
        has_any_quality_data = False
        for param_result in param_results.values():
            for value_key in param_result.runtime_stats.keys():
                quality_stats = param_result.quality_stats.get(value_key, {})
                if quality_stats.get('mean', 0) > 0:
                    has_any_quality_data = True
                    break
            if has_any_quality_data:
                break
        
        # 根据是否有质量数据决定布局
        if has_any_quality_data:
            # 有质量数据：2列布局（延迟 + 质量）
            fig, axes = plt.subplots(num_params, 2, figsize=(12, 4 * num_params))
            if num_params == 1:
                axes = axes.reshape(1, -1)
        else:
            # 无质量数据：1列布局（仅延迟）
            fig, axes = plt.subplots(num_params, 1, figsize=(8, 4 * num_params))
            if num_params == 1:
                axes = np.array([axes])
        
        for idx, (param_name, param_result) in enumerate(param_results.items()):
            if has_any_quality_data:
                ax_latency = axes[idx, 0]
                ax_quality = axes[idx, 1]
            else:
                ax_latency = axes[idx]
                ax_quality = None
            
            # 获取参数值和对应的统计数据
            param_values = []
            latencies = []
            latency_stds = []
            qualities = []
            quality_stds = []
            
            for value_key in param_result.runtime_stats.keys():
                param_values.append(value_key)
                
                latency_stats = param_result.runtime_stats.get(value_key, {})
                latencies.append(latency_stats.get('mean', 0))
                latency_stds.append(latency_stats.get('std', 0))
                
                quality_stats = param_result.quality_stats.get(value_key, {})
                qualities.append(quality_stats.get('mean', 0))
                quality_stds.append(quality_stats.get('std', 0))
            
            x = np.arange(len(param_values))
            
            # 绘制延迟图
            bars = ax_latency.bar(
                x, latencies, yerr=latency_stds,
                color='#3498db', capsize=5,
                edgecolor='black', linewidth=0.5,
            )
            ax_latency.set_xlabel(param_name)
            ax_latency.set_ylabel('延迟 (ms)')
            ax_latency.set_title(f'{param_name} vs 延迟')
            ax_latency.set_xticks(x)
            ax_latency.set_xticklabels(param_values, rotation=45, ha='right')
            ax_latency.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for i, (val, std) in enumerate(zip(latencies, latency_stds)):
                ax_latency.annotate(
                    f'{val:.1f}',
                    xy=(i, val),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', fontsize=8,
                )
            
            # 绘制质量图（仅当有质量数据时）
            if ax_quality is not None:
                if any(q > 0 for q in qualities):
                    ax_quality.bar(
                        x, qualities, yerr=quality_stds,
                        color='#2ecc71', capsize=5,
                        edgecolor='black', linewidth=0.5,
                    )
                    ax_quality.set_xlabel(param_name)
                    ax_quality.set_ylabel('CLIPScore')
                    ax_quality.set_title(f'{param_name} vs CLIPScore')
                    ax_quality.set_xticks(x)
                    ax_quality.set_xticklabels(param_values, rotation=45, ha='right')
                    ax_quality.grid(axis='y', alpha=0.3)
                    
                    for i, (val, std) in enumerate(zip(qualities, quality_stds)):
                        ax_quality.annotate(
                            f'{val:.3f}',
                            xy=(i, val),
                            xytext=(0, 5),
                            textcoords='offset points',
                            ha='center', fontsize=8,
                        )
                else:
                    ax_quality.text(
                        0.5, 0.5, '无质量数据',
                        ha='center', va='center',
                        transform=ax_quality.transAxes,
                        fontsize=12, color='gray',
                    )
                    ax_quality.set_title(f'{param_name} vs CLIPScore')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        saved_paths = self._save_figure(fig, filename)
        plt.close(fig)
        
        logger.info(f"Generated parameter sensitivity plot: {filename}")
        return saved_paths


    def create_comparison_grid(
        self,
        images: Dict[str, Image.Image],
        prompts: List[str] = None,
        title: str = "生成样例对比",
        filename: str = "comparison_grid",
        max_cols: int = 4,
    ) -> List[str]:
        """
        生成生成样例的对比网格图
        
        Args:
            images: 配置名称到图像的映射 Dict[config_name, Image]
                   或 Dict[config_name, List[Image]] 用于多个 prompt
            prompts: 对应的 prompt 列表 (可选)
            title: 图表标题
            filename: 输出文件名 (不含扩展名)
            max_cols: 每行最大列数
        
        Returns:
            保存的文件路径列表
        
        Requirements: 11.5
        """
        if not images:
            logger.warning("No images to create grid")
            return []
        
        config_names = list(images.keys())
        
        # 检查是否是多图像格式
        first_value = images[config_names[0]]
        if isinstance(first_value, list):
            # 多 prompt 格式: Dict[config_name, List[Image]]
            return self._create_multi_prompt_grid(
                images, prompts, title, filename, max_cols
            )
        else:
            # 单 prompt 格式: Dict[config_name, Image]
            return self._create_single_prompt_grid(
                images, prompts, title, filename, max_cols
            )
    
    def _create_single_prompt_grid(
        self,
        images: Dict[str, Image.Image],
        prompts: List[str],
        title: str,
        filename: str,
        max_cols: int,
    ) -> List[str]:
        """创建单 prompt 的对比网格"""
        config_names = list(images.keys())
        num_images = len(config_names)
        
        # 计算网格布局
        num_cols = min(num_images, max_cols)
        num_rows = (num_images + num_cols - 1) // num_cols
        
        # 创建图
        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(4 * num_cols, 4 * num_rows + 1)
        )
        
        # 确保 axes 是二维数组
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 绘制图像
        for idx, config_name in enumerate(config_names):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
            
            img = images[config_name]
            ax.imshow(img)
            ax.set_title(config_name, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(num_images, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis('off')
        
        # 添加 prompt 信息
        if prompts and len(prompts) > 0:
            prompt_text = prompts[0][:100] + "..." if len(prompts[0]) > 100 else prompts[0]
            fig.text(
                0.5, 0.02, f"Prompt: {prompt_text}",
                ha='center', fontsize=10, style='italic',
                wrap=True,
            )
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        saved_paths = self._save_figure(fig, filename)
        plt.close(fig)
        
        logger.info(f"Generated comparison grid: {filename}")
        return saved_paths
    
    def _create_multi_prompt_grid(
        self,
        images: Dict[str, List[Image.Image]],
        prompts: List[str],
        title: str,
        filename: str,
        max_cols: int,
    ) -> List[str]:
        """创建多 prompt 的对比网格"""
        config_names = list(images.keys())
        num_configs = len(config_names)
        num_prompts = len(images[config_names[0]]) if config_names else 0
        
        if num_prompts == 0:
            logger.warning("No images in the list")
            return []
        
        # 创建网格: 行=prompts, 列=configs
        fig, axes = plt.subplots(
            num_prompts, num_configs,
            figsize=(3 * num_configs, 3 * num_prompts + 1)
        )
        
        # 确保 axes 是二维数组
        if num_prompts == 1 and num_configs == 1:
            axes = np.array([[axes]])
        elif num_prompts == 1:
            axes = axes.reshape(1, -1)
        elif num_configs == 1:
            axes = axes.reshape(-1, 1)
        
        # 绘制图像
        for col_idx, config_name in enumerate(config_names):
            for row_idx, img in enumerate(images[config_name]):
                ax = axes[row_idx, col_idx]
                ax.imshow(img)
                ax.axis('off')
                
                # 第一行添加配置名称
                if row_idx == 0:
                    ax.set_title(config_name, fontsize=10, fontweight='bold')
        
        # 添加 prompt 标签
        if prompts:
            for row_idx, prompt in enumerate(prompts[:num_prompts]):
                prompt_short = prompt[:30] + "..." if len(prompt) > 30 else prompt
                axes[row_idx, 0].set_ylabel(
                    prompt_short,
                    fontsize=8,
                    rotation=0,
                    ha='right',
                    va='center',
                    labelpad=60,
                )
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        saved_paths = self._save_figure(fig, filename)
        plt.close(fig)
        
        logger.info(f"Generated multi-prompt comparison grid: {filename}")
        return saved_paths
