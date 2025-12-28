"""
ReportGenerator - 实验报告生成模块

生成 Markdown 格式的实验报告，支持自动填充实验数据和图表路径。
同时支持生成 LaTeX 格式的表格，可直接用于论文。

Requirements: 13.5, 13.6, 13.7
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.models import ExperimentConfig, ExperimentSummary
from .templates import (
    EXPERIMENT_REPORT_TEMPLATE,
    CONFIG_TABLE_TEMPLATE,
    METRICS_TABLE_TEMPLATE,
    SAMPLE_IMAGES_TEMPLATE,
    CHART_REFERENCE_TEMPLATE,
    CONCLUSIONS_TEMPLATE,
    LATEX_COMPARISON_TABLE_TEMPLATE,
    LATEX_ABLATION_TABLE_TEMPLATE,
    MINIMAL_REPORT_TEMPLATE,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    生成实验报告
    
    支持:
    - Markdown 格式实验报告生成
    - 自动填充实验数据和图表路径
    - LaTeX 格式表格生成
    
    Requirements: 13.5, 13.6, 13.7
    """
    
    def __init__(self, template_path: Optional[str] = None, output_dir: str = "outputs/reports"):
        """
        初始化报告生成器
        
        Args:
            template_path: 自定义模板路径 (可选，使用内置模板)
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_path = template_path
        self._custom_template: Optional[str] = None
        
        if template_path and Path(template_path).exists():
            with open(template_path, "r", encoding="utf-8") as f:
                self._custom_template = f.read()
            logger.info(f"Loaded custom template from: {template_path}")
        
        logger.info(f"ReportGenerator initialized: output_dir={self.output_dir}")

    def generate_experiment_report(
        self,
        experiment_summary: ExperimentSummary,
        charts: List[str] = None,
        sample_images: List[str] = None,
        csv_path: str = "",
        json_path: str = "",
        output_filename: str = None,
    ) -> str:
        """
        生成 Markdown 格式实验报告
        
        Args:
            experiment_summary: ExperimentSummary 实例，包含实验统计数据
            charts: 图表文件路径列表
            sample_images: 样例图像文件路径列表
            csv_path: CSV 数据文件路径
            json_path: JSON 日志文件路径
            output_filename: 输出文件名 (可选，默认使用实验名称)
        
        Returns:
            生成的报告文件路径
        
        Requirements: 13.5, 13.6
        """
        charts = charts or []
        sample_images = sample_images or []
        
        # 使用自定义模板或内置模板
        template = self._custom_template or EXPERIMENT_REPORT_TEMPLATE
        
        # 准备模板数据
        template_data = self._prepare_template_data(
            experiment_summary, charts, sample_images, csv_path, json_path
        )
        
        # 填充模板
        try:
            report_content = template.format(**template_data)
        except KeyError as e:
            logger.error(f"Template key error: {e}")
            raise ValueError(f"Missing template key: {e}")
        
        # 确定输出文件名
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{experiment_summary.experiment_name}_{timestamp}.md"
        
        # 保存报告
        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Generated experiment report: {output_path}")
        return str(output_path)
    
    def _prepare_template_data(
        self,
        summary: ExperimentSummary,
        charts: List[str],
        sample_images: List[str],
        csv_path: str,
        json_path: str,
    ) -> Dict[str, str]:
        """准备模板填充数据"""
        # 基本信息
        data = {
            "experiment_name": summary.experiment_name,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": summary.experiment_name[:8],
            "gpu_info": summary.gpu_info or "Unknown",
            "cuda_version": summary.cuda_version or "Unknown",
            "pytorch_version": summary.pytorch_version or "Unknown",
            "total_runs": str(summary.total_runs),
            "csv_path": csv_path or "N/A",
            "json_path": json_path or "N/A",
        }
        
        # 配置表格
        data["configs_table"] = self._generate_configs_table(summary.configs)
        
        # 延迟表格和图表
        data["latency_table"] = self._generate_metrics_table(
            summary.latency_stats, "延迟 (ms)"
        )
        data["latency_chart"] = self._find_chart_reference(charts, "latency", "comparison")
        
        # 显存表格和图表
        data["vram_table"] = self._generate_metrics_table(
            summary.vram_stats, "显存 (MB)"
        )
        data["vram_chart"] = self._find_chart_reference(charts, "vram", "comparison")
        
        # 质量表格和图表
        data["quality_table"] = self._generate_metrics_table(
            summary.quality_stats, "CLIPScore"
        )
        data["quality_chart"] = self._find_chart_reference(charts, "quality", "clip")
        
        # 最优配置
        data["best_speed_config"] = summary.best_speed_config or "N/A"
        data["best_quality_config"] = summary.best_quality_config or "N/A"
        data["best_tradeoff_config"] = summary.best_tradeoff_config or "N/A"
        
        # 样例图像
        data["sample_images"] = self._generate_sample_images_section(sample_images)
        
        # 图表列表
        data["chart_list"] = self._generate_chart_list(charts)
        
        # 结论
        data["conclusions"] = self._generate_conclusions(summary)
        
        return data
    
    def _generate_configs_table(self, configs: List[ExperimentConfig]) -> str:
        """生成配置表格"""
        if not configs:
            return "*无配置数据*"
        
        rows = []
        for config in configs:
            lcm_status = "✓" if config.use_lcm_lora else "✗"
            row = f"| {config.name} | {config.scheduler_type} | {config.num_steps} | {config.guidance_scale} | {lcm_status} |"
            rows.append(row)
        
        return CONFIG_TABLE_TEMPLATE.format(rows="\n".join(rows))
    
    def _generate_metrics_table(
        self, 
        stats: Dict[str, Dict[str, float]], 
        metric_name: str
    ) -> str:
        """生成指标统计表格"""
        if not stats:
            return f"*无{metric_name}数据*"
        
        rows = []
        for config_name, stat in stats.items():
            mean = stat.get("mean", 0)
            std = stat.get("std", 0)
            min_val = stat.get("min", 0)
            max_val = stat.get("max", 0)
            
            # 根据数值大小选择格式
            if mean > 100:
                row = f"| {config_name} | {mean:.1f} | {std:.1f} | {min_val:.1f} | {max_val:.1f} |"
            else:
                row = f"| {config_name} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |"
            rows.append(row)
        
        return METRICS_TABLE_TEMPLATE.format(rows="\n".join(rows))
    
    def _find_chart_reference(
        self, 
        charts: List[str], 
        *keywords: str
    ) -> str:
        """查找匹配关键词的图表并生成引用"""
        for chart_path in charts:
            chart_name = Path(chart_path).stem.lower()
            if any(kw.lower() in chart_name for kw in keywords):
                return CHART_REFERENCE_TEMPLATE.format(
                    chart_name=Path(chart_path).stem,
                    chart_path=chart_path
                )
        return ""
    
    def _generate_sample_images_section(self, sample_images: List[str]) -> str:
        """生成样例图像部分"""
        if not sample_images:
            return "*无样例图像*"
        
        sections = []
        for img_path in sample_images:
            img_name = Path(img_path).stem
            section = f"![{img_name}]({img_path})"
            sections.append(section)
        
        return "\n\n".join(sections)
    
    def _generate_chart_list(self, charts: List[str]) -> str:
        """生成图表文件列表"""
        if not charts:
            return "*无图表文件*"
        
        items = []
        for chart_path in charts:
            chart_name = Path(chart_path).stem
            items.append(f"- `{chart_path}`: {chart_name}")
        
        return "\n".join(items)
    
    def _generate_conclusions(self, summary: ExperimentSummary) -> str:
        """生成实验结论"""
        # 提取关键数据
        lcm_config = None
        baseline_config = None
        
        for config_name in summary.latency_stats.keys():
            if "lcm" in config_name.lower():
                if lcm_config is None or summary.latency_stats[config_name]["mean"] < summary.latency_stats[lcm_config]["mean"]:
                    lcm_config = config_name
            elif "euler" in config_name.lower():
                baseline_config = config_name
        
        if not lcm_config or not baseline_config:
            return "*数据不足，无法生成结论*"
        
        lcm_latency = summary.latency_stats[lcm_config]["mean"]
        baseline_latency = summary.latency_stats[baseline_config]["mean"]
        speedup = baseline_latency / lcm_latency if lcm_latency > 0 else 0
        
        lcm_quality = summary.quality_stats.get(lcm_config, {}).get("mean", 0)
        baseline_quality = summary.quality_stats.get(baseline_config, {}).get("mean", 0)
        
        quality_comparison = "基本持平" if abs(lcm_quality - baseline_quality) < 0.01 else (
            "略有提升" if lcm_quality > baseline_quality else "略有下降"
        )
        
        baseline_vram = summary.vram_stats.get(baseline_config, {}).get("mean", 0)
        optimized_vram = summary.vram_stats.get(lcm_config, {}).get("mean", 0)
        vram_saving = (baseline_vram - optimized_vram) / baseline_vram * 100 if baseline_vram > 0 else 0
        
        # 提取步数
        lcm_steps = "4"
        baseline_steps = "20"
        for config in summary.configs:
            if config.name == lcm_config:
                lcm_steps = str(config.num_steps)
            elif config.name == baseline_config:
                baseline_steps = str(config.num_steps)
        
        return CONCLUSIONS_TEMPLATE.format(
            lcm_steps=lcm_steps,
            baseline_steps=baseline_steps,
            speedup=speedup,
            lcm_quality=lcm_quality,
            baseline_quality=baseline_quality,
            quality_comparison=quality_comparison,
            baseline_vram=baseline_vram,
            optimized_vram=optimized_vram,
            vram_saving=vram_saving,
            recommended_config=summary.best_tradeoff_config or lcm_config,
        )

    def generate_latex_tables(
        self,
        experiment_summary: ExperimentSummary,
        table_type: str = "comparison",
        caption: str = None,
        label: str = None,
    ) -> str:
        """
        生成 LaTeX 格式表格
        
        Args:
            experiment_summary: ExperimentSummary 实例
            table_type: 表格类型 ("comparison", "ablation", "quality")
            caption: 表格标题
            label: LaTeX 标签
        
        Returns:
            LaTeX 表格字符串
        
        Requirements: 13.7
        """
        if table_type == "comparison":
            return self._generate_latex_comparison_table(
                experiment_summary, caption, label
            )
        elif table_type == "ablation":
            return self._generate_latex_ablation_table(
                experiment_summary, caption, label
            )
        elif table_type == "quality":
            return self._generate_latex_quality_table(
                experiment_summary, caption, label
            )
        else:
            raise ValueError(f"Unknown table type: {table_type}")
    
    def _generate_latex_comparison_table(
        self,
        summary: ExperimentSummary,
        caption: str = None,
        label: str = None,
    ) -> str:
        """生成对比实验 LaTeX 表格"""
        caption = caption or "LCM-LoRA 加速效果对比"
        label = label or "comparison"
        
        # 表头
        header = "配置 & 步数 & 延迟 (ms) & 显存 (MB) & CLIPScore"
        col_spec = "cccc"
        
        # 表格行
        rows = []
        for config in summary.configs:
            config_name = config.name
            steps = config.num_steps
            
            latency = summary.latency_stats.get(config_name, {}).get("mean", 0)
            latency_std = summary.latency_stats.get(config_name, {}).get("std", 0)
            
            vram = summary.vram_stats.get(config_name, {}).get("mean", 0)
            
            quality = summary.quality_stats.get(config_name, {}).get("mean", 0)
            quality_std = summary.quality_stats.get(config_name, {}).get("std", 0)
            
            row = f"{config_name} & {steps} & {latency:.1f} $\\pm$ {latency_std:.1f} & {vram:.0f} & {quality:.4f} $\\pm$ {quality_std:.4f} \\\\"
            rows.append(row)
        
        return LATEX_COMPARISON_TABLE_TEMPLATE.format(
            caption=caption,
            label=label,
            col_spec=col_spec,
            header=header,
            rows="\n".join(rows),
        )
    
    def _generate_latex_ablation_table(
        self,
        summary: ExperimentSummary,
        caption: str = None,
        label: str = None,
    ) -> str:
        """生成消融实验 LaTeX 表格"""
        caption = caption or "消融实验结果"
        label = label or "ablation"
        
        # 获取基线数据
        baseline_name = "full_optimization"
        baseline_latency = summary.latency_stats.get(baseline_name, {}).get("mean", 0)
        baseline_vram = summary.vram_stats.get(baseline_name, {}).get("mean", 0)
        
        rows = []
        for config_name, latency_stat in summary.latency_stats.items():
            latency = latency_stat.get("mean", 0)
            vram = summary.vram_stats.get(config_name, {}).get("mean", 0)
            
            latency_diff = latency - baseline_latency
            vram_diff = vram - baseline_vram
            
            latency_diff_str = f"+{latency_diff:.1f}" if latency_diff > 0 else f"{latency_diff:.1f}"
            vram_diff_str = f"+{vram_diff:.0f}" if vram_diff > 0 else f"{vram_diff:.0f}"
            
            if config_name == baseline_name:
                latency_diff_str = "-"
                vram_diff_str = "-"
            
            row = f"{config_name} & {latency:.1f} & {vram:.0f} & {latency_diff_str} & {vram_diff_str} \\\\"
            rows.append(row)
        
        return LATEX_ABLATION_TABLE_TEMPLATE.format(
            caption=caption,
            label=label,
            rows="\n".join(rows),
        )
    
    def _generate_latex_quality_table(
        self,
        summary: ExperimentSummary,
        caption: str = None,
        label: str = None,
    ) -> str:
        """生成质量评估 LaTeX 表格"""
        caption = caption or "图像质量评估结果"
        label = label or "quality"
        
        header = "配置 & CLIPScore & FID & LPIPS"
        col_spec = "ccc"
        
        rows = []
        for config_name, quality_stat in summary.quality_stats.items():
            clip_score = quality_stat.get("mean", 0)
            clip_std = quality_stat.get("std", 0)
            
            # FID 和 LPIPS 可能不存在
            fid = quality_stat.get("fid", "-")
            lpips = quality_stat.get("lpips", "-")
            
            if isinstance(fid, (int, float)):
                fid = f"{fid:.2f}"
            if isinstance(lpips, (int, float)):
                lpips = f"{lpips:.4f}"
            
            row = f"{config_name} & {clip_score:.4f} $\\pm$ {clip_std:.4f} & {fid} & {lpips} \\\\"
            rows.append(row)
        
        return LATEX_COMPARISON_TABLE_TEMPLATE.format(
            caption=caption,
            label=label,
            col_spec=col_spec,
            header=header,
            rows="\n".join(rows),
        )
    
    def generate_minimal_report(
        self,
        experiment_summary: ExperimentSummary,
        output_filename: str = None,
    ) -> str:
        """
        生成简化版实验摘要报告
        
        Args:
            experiment_summary: ExperimentSummary 实例
            output_filename: 输出文件名 (可选)
        
        Returns:
            生成的报告文件路径
        """
        # 获取最优配置的具体数值
        best_speed_latency = 0
        best_quality_score = 0
        
        if experiment_summary.best_speed_config:
            best_speed_latency = experiment_summary.latency_stats.get(
                experiment_summary.best_speed_config, {}
            ).get("mean", 0)
        
        if experiment_summary.best_quality_config:
            best_quality_score = experiment_summary.quality_stats.get(
                experiment_summary.best_quality_config, {}
            ).get("mean", 0)
        
        # 生成表格
        latency_table = self._generate_metrics_table(
            experiment_summary.latency_stats, "延迟 (ms)"
        )
        quality_table = self._generate_metrics_table(
            experiment_summary.quality_stats, "CLIPScore"
        )
        
        # 填充模板
        report_content = MINIMAL_REPORT_TEMPLATE.format(
            experiment_name=experiment_summary.experiment_name,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            best_speed_config=experiment_summary.best_speed_config or "N/A",
            best_speed_latency=best_speed_latency,
            best_quality_config=experiment_summary.best_quality_config or "N/A",
            best_quality_score=best_quality_score,
            best_tradeoff_config=experiment_summary.best_tradeoff_config or "N/A",
            latency_table=latency_table,
            quality_table=quality_table,
        )
        
        # 确定输出文件名
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{experiment_summary.experiment_name}_summary_{timestamp}.md"
        
        # 保存报告
        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Generated minimal report: {output_path}")
        return str(output_path)
    
    def export_latex_tables_to_file(
        self,
        experiment_summary: ExperimentSummary,
        output_filename: str = None,
    ) -> str:
        """
        导出所有 LaTeX 表格到文件
        
        Args:
            experiment_summary: ExperimentSummary 实例
            output_filename: 输出文件名 (可选)
        
        Returns:
            生成的 LaTeX 文件路径
        
        Requirements: 13.7
        """
        # 生成所有表格
        tables = []
        
        # 对比表格
        tables.append("% 对比实验表格")
        tables.append(self.generate_latex_tables(
            experiment_summary, "comparison",
            caption="LCM-LoRA 加速效果对比",
            label="comparison"
        ))
        tables.append("")
        
        # 质量表格
        if experiment_summary.quality_stats:
            tables.append("% 质量评估表格")
            tables.append(self.generate_latex_tables(
                experiment_summary, "quality",
                caption="图像质量评估结果",
                label="quality"
            ))
        
        content = "\n".join(tables)
        
        # 确定输出文件名
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{experiment_summary.experiment_name}_tables_{timestamp}.tex"
        
        # 保存文件
        output_path = self.output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Exported LaTeX tables: {output_path}")
        return str(output_path)
