"""
Report templates for experiment reports.

Contains Markdown templates with placeholders for experiment data.

Requirements: 13.5
"""

# Main experiment report template
EXPERIMENT_REPORT_TEMPLATE = """# {experiment_name} 实验报告

**生成时间**: {generation_time}  
**实验 ID**: {experiment_id}

---

## 1. 实验概述

本报告记录了 LCM-LoRA 扩散模型加速实验的完整结果，包括性能对比、质量评估和参数分析。

### 1.1 实验环境

| 项目 | 值 |
|------|-----|
| GPU | {gpu_info} |
| CUDA 版本 | {cuda_version} |
| PyTorch 版本 | {pytorch_version} |
| 总运行次数 | {total_runs} |

### 1.2 测试配置

{configs_table}

---

## 2. 性能对比结果

### 2.1 延迟对比

{latency_table}

{latency_chart}

### 2.2 显存占用对比

{vram_table}

{vram_chart}

---

## 3. 质量评估结果

### 3.1 CLIPScore 对比

{quality_table}

{quality_chart}

---

## 4. 最优配置分析

| 指标 | 最优配置 |
|------|----------|
| 最快速度 | {best_speed_config} |
| 最高质量 | {best_quality_config} |
| 最佳权衡 | {best_tradeoff_config} |

---

## 5. 生成样例对比

{sample_images}

---

## 6. 结论

{conclusions}

---

## 附录

### A. 完整实验数据

详细实验数据请参考:
- CSV 数据文件: `{csv_path}`
- JSON 日志文件: `{json_path}`

### B. 图表文件

{chart_list}

---

*本报告由 LCM-LoRA 加速实验系统自动生成*
"""

# Configuration table template
CONFIG_TABLE_TEMPLATE = """| 配置名称 | 调度器 | 步数 | Guidance Scale | LCM-LoRA |
|----------|--------|------|----------------|----------|
{rows}"""

# Metrics table template (for latency, vram, quality)
METRICS_TABLE_TEMPLATE = """| 配置 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
{rows}"""

# Sample images section template
SAMPLE_IMAGES_TEMPLATE = """### Prompt: {prompt}

{image_grid}
"""

# Chart reference template
CHART_REFERENCE_TEMPLATE = """![{chart_name}]({chart_path})"""

# Conclusions template
CONCLUSIONS_TEMPLATE = """基于实验结果，我们得出以下结论:

1. **加速效果**: LCM-LoRA 在 {lcm_steps} 步配置下，相比 Euler {baseline_steps} 步基线，延迟降低了 **{speedup:.1f}x**。

2. **质量保持**: LCM-LoRA 配置的 CLIPScore 为 {lcm_quality:.4f}，基线为 {baseline_quality:.4f}，质量{quality_comparison}。

3. **显存优化**: 启用优化后，峰值显存从 {baseline_vram:.0f} MB 降低到 {optimized_vram:.0f} MB，节省 {vram_saving:.1f}%。

4. **推荐配置**: 综合考虑速度和质量，推荐使用 **{recommended_config}** 配置。
"""

# LaTeX table templates
LATEX_COMPARISON_TABLE_TEMPLATE = r"""\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{tab:{label}}}
\begin{{tabular}}{{l{col_spec}}}
\toprule
{header} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

LATEX_ABLATION_TABLE_TEMPLATE = r"""\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{tab:{label}}}
\begin{{tabular}}{{lcccc}}
\toprule
配置 & 延迟 (ms) & 显存 (MB) & 延迟变化 & 显存变化 \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

# Minimal report template for quick summaries
MINIMAL_REPORT_TEMPLATE = """# {experiment_name} 实验摘要

**时间**: {generation_time}

## 关键结果

- **最快配置**: {best_speed_config} ({best_speed_latency:.1f} ms)
- **最高质量**: {best_quality_config} (CLIPScore: {best_quality_score:.4f})
- **推荐配置**: {best_tradeoff_config}

## 性能对比

{latency_table}

## 质量对比

{quality_table}
"""
