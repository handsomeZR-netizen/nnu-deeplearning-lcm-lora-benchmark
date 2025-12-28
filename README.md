# LCM-LoRA 扩散模型加速系统

基于 LCM-LoRA 的 Stable Diffusion 实时推理加速与显存优化实验评测系统。

## 项目简介

本项目实现了面向消费级显卡的扩散模型实时推理加速系统，通过集成 LCM-LoRA 加速模块，将传统 20-50 步采样压缩到 2-8 步，并结合多种推理优化技术，构建了一个可复现的实验评测系统和交互式生成系统。

### 核心特性

- **LCM-LoRA 加速**: 2-8 步快速采样，显著降低推理延迟
- **多种显存优化**: Attention Slicing、VAE Slicing/Tiling、xFormers/SDPA
- **完整评测系统**: 对比实验、消融实验、参数分析
- **质量评估**: CLIPScore、FID、LPIPS 指标
- **交互式界面**: Gradio Web UI 支持实时生成和对比
- **自动化报告**: Markdown 报告和 LaTeX 表格生成

## 环境要求

- Python 3.10+
- PyTorch 2.0+ (支持 CUDA)
- NVIDIA GPU (建议 8GB+ 显存)
- CUDA 11.8+ / 12.x

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装 PyTorch (根据 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 下载模型

模型会在首次运行时自动下载，或手动下载：

```bash
# 基础模型 (DreamShaper-7)
huggingface-cli download Lykon/dreamshaper-7 --local-dir models/dreamshaper-7

# LCM-LoRA 权重
huggingface-cli download latent-consistency/lcm-lora-sdv1-5 --local-dir models/lcm-lora-sdv1-5
```

### 3. 启动交互式界面

```bash
python -m src.ui.app
```

访问 http://localhost:7860 使用 Web 界面。

## 项目结构

```
.
├── src/
│   ├── core/           # 核心推理管线
│   │   ├── models.py   # 数据模型定义
│   │   └── pipeline.py # PipelineManager
│   ├── benchmark/      # 基准测试
│   │   ├── runner.py   # BenchmarkRunner
│   │   └── logger.py   # ExperimentLogger
│   ├── metrics/        # 质量评估
│   │   └── collector.py # MetricsCollector
│   ├── dataset/        # 数据集构建
│   │   └── builder.py  # DatasetBuilder
│   ├── visualization/  # 可视化
│   │   └── visualizer.py
│   ├── report/         # 报告生成
│   │   ├── generator.py
│   │   └── templates.py
│   ├── ui/             # Gradio 界面
│   │   └── app.py
│   └── utils/          # 工具函数
│       └── environment.py # 环境管理
├── configs/            # 配置文件
├── models/             # 模型文件
├── outputs/            # 输出目录
│   ├── images/         # 生成图像
│   ├── logs/           # 实验日志
│   ├── reports/        # 实验报告
│   └── charts/         # 图表
├── tests/              # 测试文件
└── README.md
```

## 使用指南

### 命令行生成

```python
from src.core.pipeline import PipelineManager

# 初始化管线
pm = PipelineManager(
    model_dir="models/dreamshaper-7",
    lcm_lora_dir="models/lcm-lora-sdv1-5"
)

# 加载 LCM 管线
pm.load_lcm_pipeline()
pm.apply_optimizations(attention_slicing=True, vae_slicing=True)

# 生成图像
result = pm.generate(
    prompt="a beautiful sunset over mountains",
    num_steps=4,
    guidance_scale=1.0,
    seed=42
)

result.image.save("output.png")
print(f"延迟: {result.latency_ms:.2f}ms, 显存: {result.peak_vram_mb:.2f}MB")
```

### 运行基准测试

```python
from src.benchmark.runner import BenchmarkRunner
from src.core.pipeline import PipelineManager

pm = PipelineManager("models/dreamshaper-7", "models/lcm-lora-sdv1-5")
runner = BenchmarkRunner(pm, "outputs")

# 运行对比实验
results = runner.run_comparison_experiment(
    prompts=["a cat", "a dog", "a landscape"],
    seeds=[42, 123, 456],
    configs=[...]  # 实验配置
)
```

### 质量评估

```python
from src.metrics.collector import MetricsCollector

collector = MetricsCollector()

# 计算 CLIPScore
score = collector.compute_clip_score(image, prompt)
print(f"CLIPScore: {score:.4f}")
```

### 生成实验报告

```python
from src.report.generator import ReportGenerator

generator = ReportGenerator("outputs/reports")
report_path = generator.generate_experiment_report(
    experiment_summary=summary,
    charts=["chart1.png", "chart2.png"],
    sample_images=["sample1.png", "sample2.png"]
)
```

## 配置说明

### 默认配置 (configs/default_config.yaml)

```yaml
model:
  base_model: "models/dreamshaper-7"
  lcm_lora: "models/lcm-lora-sdv1-5"
  
generation:
  default_steps: 4
  default_guidance: 1.0
  default_resolution: [512, 512]
  
optimizations:
  attention_slicing: true
  vae_slicing: true
  vae_tiling: false
  xformers: false
  sdpa: true
```

## 依赖冲突解决方案

### 问题 1: transformers 与 huggingface_hub 版本冲突

**症状**: `ImportError` 或 `AttributeError` 相关错误

**原因**: transformers >= 4.40 需要 huggingface_hub >= 0.23

**解决方案**:
```bash
pip install huggingface_hub>=0.23.0
pip install transformers>=4.40.0
```

### 问题 2: xFormers 安装失败

**症状**: 编译错误或 CUDA 版本不匹配

**解决方案**:
```bash
# 方法 1: 使用预编译版本
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# 方法 2: 跳过 xFormers，使用 PyTorch SDPA
# 系统会自动回退到 SDPA，无需额外配置
```

### 问题 3: CUDA 内存不足 (OOM)

**症状**: `torch.cuda.OutOfMemoryError`

**解决方案**:
1. 启用更多显存优化:
```python
pm.apply_optimizations(
    attention_slicing=True,
    vae_slicing=True,
    vae_tiling=True  # 对于高分辨率
)
```

2. 降低分辨率到 512x512

3. 使用 CPU offload (最后手段):
```python
pipe.enable_model_cpu_offload()
```

### 问题 4: diffusers 版本兼容性

**症状**: API 变更导致的错误

**解决方案**:
```bash
# 推荐版本
pip install diffusers>=0.25.0,<0.30.0
```

### 问题 5: CLIP 模型加载失败

**症状**: 无法加载 CLIP 模型进行评估

**解决方案**:
```bash
pip install git+https://github.com/openai/CLIP.git
# 或
pip install open-clip-torch
```

## 环境信息收集

使用内置工具收集环境信息：

```python
from src.utils.environment import EnvironmentManager

em = EnvironmentManager("outputs")

# 收集环境信息
info = em.collect_environment_info()
print(f"PyTorch: {info.pytorch_version}")
print(f"CUDA: {info.cuda_version}")
print(f"GPU: {info.gpus[0].name if info.gpus else 'N/A'}")

# 导出 requirements.txt
em.export_requirements("requirements.txt")

# 导出环境 JSON
em.export_environment_json("environment.json")

# 生成环境报告
report = em.generate_environment_report()
print(report)

# 检查依赖兼容性
check_result = em.check_dependencies()
for warning in check_result['warnings']:
    print(f"警告: {warning}")
```

## 实验复现

为确保实验可复现，请：

1. 使用相同的随机种子
2. 记录完整的环境信息
3. 使用相同的 prompt 数据集
4. 保存所有实验配置

```python
# 导出环境信息
em = EnvironmentManager()
em.export_requirements("requirements_frozen.txt")
em.export_environment_json("environment.json")
```

## 性能参考

在 RTX 3080 (10GB) 上的典型性能：

| 配置 | 步数 | 延迟 | 显存 | CLIPScore |
|------|------|------|------|-----------|
| Euler | 20 | ~2.5s | ~6GB | 0.32 |
| DPM-Solver | 20 | ~2.3s | ~6GB | 0.31 |
| LCM-LoRA | 4 | ~0.5s | ~5GB | 0.30 |
| LCM-LoRA | 8 | ~0.9s | ~5GB | 0.31 |

## 许可证

MIT License

## 致谢

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model)
- [DreamShaper](https://civitai.com/models/4384/dreamshaper)
- [Diffusers](https://github.com/huggingface/diffusers)
