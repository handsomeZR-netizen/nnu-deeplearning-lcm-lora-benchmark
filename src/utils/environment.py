"""
Environment Manager - 收集和管理环境信息

实现 Requirements 12.1, 12.2, 12.3, 12.4:
- 导出 pip freeze 到 requirements.txt
- 记录 Python、PyTorch、CUDA 版本
- 记录 GPU 型号和驱动版本
- 记录所有关键依赖的版本约束
"""

import sys
import os
import subprocess
import json
import platform
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class GPUInfo:
    """GPU 信息"""
    name: str
    memory_total_mb: int
    driver_version: str
    cuda_version: str
    compute_capability: Optional[str] = None


@dataclass
class EnvironmentInfo:
    """完整环境信息"""
    # 系统信息
    os_name: str
    os_version: str
    platform: str
    
    # Python 信息
    python_version: str
    python_executable: str
    
    # PyTorch 信息
    pytorch_version: str
    pytorch_cuda_version: Optional[str]
    pytorch_cudnn_version: Optional[str]
    
    # CUDA 信息
    cuda_available: bool
    cuda_version: Optional[str]
    
    # GPU 信息
    gpu_count: int
    gpus: List[GPUInfo] = field(default_factory=list)
    
    # 关键依赖版本
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    # 时间戳
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = asdict(self)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EnvironmentInfo':
        """从字典创建"""
        gpus_data = data.pop('gpus', [])
        gpus = [GPUInfo(**gpu) for gpu in gpus_data]
        return cls(gpus=gpus, **data)


class EnvironmentManager:
    """环境信息管理器"""
    
    # 关键依赖列表
    KEY_DEPENDENCIES = [
        'torch',
        'torchvision',
        'diffusers',
        'transformers',
        'accelerate',
        'safetensors',
        'huggingface_hub',
        'xformers',
        'gradio',
        'matplotlib',
        'numpy',
        'pillow',
        'scipy',
        'tqdm',
        'pyyaml',
        'hypothesis',
        'pytest',
        'clip',
        'lpips',
    ]
    
    def __init__(self, output_dir: str = "outputs"):
        """
        初始化环境管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_environment_info(self) -> EnvironmentInfo:
        """
        收集完整的环境信息
        
        Returns:
            EnvironmentInfo: 环境信息对象
        """
        # 收集系统信息
        os_name = platform.system()
        os_version = platform.version()
        platform_info = platform.platform()
        
        # 收集 Python 信息
        python_version = sys.version
        python_executable = sys.executable
        
        # 收集 PyTorch 信息
        pytorch_info = self._collect_pytorch_info()
        
        # 收集 CUDA 信息
        cuda_info = self._collect_cuda_info()
        
        # 收集 GPU 信息
        gpu_info = self._collect_gpu_info()
        
        # 收集依赖版本
        dependencies = self._collect_dependencies()
        
        return EnvironmentInfo(
            os_name=os_name,
            os_version=os_version,
            platform=platform_info,
            python_version=python_version,
            python_executable=python_executable,
            pytorch_version=pytorch_info.get('version', 'N/A'),
            pytorch_cuda_version=pytorch_info.get('cuda_version'),
            pytorch_cudnn_version=pytorch_info.get('cudnn_version'),
            cuda_available=cuda_info.get('available', False),
            cuda_version=cuda_info.get('version'),
            gpu_count=gpu_info.get('count', 0),
            gpus=gpu_info.get('gpus', []),
            dependencies=dependencies,
        )
    
    def _collect_pytorch_info(self) -> Dict:
        """收集 PyTorch 信息"""
        info = {}
        try:
            import torch
            info['version'] = torch.__version__
            
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
                if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                    info['cudnn_version'] = str(torch.backends.cudnn.version())
        except ImportError:
            info['version'] = 'Not installed'
        
        return info
    
    def _collect_cuda_info(self) -> Dict:
        """收集 CUDA 信息"""
        info = {'available': False, 'version': None}
        
        try:
            import torch
            if torch.cuda.is_available():
                info['available'] = True
                info['version'] = torch.version.cuda
        except ImportError:
            pass
        
        return info
    
    def _collect_gpu_info(self) -> Dict:
        """收集 GPU 信息"""
        info = {'count': 0, 'gpus': []}
        
        try:
            import torch
            if torch.cuda.is_available():
                info['count'] = torch.cuda.device_count()
                
                for i in range(info['count']):
                    props = torch.cuda.get_device_properties(i)
                    
                    # 获取驱动版本
                    driver_version = self._get_nvidia_driver_version()
                    
                    gpu = GPUInfo(
                        name=props.name,
                        memory_total_mb=props.total_memory // (1024 * 1024),
                        driver_version=driver_version,
                        cuda_version=torch.version.cuda or 'N/A',
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                    info['gpus'].append(gpu)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to collect GPU info: {e}")
        
        return info
    
    def _get_nvidia_driver_version(self) -> str:
        """获取 NVIDIA 驱动版本"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return 'N/A'
    
    def _collect_dependencies(self) -> Dict[str, str]:
        """收集关键依赖版本"""
        dependencies = {}
        
        for package in self.KEY_DEPENDENCIES:
            version = self._get_package_version(package)
            if version:
                dependencies[package] = version
        
        return dependencies
    
    def _get_package_version(self, package_name: str) -> Optional[str]:
        """获取包版本"""
        try:
            # 尝试使用 importlib.metadata
            from importlib.metadata import version
            return version(package_name)
        except Exception:
            pass
        
        # 尝试直接导入
        try:
            module = __import__(package_name.replace('-', '_'))
            if hasattr(module, '__version__'):
                return module.__version__
        except ImportError:
            pass
        
        return None
    
    def export_requirements(self, output_path: Optional[str] = None) -> str:
        """
        导出 requirements.txt
        
        Args:
            output_path: 输出路径，默认为 output_dir/requirements.txt
            
        Returns:
            str: 输出文件路径
        """
        if output_path is None:
            output_path = str(self.output_dir / "requirements.txt")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                return output_path
            else:
                raise RuntimeError(f"pip freeze failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("pip freeze timed out")
    
    def export_environment_json(self, output_path: Optional[str] = None) -> str:
        """
        导出环境信息为 JSON
        
        Args:
            output_path: 输出路径
            
        Returns:
            str: 输出文件路径
        """
        if output_path is None:
            output_path = str(self.output_dir / "environment.json")
        
        env_info = self.collect_environment_info()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(env_info.to_dict(), f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def generate_environment_report(self) -> str:
        """
        生成环境报告 (Markdown 格式)
        
        Returns:
            str: Markdown 格式的环境报告
        """
        env_info = self.collect_environment_info()
        
        lines = [
            "# 环境信息报告",
            "",
            f"生成时间: {env_info.collected_at}",
            "",
            "## 系统信息",
            "",
            f"- 操作系统: {env_info.os_name}",
            f"- 系统版本: {env_info.os_version}",
            f"- 平台: {env_info.platform}",
            "",
            "## Python 环境",
            "",
            f"- Python 版本: {env_info.python_version.split()[0]}",
            f"- Python 路径: {env_info.python_executable}",
            "",
            "## PyTorch 环境",
            "",
            f"- PyTorch 版本: {env_info.pytorch_version}",
            f"- CUDA 版本 (PyTorch): {env_info.pytorch_cuda_version or 'N/A'}",
            f"- cuDNN 版本: {env_info.pytorch_cudnn_version or 'N/A'}",
            "",
            "## CUDA 环境",
            "",
            f"- CUDA 可用: {'是' if env_info.cuda_available else '否'}",
            f"- CUDA 版本: {env_info.cuda_version or 'N/A'}",
            "",
            "## GPU 信息",
            "",
            f"- GPU 数量: {env_info.gpu_count}",
            "",
        ]
        
        for i, gpu in enumerate(env_info.gpus):
            lines.extend([
                f"### GPU {i}",
                "",
                f"- 名称: {gpu.name}",
                f"- 显存: {gpu.memory_total_mb} MB",
                f"- 驱动版本: {gpu.driver_version}",
                f"- 计算能力: {gpu.compute_capability or 'N/A'}",
                "",
            ])
        
        lines.extend([
            "## 关键依赖版本",
            "",
            "| 包名 | 版本 |",
            "|------|------|",
        ])
        
        for package, version in sorted(env_info.dependencies.items()):
            lines.append(f"| {package} | {version} |")
        
        return "\n".join(lines)
    
    def check_dependencies(self) -> Dict[str, any]:
        """
        检查依赖版本兼容性
        
        Returns:
            Dict: 包含警告和建议的字典
        """
        warnings = []
        suggestions = []
        
        deps = self._collect_dependencies()
        
        # 检查 transformers 和 huggingface_hub 兼容性
        if 'transformers' in deps and 'huggingface_hub' in deps:
            try:
                from packaging.version import parse as parse_version
                
                transformers_ver = parse_version(deps['transformers'])
                hf_hub_ver = parse_version(deps['huggingface_hub'])
                
                if transformers_ver >= parse_version("4.40") and hf_hub_ver < parse_version("0.23"):
                    warnings.append(
                        "transformers >= 4.40 与 huggingface_hub < 0.23 存在潜在兼容性问题"
                    )
                    suggestions.append(
                        "建议升级 huggingface_hub: pip install huggingface_hub>=0.23"
                    )
            except ImportError:
                pass
        
        # 检查 xformers 是否安装
        if 'xformers' not in deps:
            suggestions.append(
                "建议安装 xformers 以获得更好的显存优化: pip install xformers"
            )
        
        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'dependencies': deps,
        }
