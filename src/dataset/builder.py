"""
DatasetBuilder - Evaluation dataset construction from COCO Captions.

Provides functionality to load, filter, sample, and analyze prompts
from COCO Captions dataset for benchmark evaluation.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# Category keywords for classification
CATEGORY_KEYWORDS = {
    "simple_object": [
        "apple", "banana", "orange", "cup", "bottle", "book", "phone",
        "chair", "table", "lamp", "clock", "vase", "bowl", "plate",
        "ball", "toy", "box", "bag", "shoe", "hat", "umbrella",
    ],
    "portrait": [
        "man", "woman", "person", "people", "boy", "girl", "child",
        "baby", "face", "portrait", "selfie", "group", "crowd",
        "standing", "sitting", "walking", "smiling", "looking",
    ],
    "complex_scene": [
        "street", "city", "building", "park", "beach", "mountain",
        "forest", "river", "ocean", "sky", "sunset", "landscape",
        "room", "kitchen", "bedroom", "bathroom", "office", "restaurant",
        "traffic", "crowd", "market", "station", "airport",
    ],
}


@dataclass
class PromptEntry:
    """Single prompt entry with metadata"""
    text: str
    category: str
    length: int
    source_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptEntry":
        return cls(
            text=str(data["text"]),
            category=str(data["category"]),
            length=int(data["length"]),
            source_id=data.get("source_id"),
        )


@dataclass
class EvaluationDataset:
    """Evaluation dataset containing prompts and metadata"""
    name: str
    prompts: List[PromptEntry]
    created_at: datetime = field(default_factory=datetime.now)
    source: str = "coco_captions"
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def get_prompts_by_category(self, category: str) -> List[PromptEntry]:
        """Get prompts filtered by category"""
        return [p for p in self.prompts if p.category == category]
    
    def get_prompt_texts(self) -> List[str]:
        """Get list of prompt text strings"""
        return [p.text for p in self.prompts]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prompts": [p.to_dict() for p in self.prompts],
            "created_at": self.created_at.isoformat(),
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        return cls(
            name=str(data["name"]),
            prompts=[PromptEntry.from_dict(p) for p in data["prompts"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            source=str(data.get("source", "coco_captions")),
        )


@dataclass
class DatasetStats:
    """Dataset statistics"""
    total_prompts: int
    category_distribution: Dict[str, int]
    length_distribution: Dict[str, int]  # "short", "medium", "long"
    avg_length: float
    min_length: int
    max_length: int
    keyword_frequency: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetStats":
        return cls(
            total_prompts=int(data["total_prompts"]),
            category_distribution=dict(data["category_distribution"]),
            length_distribution=dict(data["length_distribution"]),
            avg_length=float(data["avg_length"]),
            min_length=int(data["min_length"]),
            max_length=int(data["max_length"]),
            keyword_frequency=dict(data["keyword_frequency"]),
        )


class DatasetBuilder:
    """
    构建评测数据集
    
    Supports:
    - Loading COCO Captions data from JSON files
    - Filtering and sampling prompts
    - Categorizing prompts (simple_object, portrait, complex_scene)
    - Analyzing dataset statistics
    - Exporting to JSON format
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    
    # Length thresholds for categorization
    MIN_LENGTH = 10  # Minimum caption length to include
    MAX_LENGTH = 200  # Maximum caption length to include
    SHORT_THRESHOLD = 50
    MEDIUM_THRESHOLD = 100
    
    def __init__(self, coco_captions_path: Optional[str] = None):
        """
        初始化数据集构建器
        
        Args:
            coco_captions_path: COCO Captions JSON 文件路径
                               如果为 None，将使用内置示例数据
        """
        self.coco_captions_path = coco_captions_path
        self._raw_captions: List[Dict[str, Any]] = []
        self._loaded = False
        
        logger.info(f"DatasetBuilder 初始化: path={coco_captions_path}")
    
    def load_captions(self) -> int:
        """
        加载 COCO Captions 数据
        
        Returns:
            加载的 caption 数量
        
        Requirements: 9.1
        """
        if self.coco_captions_path and Path(self.coco_captions_path).exists():
            return self._load_from_file(self.coco_captions_path)
        else:
            # Use built-in sample data for testing/demo
            return self._load_sample_data()
    
    def _load_from_file(self, filepath: str) -> int:
        """Load captions from COCO format JSON file"""
        path = Path(filepath)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # COCO format: {"annotations": [{"caption": "...", "id": ...}, ...]}
        if "annotations" in data:
            self._raw_captions = [
                {"text": ann["caption"], "id": str(ann.get("id", i))}
                for i, ann in enumerate(data["annotations"])
            ]
        # Simple format: [{"caption": "..."}, ...] or ["caption1", ...]
        elif isinstance(data, list):
            if data and isinstance(data[0], str):
                self._raw_captions = [
                    {"text": cap, "id": str(i)}
                    for i, cap in enumerate(data)
                ]
            else:
                self._raw_captions = [
                    {"text": item.get("caption", item.get("text", "")), 
                     "id": str(item.get("id", i))}
                    for i, item in enumerate(data)
                ]
        else:
            raise ValueError(f"Unsupported COCO captions format in {filepath}")
        
        self._loaded = True
        logger.info(f"从文件加载 {len(self._raw_captions)} 条 captions")
        return len(self._raw_captions)
    
    def _load_sample_data(self) -> int:
        """Load built-in sample captions for testing"""
        # Sample captions covering different categories
        sample_captions = [
            # Simple objects
            "A red apple sitting on a wooden table",
            "A white cup filled with hot coffee",
            "A colorful book with a blue cover",
            "An orange basketball on the court",
            "A glass bottle of water on the desk",
            "A vintage clock hanging on the wall",
            "A ceramic vase with fresh flowers",
            "A leather bag placed on a chair",
            "A pair of running shoes by the door",
            "A small toy car on the floor",
            
            # Portraits
            "A young woman smiling at the camera",
            "A man in a suit walking down the street",
            "A group of children playing in the park",
            "A person sitting on a bench reading a book",
            "An elderly man with a white beard",
            "A baby sleeping peacefully in a crib",
            "Two people having a conversation at a cafe",
            "A woman standing near a window looking outside",
            "A boy riding a bicycle in the neighborhood",
            "A girl holding a colorful balloon",
            
            # Complex scenes
            "A busy city street with cars and pedestrians",
            "A beautiful sunset over the ocean with waves",
            "A cozy living room with a fireplace and sofa",
            "A mountain landscape covered in snow",
            "A crowded market with various food stalls",
            "A modern kitchen with stainless steel appliances",
            "A peaceful forest path with tall trees",
            "A beach scene with umbrellas and sunbathers",
            "An airport terminal with travelers and luggage",
            "A restaurant interior with tables and diners",
            
            # Mixed/Other
            "A cat sleeping on a comfortable couch",
            "A dog running through a green field",
            "A bird perched on a tree branch",
            "A plate of delicious pasta with tomato sauce",
            "A laptop computer on an office desk",
            "A bicycle parked against a brick wall",
            "A garden with colorful flowers blooming",
            "A train station platform with waiting passengers",
            "A coffee shop with customers and baristas",
            "A library with rows of bookshelves",
        ]
        
        self._raw_captions = [
            {"text": cap, "id": str(i)}
            for i, cap in enumerate(sample_captions)
        ]
        
        self._loaded = True
        logger.info(f"加载 {len(self._raw_captions)} 条示例 captions")
        return len(self._raw_captions)
    
    def _classify_caption(self, text: str) -> str:
        """
        Classify caption into category based on keywords
        
        Requirements: 9.2
        """
        text_lower = text.lower()
        
        # Count keyword matches for each category
        scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = score
        
        # Return category with highest score, default to "other"
        if max(scores.values()) > 0:
            return max(scores.keys(), key=lambda k: scores[k])
        return "other"
    
    def _filter_caption(self, text: str) -> bool:
        """
        Filter caption based on length and quality criteria
        
        Requirements: 9.3
        """
        # Length check
        if len(text) < self.MIN_LENGTH or len(text) > self.MAX_LENGTH:
            return False
        
        # Basic quality checks
        # - Not too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in " .,!?'-")
        if special_chars > len(text) * 0.2:
            return False
        
        # - Contains actual words
        words = text.split()
        if len(words) < 3:
            return False
        
        return True
    
    def _remove_duplicates(self, captions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or near-duplicate captions"""
        seen: Set[str] = set()
        unique = []
        
        for cap in captions:
            # Normalize for comparison
            normalized = cap["text"].lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(cap)
        
        return unique

    
    def build_evaluation_dataset(
        self,
        num_samples: int = 200,
        categories: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> EvaluationDataset:
        """
        从 COCO Captions 构建评测数据集
        
        Args:
            num_samples: 采样数量 (200-500)
            categories: 类别过滤 ["simple_object", "portrait", "complex_scene"]
                       如果为 None，包含所有类别
            seed: 随机种子，用于可复现的采样
        
        Returns:
            EvaluationDataset 实例
        
        Requirements: 9.1, 9.3
        """
        if not self._loaded:
            self.load_captions()
        
        if seed is not None:
            random.seed(seed)
        
        # Filter captions
        filtered = [
            cap for cap in self._raw_captions
            if self._filter_caption(cap["text"])
        ]
        
        # Remove duplicates
        filtered = self._remove_duplicates(filtered)
        
        logger.info(f"过滤后剩余 {len(filtered)} 条 captions")
        
        # Classify and create prompt entries
        prompts: List[PromptEntry] = []
        for cap in filtered:
            text = cap["text"]
            category = self._classify_caption(text)
            
            # Apply category filter if specified
            if categories and category not in categories:
                continue
            
            prompts.append(PromptEntry(
                text=text,
                category=category,
                length=len(text),
                source_id=cap.get("id"),
            ))
        
        # Sample if we have more than requested
        if len(prompts) > num_samples:
            prompts = random.sample(prompts, num_samples)
        
        logger.info(f"构建数据集: {len(prompts)} 条 prompts")
        
        return EvaluationDataset(
            name=f"eval_dataset_{num_samples}",
            prompts=prompts,
        )
    
    def analyze_dataset(self, dataset: EvaluationDataset) -> DatasetStats:
        """
        分析数据集统计信息
        
        Returns:
            DatasetStats: 长度分布、关键词频率、类别分布
        
        Requirements: 9.2, 9.4
        """
        if not dataset.prompts:
            return DatasetStats(
                total_prompts=0,
                category_distribution={},
                length_distribution={"short": 0, "medium": 0, "long": 0},
                avg_length=0.0,
                min_length=0,
                max_length=0,
                keyword_frequency={},
            )
        
        # Category distribution
        category_counts: Dict[str, int] = Counter(p.category for p in dataset.prompts)
        
        # Length distribution
        lengths = [p.length for p in dataset.prompts]
        length_dist = {
            "short": sum(1 for l in lengths if l < self.SHORT_THRESHOLD),
            "medium": sum(1 for l in lengths if self.SHORT_THRESHOLD <= l < self.MEDIUM_THRESHOLD),
            "long": sum(1 for l in lengths if l >= self.MEDIUM_THRESHOLD),
        }
        
        # Keyword frequency (top words)
        all_words: List[str] = []
        for prompt in dataset.prompts:
            # Extract words, lowercase, filter short words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', prompt.text.lower())
            all_words.extend(words)
        
        # Get top 50 keywords, excluding common stop words
        stop_words = {
            "the", "and", "with", "for", "that", "this", "from", "are",
            "was", "were", "been", "being", "have", "has", "had", "having",
        }
        word_counts = Counter(w for w in all_words if w not in stop_words)
        keyword_freq = dict(word_counts.most_common(50))
        
        stats = DatasetStats(
            total_prompts=len(dataset.prompts),
            category_distribution=dict(category_counts),
            length_distribution=length_dist,
            avg_length=sum(lengths) / len(lengths),
            min_length=min(lengths),
            max_length=max(lengths),
            keyword_frequency=keyword_freq,
        )
        
        logger.info(
            f"数据集分析: {stats.total_prompts} prompts, "
            f"avg_length={stats.avg_length:.1f}, "
            f"categories={list(stats.category_distribution.keys())}"
        )
        
        return stats
    
    def export_prompts(
        self,
        dataset: EvaluationDataset,
        output_path: str,
    ) -> Path:
        """
        导出 JSON 格式的 prompt 列表
        
        Args:
            dataset: 要导出的数据集
            output_path: 输出文件路径
        
        Returns:
            导出文件的路径
        
        Requirements: 9.5
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"导出数据集到 {path}")
        return path
    
    def export_stats_report(
        self,
        stats: DatasetStats,
        output_path: str,
    ) -> Path:
        """
        生成数据集统计报告
        
        Args:
            stats: 数据集统计信息
            output_path: 输出文件路径
        
        Returns:
            报告文件的路径
        
        Requirements: 9.5
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": stats.to_dict(),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"导出统计报告到 {path}")
        return path
    
    def load_dataset(self, filepath: str) -> EvaluationDataset:
        """
        从 JSON 文件加载已保存的数据集
        
        Args:
            filepath: JSON 文件路径
        
        Returns:
            EvaluationDataset 实例
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return EvaluationDataset.from_dict(data)
