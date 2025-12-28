"""
Tests for DatasetBuilder module.

Verifies COCO Captions loading, filtering, analysis, and export functionality.
Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.dataset.builder import (
    DatasetBuilder,
    EvaluationDataset,
    DatasetStats,
    PromptEntry,
)


class TestDatasetBuilder:
    """Test DatasetBuilder functionality"""
    
    def test_load_sample_data(self):
        """Test loading built-in sample captions"""
        builder = DatasetBuilder()
        count = builder.load_captions()
        
        assert count > 0
        assert builder._loaded is True
    
    def test_build_evaluation_dataset(self):
        """Test building evaluation dataset with sampling"""
        builder = DatasetBuilder()
        dataset = builder.build_evaluation_dataset(num_samples=20, seed=42)
        
        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset) <= 20
        assert len(dataset.prompts) > 0
        
        # Check prompt structure
        for prompt in dataset.prompts:
            assert isinstance(prompt, PromptEntry)
            assert len(prompt.text) > 0
            assert prompt.category in ["simple_object", "portrait", "complex_scene", "other"]
            assert prompt.length == len(prompt.text)
    
    def test_build_dataset_with_category_filter(self):
        """Test building dataset with category filtering"""
        builder = DatasetBuilder()
        dataset = builder.build_evaluation_dataset(
            num_samples=50,
            categories=["portrait"],
            seed=42,
        )
        
        # All prompts should be in portrait category
        for prompt in dataset.prompts:
            assert prompt.category == "portrait"
    
    def test_analyze_dataset(self):
        """Test dataset analysis functionality"""
        builder = DatasetBuilder()
        dataset = builder.build_evaluation_dataset(num_samples=30, seed=42)
        stats = builder.analyze_dataset(dataset)
        
        assert isinstance(stats, DatasetStats)
        assert stats.total_prompts == len(dataset)
        assert stats.avg_length > 0
        assert stats.min_length <= stats.avg_length <= stats.max_length
        
        # Check length distribution
        assert "short" in stats.length_distribution
        assert "medium" in stats.length_distribution
        assert "long" in stats.length_distribution
        
        # Check category distribution
        assert len(stats.category_distribution) > 0
    
    def test_export_prompts_json(self):
        """Test exporting dataset to JSON format"""
        builder = DatasetBuilder()
        dataset = builder.build_evaluation_dataset(num_samples=10, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_prompts.json"
            result_path = builder.export_prompts(dataset, str(output_path))
            
            assert result_path.exists()
            
            # Verify JSON content
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert "name" in data
            assert "prompts" in data
            assert len(data["prompts"]) == len(dataset)
    
    def test_export_stats_report(self):
        """Test exporting statistics report"""
        builder = DatasetBuilder()
        dataset = builder.build_evaluation_dataset(num_samples=10, seed=42)
        stats = builder.analyze_dataset(dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stats_report.json"
            result_path = builder.export_stats_report(stats, str(output_path))
            
            assert result_path.exists()
            
            # Verify JSON content
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert "generated_at" in data
            assert "statistics" in data
            assert data["statistics"]["total_prompts"] == stats.total_prompts
    
    def test_load_exported_dataset(self):
        """Test loading a previously exported dataset"""
        builder = DatasetBuilder()
        original = builder.build_evaluation_dataset(num_samples=10, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.json"
            builder.export_prompts(original, str(output_path))
            
            # Load it back
            loaded = builder.load_dataset(str(output_path))
            
            assert len(loaded) == len(original)
            assert loaded.name == original.name
            
            # Check prompts match
            for orig, load in zip(original.prompts, loaded.prompts):
                assert orig.text == load.text
                assert orig.category == load.category
    
    def test_deterministic_sampling(self):
        """Test that same seed produces same results"""
        builder1 = DatasetBuilder()
        builder2 = DatasetBuilder()
        
        dataset1 = builder1.build_evaluation_dataset(num_samples=15, seed=123)
        dataset2 = builder2.build_evaluation_dataset(num_samples=15, seed=123)
        
        assert len(dataset1) == len(dataset2)
        for p1, p2 in zip(dataset1.prompts, dataset2.prompts):
            assert p1.text == p2.text


class TestPromptEntry:
    """Test PromptEntry dataclass"""
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip"""
        entry = PromptEntry(
            text="A cat on a mat",
            category="simple_object",
            length=14,
            source_id="123",
        )
        
        data = entry.to_dict()
        restored = PromptEntry.from_dict(data)
        
        assert restored.text == entry.text
        assert restored.category == entry.category
        assert restored.length == entry.length
        assert restored.source_id == entry.source_id


class TestEvaluationDataset:
    """Test EvaluationDataset dataclass"""
    
    def test_get_prompts_by_category(self):
        """Test filtering prompts by category"""
        prompts = [
            PromptEntry("A red apple", "simple_object", 11),
            PromptEntry("A woman smiling", "portrait", 15),
            PromptEntry("A busy street", "complex_scene", 13),
        ]
        dataset = EvaluationDataset(name="test", prompts=prompts)
        
        portraits = dataset.get_prompts_by_category("portrait")
        assert len(portraits) == 1
        assert portraits[0].text == "A woman smiling"
    
    def test_get_prompt_texts(self):
        """Test getting list of prompt texts"""
        prompts = [
            PromptEntry("Text 1", "other", 6),
            PromptEntry("Text 2", "other", 6),
        ]
        dataset = EvaluationDataset(name="test", prompts=prompts)
        
        texts = dataset.get_prompt_texts()
        assert texts == ["Text 1", "Text 2"]


class TestDatasetStats:
    """Test DatasetStats dataclass"""
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip"""
        stats = DatasetStats(
            total_prompts=100,
            category_distribution={"portrait": 50, "other": 50},
            length_distribution={"short": 30, "medium": 50, "long": 20},
            avg_length=45.5,
            min_length=10,
            max_length=150,
            keyword_frequency={"cat": 10, "dog": 8},
        )
        
        data = stats.to_dict()
        restored = DatasetStats.from_dict(data)
        
        assert restored.total_prompts == stats.total_prompts
        assert restored.avg_length == stats.avg_length
        assert restored.category_distribution == stats.category_distribution
