"""
Dataset module for evaluation dataset construction.

Provides DatasetBuilder for loading, filtering, and analyzing
COCO Captions data for benchmark evaluation.
"""

from .builder import DatasetBuilder, EvaluationDataset, DatasetStats

__all__ = ["DatasetBuilder", "EvaluationDataset", "DatasetStats"]
