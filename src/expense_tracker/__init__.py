"""Receipt-driven expense tracker: OCR, field parsing and hybrid auto-categorization."""
from .categorizer import HybridCategorizer, MLCategorizer, RuleCategorizer
from .models import Expense, OcrLine, Prediction, ReceiptFields
from .pipeline import InvalidImageError, PipelineResult, ReceiptPipeline
from .preprocess import PreprocessConfig

__all__ = ["HybridCategorizer", "MLCategorizer", "RuleCategorizer", "Expense", "OcrLine", "Prediction",
           "ReceiptFields", "InvalidImageError", "PipelineResult", "ReceiptPipeline", "PreprocessConfig"]
__version__ = "1.0.0"
