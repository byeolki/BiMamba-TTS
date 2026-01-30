from .inference import main as inference_main
from .preprocess import main as preprocess_main
from .train import main as train_main

__all__ = ["train_main", "inference_main", "preprocess_main"]
