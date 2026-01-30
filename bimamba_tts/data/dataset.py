import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(self, preprocessed_dir: str, split: str = "train", sort: bool = True):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split

        metadata_path = self.preprocessed_dir / f"{split}.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        if sort:
            self.metadata = sorted(self.metadata, key=lambda x: x["text_len"])

        stats_path = self.preprocessed_dir / "stats.json"
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        item = self.metadata[idx]
        basename = item["basename"]

        text = np.load(self.preprocessed_dir / "text" / f"{basename}.npy")
        mel = np.load(self.preprocessed_dir / "mel" / f"{basename}.npy")
        duration = np.load(self.preprocessed_dir / "duration" / f"{basename}.npy")
        pitch = np.load(self.preprocessed_dir / "pitch" / f"{basename}.npy")
        energy = np.load(self.preprocessed_dir / "energy" / f"{basename}.npy")

        return {
            "basename": basename,
            "text": torch.LongTensor(text),
            "mel": torch.FloatTensor(mel),
            "duration": torch.LongTensor(duration),
            "pitch": torch.FloatTensor(pitch),
            "energy": torch.FloatTensor(energy),
            "text_len": len(text),
            "mel_len": len(mel),
        }
