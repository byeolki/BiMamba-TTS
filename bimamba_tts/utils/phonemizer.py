import re
from typing import List

from g2p_en import G2p


class Phonemizer:
    def __init__(self):
        self.g2p = G2p()
        self.pad = "_"
        self.eos = "~"
        self.phonemes = [self.pad, self.eos]

        arpa = [
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "B",
            "CH",
            "D",
            "DH",
            "EH",
            "ER",
            "EY",
            "F",
            "G",
            "HH",
            "IH",
            "IY",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW",
            "OY",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH",
            "UW",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        ]

        self.phonemes.extend(arpa)
        for i in range(3):
            self.phonemes.extend([f"{p}{i}" for p in arpa])

        self.phoneme_to_id = {p: idx for idx, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {idx: p for idx, p in enumerate(self.phonemes)}

    def __len__(self):
        return len(self.phonemes)

    def text_to_sequence(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        phonemes = self.g2p(text)
        sequence = [self.phoneme_to_id.get(p, 0) for p in phonemes]
        sequence.append(self.phoneme_to_id[self.eos])
        return sequence

    def sequence_to_text(self, sequence: List[int]) -> str:
        phonemes = [self.id_to_phoneme.get(idx, self.pad) for idx in sequence]
        return " ".join(phonemes)

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s\'\-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
