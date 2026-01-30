import os
import subprocess
from typing import List, Tuple

import textgrid


class MontrealForcedAligner:
    def __init__(
        self,
        corpus_dir: str,
        dict_path: str,
        acoustic_model: str = "english_us_arpa",
        output_dir: str = None,
    ):
        self.corpus_dir = corpus_dir
        self.dict_path = dict_path
        self.acoustic_model = acoustic_model
        self.output_dir = output_dir or os.path.join(corpus_dir, "aligned")

    def align(self):
        cmd = [
            "mfa",
            "align",
            self.corpus_dir,
            self.dict_path,
            self.acoustic_model,
            self.output_dir,
            "--clean",
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Alignment completed. Output saved to {self.output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"MFA alignment failed: {e}")
            raise

    def extract_durations(self, textgrid_path: str) -> List[Tuple[str, float]]:
        tg = textgrid.TextGrid.fromFile(textgrid_path)

        durations = []
        for tier in tg:
            if tier.name == "phones":
                for interval in tier:
                    phone = interval.mark
                    duration = interval.maxTime - interval.minTime
                    if phone and phone != "":
                        durations.append((phone, duration))

        return durations

    def get_phoneme_durations(
        self, textgrid_path: str, hop_length: int, sampling_rate: int
    ) -> List[int]:
        durations = self.extract_durations(textgrid_path)

        frame_durations = []
        for phone, duration in durations:
            n_frames = int(duration * sampling_rate / hop_length + 0.5)
            frame_durations.append(n_frames)

        return frame_durations
