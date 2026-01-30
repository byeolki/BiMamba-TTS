from typing import Dict, List

import torch


class BiMambaTTSCollate:
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict]) -> Dict:
        basenames = [item["basename"] for item in batch]
        text_lens = torch.LongTensor([item["text_len"] for item in batch])
        mel_lens = torch.LongTensor([item["mel_len"] for item in batch])

        max_text_len = max(text_lens)
        max_mel_len = max(mel_lens)

        batch_size = len(batch)
        n_mel_channels = batch[0]["mel"].size(-1)

        texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        mels = torch.zeros(batch_size, max_mel_len, n_mel_channels)
        durations = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        pitches = torch.zeros(batch_size, max_mel_len)
        energies = torch.zeros(batch_size, max_mel_len)

        for i, item in enumerate(batch):
            text_len = item["text_len"]
            mel_len = item["mel_len"]
            dur_len = len(item["duration"])

            actual_text_len = min(text_len, dur_len)

            texts[i, :text_len] = item["text"]
            mels[i, :mel_len] = item["mel"]
            durations[i, :actual_text_len] = item["duration"][:actual_text_len]
            pitches[i, :mel_len] = item["pitch"]
            energies[i, :mel_len] = item["energy"]

        src_masks = self.get_mask_from_lengths(text_lens, max_text_len)
        mel_masks = self.get_mask_from_lengths(mel_lens, max_mel_len)

        return {
            "basenames": basenames,
            "texts": texts,
            "text_lens": text_lens,
            "mels": mels,
            "mel_lens": mel_lens,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "src_masks": src_masks,
            "mel_masks": mel_masks,
        }

    def get_mask_from_lengths(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        batch_size = lengths.size(0)
        ids = (
            torch.arange(0, max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = ids < lengths.unsqueeze(1)
        return mask
