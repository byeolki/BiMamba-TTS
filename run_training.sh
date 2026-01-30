#!/bin/bash

set -e

export PATH="/root/.local/bin:$PATH"

poetry run python bimamba_tts/cli/train.py --config configs/config.yaml

CHECKPOINT_DIR="experiments/bimamba-tts/checkpoints"
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.pt"

if [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "Error: checkpoint not found"
    exit 1
fi

OUTPUT_DIR="experiments/bimamba-tts/outputs"
mkdir -p "$OUTPUT_DIR"

SENTENCES=(
    "The quick brown fox jumps over the lazy dog."
    "Hello world, this is a test of the BiMamba TTS model."
    "Speech synthesis is the artificial production of human speech."
)

for i in "${!SENTENCES[@]}"; do
    TEXT="${SENTENCES[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/result_sample_${i}.wav"

    poetry run python bimamba_tts/cli/inference.py \
        --config configs/config.yaml \
        --checkpoint "$LATEST_CHECKPOINT" \
        --text "$TEXT" \
        --output "$OUTPUT_FILE" \
        --vocoder_checkpoint pretrained/hifigan_vocoder_v3_ljspeech.pth > /dev/null 2>&1
done

cp "${OUTPUT_DIR}/result_sample_0.wav" "${OUTPUT_DIR}/result.wav"

echo "Done: ${OUTPUT_DIR}/result.wav"
