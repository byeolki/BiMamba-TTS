# BiMamba-TTS

Bidirectional Mamba for fast and high-quality non-autoregressive text-to-speech synthesis.

## Description

BiMamba-TTS introduces bidirectional state space modeling to non-autoregressive text-to-speech synthesis. By processing input sequences in both forward and backward directions with Mamba blocks, the model captures richer contextual information for improved prosody and naturalness in synthesized speech.

The bidirectional scanning strategy, inspired by Vision Mamba, enables the model to leverage full-context information during parallel mel-spectrogram generation. This approach combines the efficiency benefits of Mamba's linear complexity with enhanced context modeling for better speech quality, particularly in long utterances and complex prosodic patterns.

This repository contains the research implementation developed for academic publication.

## Citation

FastSpeech2:
```bibtex
@inproceedings{ren2021fastspeech2,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

Mamba:
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

Vision Mamba:
```bibtex
@article{zhu2024vision,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}
```
