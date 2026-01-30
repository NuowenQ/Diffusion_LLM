# Causal Diffusion Models with Disentangled Personality Encoders for Policy Simulation

This repository implements a novel framework for human behavior simulation that combines:
- **Disentangled Personality Encoding** using β-VAE and FactorVAE
- **Causal Structural Models (SCM)** with Pearl's do-calculus
- **Masked Diffusion Language Models (MDLM)** for text generation

## Key Innovations

1. **Disentangled Personality Encoder**: Maps Big Five personality traits to a low-dimensional latent space where each dimension corresponds to causally meaningful personality factors.

2. **Causal Structural Model**: Implements the pathway `Personality → Beliefs/Attitudes → Behavioral Intentions → Actions` with interventional capabilities.

3. **Diffusion-based Generation**: Uses bidirectional attention for global coherence and iterative denoising for diverse, personality-consistent responses.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# 1. Download and preprocess data
python scripts/download_data.py
python scripts/preprocess_data.py

# 2. Train personality encoder
python scripts/train_stage1.py

# 3. Train SCM layer
python scripts/train_stage2.py

# 4. Joint fine-tuning
python scripts/train_stage4.py

# 5. Generate text
python scripts/generate.py --personality openness=0.8,extraversion=0.6
```

## License

MIT License
