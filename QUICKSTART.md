# Quick Start Guide

## Installation

```bash
cd /home/nuowen/research2/Diffusion_LLM

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Test the Implementation

### 1. Test Data Loader
```bash
python -m src.data.pandora_dataset
```

Expected output:
```
Loading PANDORA dataset (split: train)...
Loaded 16000 samples
✓ Dataset loader test passed!
```

### 2. Test Personality Encoder
```bash
python -m src.models.personality.encoder
```

Expected output:
```
Testing Disentangled Personality Encoder...
✓ Personality encoder test passed!
```

### 3. Test SCM Layer
```bash
python -m src.models.causal.scm_layer
```

Expected output:
```
Testing Causal SCM Layer...
✓ Causal SCM Layer test passed!
```

### 4. Test Conditioning Mechanisms
```bash
python -m src.models.conditioning.adaln
python -m src.models.conditioning.cross_attention
python -m src.models.conditioning.cfg
```

## Train Stage 1: Personality Encoder

### Quick Test (Small-scale)
```bash
python scripts/train_stage1.py \
    --encoder_type beta_vae \
    --epochs 5 \
    --batch_size 64 \
    --output_dir experiments/test_run
```

### Full Training (Recommended)
```bash
python scripts/train_stage1.py \
    --encoder_type beta_vae \
    --latent_dim 64 \
    --beta 4.0 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --anneal_epochs 150 \
    --output_dir experiments/stage1_personality_encoder
```

Monitor training:
- Checkpoints saved every 20 epochs to `experiments/stage1_personality_encoder/`
- Best model saved as `best_model.pt`

### Training with FactorVAE (Better Disentanglement)
```bash
python scripts/train_stage1.py \
    --encoder_type factor_vae \
    --beta 4.0 \
    --gamma 1.0 \
    --epochs 200 \
    --output_dir experiments/stage1_factor_vae
```

## Next Steps

After Stage 1 completes, you can:

1. **Visualize Latent Space**: Analyze disentanglement of Big Five traits
2. **Train Stage 2**: SCM layer on top of frozen encoder (script pending)
3. **Evaluate Disentanglement**: Compute DCI, MIG, FactorVAE scores

## Troubleshooting

### Out of Memory Error
- Reduce `--batch_size` (try 128 or 64)
- Reduce `--max_length` (try 256)

### Dataset Download Issues
- Check internet connection
- Dataset auto-downloads from HuggingFace
- Cache location: `~/.cache/huggingface/datasets/`

### Import Errors
- Ensure you ran `pip install -e .`
- Check Python version (requires Python 3.8+)

## File Overview

```
Key Files:
├── src/data/pandora_dataset.py              # Dataset loader
├── src/models/personality/encoder.py        # β-VAE & FactorVAE
├── src/models/causal/scm_layer.py          # Causal SCM with do-calculus
├── src/models/conditioning/adaln.py         # Adaptive LayerNorm
├── src/models/conditioning/cross_attention.py  # Cross-attention
├── src/models/conditioning/cfg.py           # Classifier-free guidance
└── scripts/train_stage1.py                  # Training script
```

## Expected Results (Stage 1)

After 200 epochs of training:
- **Reconstruction Loss**: < 0.01 (MSE on normalized Big Five)
- **KL Divergence**: 10-30 (depends on β and capacity annealing)
- **Total Loss**: 0.5-2.0

Good disentanglement indicators:
- Each latent dimension responds primarily to one Big Five trait
- Smooth interpolation in latent space
- Consistent personality-text mapping

## Support

For issues or questions:
1. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. Review test outputs for error messages
3. Open a GitHub issue with full error log
