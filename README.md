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

### 1. Data Setup

Download and preprocess the PANDORA dataset from HuggingFace:

```bash
# Download PANDORA dataset (train/validation/test splits)
python scripts/download_data.py

# Preprocess: validate, compute statistics, tokenize
python scripts/preprocess_data.py --output-dir ./data
```

**Dataset Info:**
- Source: [Automated-Personality-Prediction](https://huggingface.co/datasets/Fatima0923/Automated-Personality-Prediction)
- Contains: Reddit comments with Big Five personality scores
- Splits: ~16k train, ~2k validation, ~2.8k test samples
- Big Five traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism (scored 0-99)

### 2. Training Pipeline (Multi-Stage)

The full training workflow consists of 4 stages:

#### Stage 1: Train Disentangled Personality Encoder
```bash
python scripts/train_stage1.py \
    --encoder_type beta_vae \
    --epochs 200 \
    --batch_size 256 \
    --output_dir experiments/stage1_personality_encoder
```
**Output:** `experiments/stage1_personality_encoder/best_model.pt`

#### Stage 2: Train Causal SCM Layer
```bash
python scripts/train_stage2.py \
    --encoder_path experiments/stage1_personality_encoder/best_model.pt \
    --epochs 100 \
    --output_dir experiments/stage2_causal_scm
```
**Output:** `experiments/stage2_causal_scm/best_model.pt`

#### Stage 3: Train MDLM with Fixed Personality Conditioning
```bash
python scripts/train_stage3.py \
    --causal_vae_path experiments/stage2_causal_scm/best_model.pt \
    --epochs 100 \
    --output_dir experiments/stage3_mdlm
```
**Output:** `experiments/stage3_mdlm/best_model.pt`

#### Stage 4: Joint Fine-tuning
```bash
python scripts/train_stage4.py \
    --mdlm_path experiments/stage3_mdlm/best_model.pt \
    --causal_vae_path experiments/stage2_causal_scm/best_model.pt \
    --epochs 50 \
    --output_dir experiments/stage4_joint
```
**Output:** `experiments/stage4_joint/best_model.pt`

### 3. Text Generation

Generate text conditioned on personality traits:

```bash
# Generate from specific personality profile
python scripts/generate.py \
    --model_path experiments/stage4_joint/best_model.pt \
    --personality 0.8,0.6,0.7,0.9,0.3

# Interactive mode
python scripts/generate.py \
    --model_path experiments/stage4_joint/best_model.pt \
    --interactive

# Counterfactual generation (intervention on extraversion)
python scripts/generate.py \
    --model_path experiments/stage4_joint/best_model.pt \
    --personality 0.5,0.5,0.5,0.5,0.5 \
    --intervene extraversion=0.9
```

## Configuration

All training scripts support command-line arguments. Use `--help` to see all options:

```bash
python scripts/train_stage1.py --help
python scripts/preprocess_data.py --help
```

Key configuration options:
- `--batch_size`: Batch size for training (default: 256)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 1e-3)
- `--tokenizer`: HuggingFace tokenizer name (default: gpt2)
- `--output_dir`: Directory to save checkpoints and logs

## Validation

To verify the entire workflow is set up correctly:

```bash
bash setup.sh
```

This will:
1. Install all dependencies
2. Verify project structure
3. Check Python syntax
4. Validate imports
5. Create required directories

See [WORKFLOW_VALIDATION.md](WORKFLOW_VALIDATION.md) for detailed validation checklist.

## License

MIT License
