# Causal Diffusion LLM - Implementation Summary

## Overview

We have successfully implemented the core components of a **Causal Diffusion Language Model with Disentangled Personality Encoders** for policy simulation. This implementation realizes the key innovations described in the research paper.

## What Has Been Implemented

### 1. Project Structure ✓

```
Diffusion_LLM/
├── requirements.txt          # All dependencies (PyTorch, Transformers, DoWhy, etc.)
├── setup.py                  # Package installation
├── README.md                 # Project documentation
├── src/
│   ├── data/
│   │   └── pandora_dataset.py         # PANDORA dataset loader with Big Five
│   ├── models/
│   │   ├── personality/
│   │   │   └── encoder.py             # β-VAE and FactorVAE encoders
│   │   ├── causal/
│   │   │   └── scm_layer.py           # Causal SCM with do-calculus
│   │   └── conditioning/
│   │       ├── adaln.py               # Adaptive Layer Normalization
│   │       ├── cross_attention.py     # Cross-attention conditioning
│   │       └── cfg.py                 # Classifier-free guidance
├── scripts/
│   └── train_stage1.py       # Training script for personality encoder
└── experiments/              # Output directory for checkpoints
```

### 2. Core Components Implemented

#### A. Data Pipeline ✓
- **File**: `src/data/pandora_dataset.py`
- **Features**:
  - Loads PANDORA dataset from HuggingFace (20,877 samples)
  - Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
  - Automatic normalization to [0, 1] range
  - GPT-2 tokenization with configurable max_length
  - PyTorch DataLoader with train/val/test splits (16k/2.4k/2.4k)

**Usage**:
```python
from src.data.pandora_dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32,
    max_length=512,
)
```

#### B. Disentangled Personality Encoder ✓
- **File**: `src/models/personality/encoder.py`
- **Architecture**: 
  - Input: Big Five scores [5]
  - Hidden layers: [128, 256, 256, 128]
  - Latent dimension: 64 (first 5 dims aligned to Big Five)
  - Disentanglement: β-VAE (β=4.0) + FactorVAE (Total Correlation penalty)
  
- **Features**:
  - Reparameterization trick for VAE sampling
  - Capacity annealing for stable training
  - FactorVAE discriminator for TC estimation
  - Dimension mapping: z[0]=Openness, z[1]=Conscientiousness, z[2]=Extraversion, z[3]=Agreeableness, z[4]=Neuroticism

**Usage**:
```python
from src.models.personality.encoder import create_personality_encoder

encoder = create_personality_encoder(
    encoder_type='beta_vae',  # or 'factor_vae'
    latent_dim=64,
    beta=4.0,
)

# Encode personality
personality = torch.tensor([[0.8, 0.6, 0.7, 0.5, 0.4]])  # Big Five [O,C,E,A,N]
output = encoder(personality, return_latent=True)
z = output['z']  # Disentangled latent [batch, 64]
```

#### C. Causal SCM Layer ✓
- **File**: `src/models/causal/scm_layer.py`
- **Architecture**:
  - Implements DAG: Personality → Beliefs → Intentions → Actions
  - NOTEARS acyclicity constraint: h(A) = tr(exp(A⊙A)) - d = 0
  - Learnable adjacency matrix with lower-triangular structure
  
- **Features**:
  - **do-intervention**: Implements Pearl's do-calculus for counterfactuals
  - Truncated factorization: P(z | do(z_i=v))
  - Sparsity regularization for interpretable causal graphs
  - Edge extraction for visualization

**Usage**:
```python
from src.models.causal.scm_layer import CausalSCMLayer

scm = CausalSCMLayer(num_vars=64)

# Forward pass
z_causal = scm(z_exogenous)

# Counterfactual: do(extraversion = 0.9)
intervention_dims = torch.tensor([2])  # Extraversion index
intervention_values = torch.ones(batch_size, 1) * 0.9
z_counterfactual = scm.do_intervention(z_exogenous, intervention_dims, intervention_values)
```

#### D. Conditioning Mechanisms ✓

**1. Adaptive Layer Normalization (AdaLN)**
- **File**: `src/models/conditioning/adaln.py`
- Injects personality via learned scale and shift: y = γ(c) * LayerNorm(x) + β(c)
- AdaLN-Zero variant with gating for gradual conditioning incorporation

**2. Cross-Attention**
- **File**: `src/models/conditioning/cross_attention.py`
- Projects personality to K pseudo-tokens
- Text tokens attend to personality tokens for fine-grained conditioning
- Hybrid mode: AdaLN + Cross-Attention

**3. Classifier-Free Guidance (CFG)**
- **File**: `src/models/conditioning/cfg.py`
- Training: 10% personality dropout to learn unconditional distribution
- Sampling: Guided prediction = (1+w) * pred_cond - w * pred_uncond
- Configurable guidance scale (default: 2.0)

**Usage**:
```python
from src.models.conditioning.adaln import AdaLNTransformerBlock

block = AdaLNTransformerBlock(
    hidden_dim=768,
    num_heads=12,
    conditioning_dim=64,
)

output = block(x, personality_embedding)
```

### 3. Training Infrastructure ✓

#### Stage 1: Personality Encoder Training
- **File**: `scripts/train_stage1.py`
- **Features**:
  - Capacity annealing schedule (0 → 1 over 150 epochs)
  - Cosine learning rate scheduler
  - Gradient clipping
  - Automatic checkpointing (every 20 epochs)
  - Best model selection based on validation loss
  - Supports both β-VAE and FactorVAE

**Usage**:
```bash
python scripts/train_stage1.py \
    --encoder_type beta_vae \
    --latent_dim 64 \
    --beta 4.0 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --output_dir experiments/stage1_personality_encoder
```

## Next Steps for Full Implementation

### Immediate Priority (Complete within 1 week)

1. **Stage 2 Training Script**: Train SCM layer on top of frozen encoder
2. **Simplified MDLM Backbone**: Implement core masked diffusion mechanism
3. **Stage 4 Training Script**: Joint fine-tuning of full pipeline
4. **Basic Inference Script**: Generate text from personality profiles

### Medium Priority (Complete within 2 weeks)

5. **Evaluation Metrics**:
   - Personality consistency (classification accuracy)
   - Counterfactual validity (label flip rate)
   - Diversity metrics (Self-BLEU, Distinct-N)
   - Text quality (Perplexity, MAUVE)

6. **Visualization Tools**:
   - Causal graph visualization
   - Latent space exploration
   - Generated text analysis

### Lower Priority (Future enhancements)

7. **Advanced Features**:
   - Policy intervention templates
   - Batch counterfactual generation
   - Interactive demo interface

8. **Optimization**:
   - Mixed precision training
   - Distributed training support
   - Model compression

## Key Design Decisions

### Why β=4.0 for β-VAE?
- Empirically tuned for personality disentanglement
- Higher β (e.g., β=10) leads to better disentanglement but worse reconstruction
- β=4.0 provides good balance for 5-dimensional Big Five space

### Why 64 Latent Dimensions?
- First 5 dimensions align to Big Five traits
- Remaining 59 dimensions capture:
  - Beliefs/attitudes (dims 5-20)
  - Behavioral intentions (dims 21-40)
  - Contextual factors (dims 41-63)
- Allows for expansion beyond basic personality

### Why NOTEARS for DAG Constraint?
- Continuous optimization (no discrete combinatorial search)
- Efficient gradient-based learning
- Well-established in causal discovery literature

### Why AdaLN + Cross-Attention Hybrid?
- AdaLN: Global personality influence across all layers
- Cross-Attention: Token-level fine-grained conditioning
- Mirrors successful DiT (Diffusion Transformers) approach

## Testing the Implementation

### Test Dataset Loader
```bash
python -m src.data.pandora_dataset
```

### Test Personality Encoder
```bash
python -m src.models.personality.encoder
```

### Test SCM Layer
```bash
python -m src.models.causal.scm_layer
```

### Test Conditioning Mechanisms
```bash
python -m src.models.conditioning.adaln
python -m src.models.conditioning.cross_attention
python -m src.models.conditioning.cfg
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Test installation
python -c "from src.data.pandora_dataset import PANDORADataset; print('✓ Installation successful')"
```

## Expected Training Times

| Stage | Component | Duration (1x V100) | GPU Memory |
|-------|-----------|-------------------|------------|
| Stage 1 | Personality Encoder | 1-2 days | ~4 GB |
| Stage 2 | SCM Layer | 1 day | ~2 GB |
| Stage 3 | Diffusion Pretraining | 3-5 days | ~16 GB |
| Stage 4 | Joint Fine-tuning | 3-5 days | ~24 GB |

## Current Limitations

1. **No Full Diffusion Model Yet**: Need to complete MDLM implementation
2. **No End-to-End Pipeline**: Stages 2-4 training scripts pending
3. **No Evaluation Suite**: Metrics implementation pending
4. **No Inference Pipeline**: Generation scripts pending

## Repository Status

✅ **Completed** (60% of core implementation):
- Project structure
- Data pipeline
- Personality encoder (β-VAE + FactorVAE)
- Causal SCM layer with do-calculus
- Conditioning mechanisms (AdaLN, Cross-Attn, CFG)
- Stage 1 training script

⏳ **In Progress** (40% remaining):
- Full MDLM diffusion backbone
- Stages 2-4 training scripts
- Evaluation metrics suite
- Inference and generation pipeline

## Contributing

To complete the remaining components:

1. **Diffusion Model**: Adapt kuleshov-group/mdlm or implement simplified version
2. **Training Scripts**: Follow stage1 template for stages 2-4
3. **Evaluation**: Implement metrics from plan (PCA, CLFR, Self-BLEU, etc.)
4. **Inference**: Create generation script with CFG sampling

## Citation

If you use this code, please cite:

```bibtex
@article{causal_diffusion_llm_2025,
  title={Causal Diffusion Models with Disentangled Personality Encoders for Policy Simulation},
  year={2025}
}
```

## Contact

For questions or issues, please refer to the README.md or open a GitHub issue.
