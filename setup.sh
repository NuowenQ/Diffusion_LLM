#!/bin/bash
# Complete Training Workflow Setup and Verification
# This script ensures all dependencies are installed and the training pipeline can run

set -e

echo "=========================================="
echo "Diffusion LLM Training Workflow Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Install Python dependencies
echo -e "\n${BLUE}[1/5] Installing Python dependencies...${NC}"
pip install -r requirements.txt
pip install -e .
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 2: Verify project structure
echo -e "\n${BLUE}[2/5] Verifying project structure...${NC}"
required_files=(
    "scripts/download_data.py"
    "scripts/preprocess_data.py"
    "scripts/train_stage1.py"
    "scripts/train_stage2.py"
    "scripts/train_stage3.py"
    "scripts/train_stage4.py"
    "scripts/generate.py"
    "src/data/pandora_dataset.py"
    "src/data/pandora_mdlm_dataloader.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${YELLOW}✗${NC} $file (MISSING)"
        exit 1
    fi
done
echo -e "${GREEN}✓ All required files present${NC}"

# Step 3: Verify Python syntax
echo -e "\n${BLUE}[3/5] Verifying Python syntax...${NC}"
python -m py_compile scripts/download_data.py
python -m py_compile scripts/preprocess_data.py
python -m py_compile scripts/train_stage1.py
echo -e "${GREEN}✓ Python syntax valid${NC}"

# Step 4: Verify imports work
echo -e "\n${BLUE}[4/5] Verifying imports...${NC}"
python -c "from src.data.pandora_dataset import PANDORADataset, create_dataloaders; print('✓ Data module imports work')"
python -c "from src.models.personality.encoder import create_personality_encoder; print('✓ Personality encoder imports work')"
python -c "from src.models.causal.scm_layer import CausalVAE; print('✓ Causal SCM imports work')"
echo -e "${GREEN}✓ All imports verified${NC}"

# Step 5: Create output directories
echo -e "\n${BLUE}[5/5] Creating output directories...${NC}"
mkdir -p experiments/stage1_personality_encoder
mkdir -p experiments/stage2_causal_scm
mkdir -p experiments/stage3_mdlm
mkdir -p experiments/stage4_joint
mkdir -p data
echo -e "${GREEN}✓ Output directories created${NC}"

echo -e "\n${GREEN}=========================================="
echo "Setup Complete! Ready to run training."
echo "=========================================="
echo ""
echo -e "${BLUE}Quick Start Guide:${NC}"
echo ""
echo "1. Download and preprocess data:"
echo "   python scripts/download_data.py"
echo "   python scripts/preprocess_data.py"
echo ""
echo "2. Train personality encoder (Stage 1):"
echo "   python scripts/train_stage1.py --epochs 10"
echo ""
echo "3. Train causal SCM layer (Stage 2):"
echo "   python scripts/train_stage2.py --encoder_path experiments/stage1_personality_encoder/best_model.pt"
echo ""
echo "4. Train MDLM with fixed personality (Stage 3):"
echo "   python scripts/train_stage3.py --causal_vae_path experiments/stage2_causal_scm/best_model.pt"
echo ""
echo "5. Joint fine-tuning (Stage 4):"
echo "   python scripts/train_stage4.py"
echo ""
echo "6. Generate text with personality conditioning:"
echo "   python scripts/generate.py --model_path experiments/stage4_joint/best_model.pt --personality 0.8,0.6,0.7,0.9,0.3"
echo ""
