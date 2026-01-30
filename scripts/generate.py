"""
Generate text with personality conditioning.

Usage:
    # Generate from specific personality
    python scripts/generate.py \
        --model_path experiments/stage4_joint/best_model.pt \
        --personality 0.8,0.6,0.7,0.9,0.3

    # Interactive mode
    python scripts/generate.py --model_path experiments/stage4_joint/best_model.pt --interactive

    # Counterfactual generation (intervention)
    python scripts/generate.py \
        --model_path experiments/stage4_joint/best_model.pt \
        --personality 0.5,0.5,0.5,0.5,0.5 \
        --intervene extraversion=0.9
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from transformers import AutoTokenizer
from pathlib import Path

from src.models.personality.encoder import create_personality_encoder
from src.models.causal.scm_layer import CausalVAE
from src.models.diffusion.mdlm import MDLM
from src.models.diffusion.sampler import CFGSampler


BIG_FIVE_TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']


def parse_personality(personality_str: str) -> torch.Tensor:
    """Parse personality from string."""
    values = [float(x) for x in personality_str.split(',')]
    assert len(values) == 5, "Personality must have 5 values (O, C, E, A, N)"
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def parse_intervention(intervention_str: str) -> tuple:
    """Parse intervention string like 'extraversion=0.9'."""
    trait, value = intervention_str.split('=')
    trait_idx = BIG_FIVE_TRAITS.index(trait.lower())
    value = float(value)
    return trait_idx, value


def load_model(model_path: str, device: torch.device):
    """Load pretrained model."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load tokenizer
    tokenizer_name = checkpoint['args'].get('tokenizer', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        mask_token_id = tokenizer.mask_token_id
    
    # Recreate CausalVAE
    if 'stage4' in model_path or 'joint' in model_path:
        # Joint model
        encoder_args = checkpoint['args']
        personality_encoder = create_personality_encoder(
            encoder_type=encoder_args.get('encoder_type', 'beta_vae'),
            input_dim=5,
            hidden_dims=encoder_args.get('hidden_dims', [128, 256, 256, 128]),
            latent_dim=encoder_args.get('latent_dim', 64),
            beta=encoder_args.get('beta', 4.0),
            gamma=encoder_args.get('gamma'),
            dropout=encoder_args.get('dropout', 0.1),
        )
        
        causal_vae = CausalVAE(personality_encoder, scm_config=None)
        causal_vae.load_state_dict(checkpoint['causal_vae_state_dict'])
        
        # Recreate MDLM
        config = checkpoint['config']
        mdlm = MDLM(config, mask_token_id)
        mdlm.load_state_dict(checkpoint['mdlm_state_dict'])
    else:
        # Stage 3 model
        config = checkpoint['config']
        mdlm = MDLM(config, mask_token_id)
        
        # Handle CFG wrapper
        if 'module.model.token_embedding.weight' in checkpoint['model_state_dict']:
            model_dict = {}
            for k, v in checkpoint['model_state_dict'].items():
                if k.startswith('module.model.'):
                    model_dict[k.replace('module.model.', '')] = v
                elif k.startswith('model.'):
                    model_dict[k.replace('model.', '')] = v
            mdlm.load_state_dict(model_dict)
        else:
            mdlm.load_state_dict(checkpoint['model_state_dict'])
        
        # Load separate CausalVAE
        causal_vae_path = Path(model_path).parent.parent / 'stage2_causal_scm' / 'best_model.pt'
        if not causal_vae_path.exists():
            raise FileNotFoundError(f"CausalVAE not found at {causal_vae_path}")
        
        causal_vae_checkpoint = torch.load(causal_vae_path, map_location=device)
        encoder_args = causal_vae_checkpoint['args']
        personality_encoder = create_personality_encoder(
            encoder_type=encoder_args['encoder_type'],
            input_dim=5,
            hidden_dims=encoder_args['hidden_dims'],
            latent_dim=encoder_args['latent_dim'],
            beta=encoder_args['beta'],
            gamma=encoder_args.get('gamma'),
            dropout=encoder_args['dropout'],
        )
        
        causal_vae = CausalVAE(personality_encoder, scm_config=None)
        causal_vae.load_state_dict(causal_vae_checkpoint['model_state_dict'])
    
    # Move to device and eval mode
    causal_vae = causal_vae.to(device).eval()
    mdlm = mdlm.to(device).eval()
    
    print(f"Model loaded successfully!")
    return causal_vae, mdlm, tokenizer


@torch.no_grad()
def generate(
    causal_vae,
    mdlm,
    tokenizer,
    personality: torch.Tensor,
    device: torch.device,
    intervention_dim: int = None,
    intervention_value: float = None,
    num_samples: int = 1,
    seq_len: int = 128,
    temperature: float = 1.0,
    guidance_scale: float = 2.0,
    num_steps: int = 50,
    sampler_type: str = 'ancestral',
):
    """Generate text conditioned on personality."""
    personality = personality.to(device)
    
    # Expand for multiple samples
    if num_samples > 1:
        personality = personality.repeat(num_samples, 1)
    
    # Get causal personality embedding
    if intervention_dim is not None:
        # Apply intervention
        z_exo = causal_vae.personality_encoder.encode_personality(personality)
        intervention_dims = torch.tensor([intervention_dim], device=device)
        intervention_values = torch.full(
            (personality.size(0), 1),
            intervention_value,
            device=device
        )
        personality_cond = causal_vae.scm_layer.do_intervention(
            z_exo,
            intervention_dims,
            intervention_values,
        )
    else:
        # Standard generation
        causal_output = causal_vae(personality, return_all=False)
        personality_cond = causal_output['z_causal']
    
    # Create sampler
    sampler = CFGSampler(
        mdlm,
        guidance_scale=guidance_scale,
        sampler_type=sampler_type,
        num_steps=num_steps,
    )
    
    # Generate
    generated_ids = sampler.sample(
        personality_cond,
        seq_len,
        temperature=temperature,
    )
    
    # Decode
    texts = []
    for i in range(num_samples):
        text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        texts.append(text)
    
    return texts


def interactive_mode(causal_vae, mdlm, tokenizer, device):
    """Interactive generation mode."""
    print("\n" + "="*60)
    print("Interactive Generation Mode")
    print("="*60)
    print("\nEnter personality traits (0-1 scale):")
    print("Traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism")
    print("\nCommands:")
    print("  /quit - Exit")
    print("  /help - Show this help")
    print("  /defaults - Use default personality (0.5, 0.5, 0.5, 0.5, 0.5)")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\nPersonality (O,C,E,A,N) or command: ").strip()
            
            if user_input == '/quit':
                print("Goodbye!")
                break
            elif user_input == '/help':
                print("\nEnter 5 comma-separated values (0-1) for personality traits.")
                print("Example: 0.8,0.6,0.7,0.9,0.3")
                continue
            elif user_input == '/defaults':
                personality_str = "0.5,0.5,0.5,0.5,0.5"
            else:
                personality_str = user_input
            
            # Parse personality
            personality = parse_personality(personality_str)
            
            # Get generation parameters
            num_samples = int(input("Number of samples (default: 1): ") or "1")
            seq_len = int(input("Sequence length (default: 128): ") or "128")
            temperature = float(input("Temperature (default: 1.0): ") or "1.0")
            guidance = float(input("Guidance scale (default: 2.0): ") or "2.0")
            
            # Generate
            print("\nGenerating...")
            texts = generate(
                causal_vae,
                mdlm,
                tokenizer,
                personality,
                device,
                num_samples=num_samples,
                seq_len=seq_len,
                temperature=temperature,
                guidance_scale=guidance,
            )
            
            # Display results
            print("\n" + "-"*60)
            print(f"Personality: {personality_str}")
            for i, text in enumerate(texts):
                print(f"\nSample {i+1}:")
                print(text)
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


def main(args):
    """Main function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    causal_vae, mdlm, tokenizer = load_model(args.model_path, device)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(causal_vae, mdlm, tokenizer, device)
    else:
        # Single generation
        if args.personality is None:
            print("Error: --personality required in non-interactive mode")
            return
        
        personality = parse_personality(args.personality)
        
        # Parse intervention if provided
        intervention_dim = None
        intervention_value = None
        if args.intervene:
            intervention_dim, intervention_value = parse_intervention(args.intervene)
            print(f"\nApplying intervention: {BIG_FIVE_TRAITS[intervention_dim]} = {intervention_value}")
        
        print(f"\nGenerating {args.num_samples} sample(s)...")
        print(f"Personality: {args.personality}")
        print(f"Parameters: seq_len={args.seq_len}, temp={args.temperature}, guidance={args.guidance_scale}")
        
        texts = generate(
            causal_vae,
            mdlm,
            tokenizer,
            personality,
            device,
            intervention_dim=intervention_dim,
            intervention_value=intervention_value,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            temperature=args.temperature,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            sampler_type=args.sampler,
        )
        
        # Display results
        print("\n" + "="*60)
        print("GENERATED TEXT")
        print("="*60)
        for i, text in enumerate(texts):
            print(f"\nSample {i+1}:")
            print(text)
            print("-"*60)
        
        # Save if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(f"Personality: {args.personality}\n")
                if args.intervene:
                    f.write(f"Intervention: {args.intervene}\n")
                f.write(f"\n{'='*60}\n\n")
                
                for i, text in enumerate(texts):
                    f.write(f"Sample {i+1}:\n{text}\n\n")
            
            print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text with personality conditioning')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    
    # Personality
    parser.add_argument('--personality', type=str,
                       help='Personality as comma-separated values (O,C,E,A,N)')
    parser.add_argument('--intervene', type=str,
                       help='Intervention: trait=value (e.g., extraversion=0.9)')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--sampler', type=str, default='ancestral',
                       choices=['ddpm', 'ddim', 'ancestral'],
                       help='Sampling strategy')
    
    # Mode
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--output', type=str,
                       help='Save output to file')
    
    args = parser.parse_args()
    main(args)
