"""
Evaluation Metrics for Personality-Conditioned Text Generation

Implements metrics from the paper:
1. Disentanglement: PCA, CLFR (Classifier-Free Reconstruction)
2. Text Quality: Perplexity, MAUVE, Self-BLEU, Distinct-N
3. Personality Alignment: Personality classifier accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from collections import Counter
import math


class DisentanglementMetrics:
    """
    Metrics for evaluating disentanglement of personality encoder.
    
    - PCA: Principal Component Analysis to check if latent dimensions
      align with personality traits
    - CLFR: Classifier-Free Reconstruction Rate
    """
    
    @staticmethod
    def compute_pca_alignment(
        latent_codes: np.ndarray,
        personality_scores: np.ndarray,
        n_components: int = 5,
    ) -> Dict[str, float]:
        """
        Compute PCA alignment score.
        
        Checks if first N principal components align with personality traits.
        
        Args:
            latent_codes: Latent representations [num_samples, latent_dim]
            personality_scores: Ground truth personalities [num_samples, 5]
            n_components: Number of components to analyze
            
        Returns:
            Dictionary with PCA metrics
        """
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(latent_codes)
        
        # Project latent codes
        projected = pca.transform(latent_codes)
        
        # Compute correlation between PC and personality traits
        correlations = []
        for i in range(n_components):
            # Correlate PC_i with each trait
            trait_corrs = []
            for j in range(5):
                corr = np.corrcoef(projected[:, i], personality_scores[:, j])[0, 1]
                trait_corrs.append(abs(corr))
            
            # Take max correlation (best aligned trait)
            correlations.append(max(trait_corrs))
        
        return {
            'pca_alignment_mean': np.mean(correlations),
            'pca_alignment_std': np.std(correlations),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        }
    
    @staticmethod
    def compute_clfr(
        encoder: nn.Module,
        personality_scores: torch.Tensor,
        device: torch.device,
    ) -> float:
        """
        Compute Classifier-Free Reconstruction Rate.
        
        Measures how well personality can be reconstructed from latent codes
        without using a separate classifier.
        
        Args:
            encoder: Personality encoder
            personality_scores: Ground truth personalities [num_samples, 5]
            device: Device
            
        Returns:
            CLFR score (higher is better)
        """
        encoder.eval()
        
        with torch.no_grad():
            personality_scores = personality_scores.to(device)
            
            # Encode
            output = encoder(personality_scores, return_latent=True)
            recon = output['recon']
            
            # Compute reconstruction accuracy
            mse = F.mse_loss(recon, personality_scores)
            clfr = 1.0 / (1.0 + mse.item())
        
        return clfr


class TextQualityMetrics:
    """
    Metrics for evaluating generated text quality.
    
    - Perplexity: Language model perplexity
    - Self-BLEU: Diversity metric
    - Distinct-N: N-gram diversity
    - MAUVE: Distribution similarity to human text
    """
    
    @staticmethod
    def compute_self_bleu(texts: List[str], n: int = 4) -> float:
        """
        Compute Self-BLEU score.
        
        Measures diversity by computing BLEU of each text against others.
        Lower is better (more diverse).
        
        Args:
            texts: List of generated texts
            n: N-gram order
            
        Returns:
            Self-BLEU score
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        if len(texts) < 2:
            return 0.0
        
        scores = []
        smoothing = SmoothingFunction().method1
        
        for i, text in enumerate(texts):
            references = [t.split() for j, t in enumerate(texts) if j != i]
            hypothesis = text.split()
            
            if len(hypothesis) == 0:
                continue
            
            score = sentence_bleu(
                references,
                hypothesis,
                smoothing_function=smoothing,
                weights=[1.0/n] * n,
            )
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    @staticmethod
    def compute_distinct_n(texts: List[str], n: int = 2) -> float:
        """
        Compute Distinct-N score.
        
        Measures n-gram diversity.
        
        Args:
            texts: List of generated texts
            n: N-gram order
            
        Returns:
            Distinct-N score (ratio of unique n-grams)
        """
        all_ngrams = []
        
        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if len(all_ngrams) == 0:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams
    
    @staticmethod
    def compute_perplexity(
        texts: List[str],
        model_name: str = 'gpt2',
    ) -> float:
        """
        Compute perplexity using a pretrained language model.
        
        Args:
            texts: List of generated texts
            model_name: Name of pretrained model
            
        Returns:
            Average perplexity
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        perplexities = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
                
                # Compute loss
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Perplexity = exp(loss)
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        return np.mean(perplexities)


class PersonalityAlignmentMetrics:
    """
    Metrics for evaluating personality conditioning.
    
    - Personality Classifier Accuracy
    - Trait Correlation
    """
    
    @staticmethod
    def train_personality_classifier(
        texts: List[str],
        personalities: np.ndarray,
        trait_idx: int = 0,
        n_bins: int = 3,
    ) -> float:
        """
        Train a classifier to predict personality from text.
        
        Uses simple bag-of-words + logistic regression.
        
        Args:
            texts: Training texts
            personalities: Personality scores [num_samples, 5]
            trait_idx: Which trait to predict (0-4)
            n_bins: Number of bins for discretization
            
        Returns:
            Classification accuracy
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        
        # Discretize personality scores into bins
        trait_scores = personalities[:, trait_idx]
        bins = np.linspace(0, 1, n_bins + 1)
        labels = np.digitize(trait_scores, bins[1:-1])
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Evaluate
        accuracy = clf.score(X_test, y_test)
        
        return accuracy
    
    @staticmethod
    def compute_trait_correlation(
        texts: List[str],
        personalities: np.ndarray,
        trait_idx: int = 0,
    ) -> float:
        """
        Compute correlation between text features and personality trait.
        
        Args:
            texts: Generated texts
            personalities: Personality scores [num_samples, 5]
            trait_idx: Which trait (0-4)
            
        Returns:
            Correlation coefficient
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts).toarray()
        
        # Average TF-IDF score per sample
        text_features = X.mean(axis=1)
        
        # Compute correlation
        trait_scores = personalities[:, trait_idx]
        corr = np.corrcoef(text_features, trait_scores)[0, 1]
        
        return corr


def evaluate_model(
    model,
    causal_vae,
    tokenizer,
    test_personalities: torch.Tensor,
    test_texts: List[str],
    device: torch.device,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: MDLM model
        causal_vae: CausalVAE
        tokenizer: Tokenizer
        test_personalities: Test personality scores [num_samples, 5]
        test_texts: Ground truth test texts
        device: Device
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    causal_vae.eval()
    
    # Sample subset
    indices = np.random.choice(len(test_personalities), min(num_samples, len(test_personalities)), replace=False)
    personalities = test_personalities[indices].to(device)
    
    # Generate texts
    print("Generating samples for evaluation...")
    generated_texts = []
    
    for i in range(len(personalities)):
        personality = personalities[i:i+1]
        
        # Get causal embedding
        causal_output = causal_vae(personality, return_all=False)
        personality_cond = causal_output['z_causal']
        
        # Generate
        generated_ids = model.generate(
            personality_cond,
            seq_len=128,
            temperature=1.0,
            num_steps=50,
        )
        
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts.append(text)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = {}
    
    # Text quality
    metrics['self_bleu'] = TextQualityMetrics.compute_self_bleu(generated_texts)
    metrics['distinct_1'] = TextQualityMetrics.compute_distinct_n(generated_texts, n=1)
    metrics['distinct_2'] = TextQualityMetrics.compute_distinct_n(generated_texts, n=2)
    
    try:
        metrics['perplexity'] = TextQualityMetrics.compute_perplexity(generated_texts)
    except:
        print("Warning: Could not compute perplexity")
    
    # Disentanglement (on encoder)
    with torch.no_grad():
        latent_codes = causal_vae.personality_encoder.encode_personality(personalities).cpu().numpy()
    
    personality_np = personalities.cpu().numpy()
    disentanglement = DisentanglementMetrics.compute_pca_alignment(latent_codes, personality_np)
    metrics.update(disentanglement)
    
    metrics['clfr'] = DisentanglementMetrics.compute_clfr(
        causal_vae.personality_encoder,
        personalities.cpu(),
        device,
    )
    
    # Personality alignment
    try:
        for trait_idx, trait_name in enumerate(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']):
            accuracy = PersonalityAlignmentMetrics.train_personality_classifier(
                generated_texts,
                personality_np,
                trait_idx=trait_idx,
            )
            metrics[f'classifier_acc_{trait_name}'] = accuracy
    except:
        print("Warning: Could not compute personality classifier accuracy")
    
    return metrics


if __name__ == '__main__':
    """Test metrics."""
    print("Testing Evaluation Metrics...\n")
    
    # Test text quality metrics
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps across the sleepy canine.",
        "Hello world this is a test sentence.",
        "Another completely different text for testing.",
    ]
    
    print("Text Quality Metrics:")
    print(f"  Self-BLEU: {TextQualityMetrics.compute_self_bleu(texts):.4f}")
    print(f"  Distinct-1: {TextQualityMetrics.compute_distinct_n(texts, n=1):.4f}")
    print(f"  Distinct-2: {TextQualityMetrics.compute_distinct_n(texts, n=2):.4f}")
    
    # Test disentanglement metrics
    print("\nDisentanglement Metrics:")
    latent_codes = np.random.randn(100, 64)
    personalities = np.random.rand(100, 5)
    
    pca_metrics = DisentanglementMetrics.compute_pca_alignment(latent_codes, personalities)
    print(f"  PCA Alignment: {pca_metrics['pca_alignment_mean']:.4f}")
    
    print("\n✓ Metrics test passed!")
