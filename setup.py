from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="causal-diffusion-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Causal Diffusion Models with Disentangled Personality Encoders for Policy Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Diffusion_LLM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "lightning>=2.0.0",
        "wandb>=0.15.0",
        "dowhy>=0.11.0",
        "networkx>=3.1",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.1",
        "einops>=0.6.1",
        "timm>=0.9.2",
        "fsspec>=2023.6.0",
        "flash-attn>=2.0.0",
    ],
)
