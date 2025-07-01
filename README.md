# Transformers ML Project

A machine learning project using Hugging Face Transformers for text classification and text generation.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Hugging Face account and API token

## Installation

1. **Create virtual environment (recommended)**
   ```bash
   python -m venv .env
   source .env/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Hugging Face Setup

You need a Hugging Face account and API token to download models and datasets.

1. **Create account**: Go to [huggingface.co](https://huggingface.co) and sign up
2. **Get API token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Login via CLI**:
   ```bash
   huggingface-cli login
   ```
   Enter your token when prompted.

**Documentation**: [Hugging Face Token Authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size in trainer files or use CPU-only mode

**Model Not Found**: Ensure training completed successfully and check `trained_models/` directory

**Import Errors**: Verify all dependencies installed with `pip install -r requirements.txt` 