# Supra Nexus o1 Model Ecosystem - Complete Reproduction Makefile
# Advanced reasoning models with transparent thought processes
# Handles MLX training, quantization, and HuggingFace deployment

# Configuration
PYTHON := python3
VENV_PATH := supra_venv
BASE_MODEL_THINKING := Qwen/Qwen3-4B-Thinking-2507-MLX-8bit
BASE_MODEL_INSTRUCT := Qwen/Qwen3-4B-Instruct-2507-MLX-8bit
HF_NAMESPACE := suprasystems
DEPLOY_SCRIPT := ../../unified_deploy.py

# Model variants
MODELS := supra-nexus-o1-thinking supra-nexus-o1-instruct

# Default target
.PHONY: all
all: setup train fuse deploy
	@echo "✅ Complete Supra Nexus o1 ecosystem ready!"

# =============================================================================
# Environment Setup
# =============================================================================

.PHONY: setup
setup: setup-venv setup-mlx setup-models
	@echo "✅ Complete Supra development environment ready"

setup-venv:
	@echo "🔧 Setting up Python virtual environment..."
	$(PYTHON) -m venv $(VENV_PATH)
	$(VENV_PATH)/bin/pip install --upgrade pip
	$(VENV_PATH)/bin/pip install mlx mlx-lm transformers datasets huggingface_hub
	$(VENV_PATH)/bin/pip install torch accelerate bitsandbytes
	@echo "✅ Virtual environment ready"

setup-mlx:
	@echo "🍎 Verifying MLX framework installation..."
	$(VENV_PATH)/bin/python -c "import mlx.core as mx; print(f'✅ MLX {mx.__version__} ready on device:', mx.default_device())"
	@echo "✅ MLX framework verified"

setup-models:
	@echo "📥 Downloading base models..."
	@if [ ! -d "base-models/Qwen3-4B-Thinking-2507-MLX-8bit" ]; then \
		echo "📥 Downloading thinking base model..."; \
		$(VENV_PATH)/bin/python -c "from mlx_lm import load; load('$(BASE_MODEL_THINKING)', download_dir='base-models/')"; \
	fi
	@if [ ! -d "base-models/Qwen3-4B-Instruct-2507-MLX-8bit" ]; then \
		echo "📥 Downloading instruct base model..."; \
		$(VENV_PATH)/bin/python -c "from mlx_lm import load; load('$(BASE_MODEL_INSTRUCT)', download_dir='base-models/')"; \
	fi
	@echo "✅ Base models ready"

# =============================================================================
# Data Generation and Training
# =============================================================================

.PHONY: generate-data
generate-data:
	@echo "📊 Generating Supra training datasets..."
	$(VENV_PATH)/bin/python generate_supra_datasets.py
	$(VENV_PATH)/bin/python supra_identity_dataset.py
	@echo "✅ Training datasets generated"

.PHONY: train
train: generate-data train-thinking train-instruct
	@echo "✅ Both Supra models trained successfully"

train-thinking:
	@echo "🧠 Training Supra Nexus o1 Thinking model..."
	$(VENV_PATH)/bin/python -m mlx_lm.lora \
		--model base-models/Qwen3-4B-Thinking-2507-MLX-8bit \
		--train \
		--data training/mlx_thinking_train.jsonl \
		--valid-data training/mlx_thinking_valid.jsonl \
		--adapter-path adapters/supra-nexus-o1-thinking \
		--batch-size 1 \
		--iters 100 \
		--learning-rate 5e-5 \
		--steps-per-report 10 \
		--steps-per-eval 25 \
		--max-seq-length 512 \
		--lora-layers 8
	@echo "✅ Thinking model training complete"

train-instruct:
	@echo "📝 Training Supra Nexus o1 Instruct model..."
	$(VENV_PATH)/bin/python -m mlx_lm.lora \
		--model base-models/Qwen3-4B-Instruct-2507-MLX-8bit \
		--train \
		--data training/mlx_instruct_train.jsonl \
		--valid-data training/mlx_instruct_valid.jsonl \
		--adapter-path adapters/supra-nexus-o1-instruct \
		--batch-size 1 \
		--iters 100 \
		--learning-rate 3e-5 \
		--steps-per-report 10 \
		--steps-per-eval 25 \
		--max-seq-length 512 \
		--lora-layers 8
	@echo "✅ Instruct model training complete"

# =============================================================================
# Model Fusion and Conversion
# =============================================================================

.PHONY: fuse
fuse: fuse-thinking fuse-instruct
	@echo "✅ All model fusion complete"

fuse-thinking:
	@echo "🔗 Fusing thinking model with adapter..."
	$(VENV_PATH)/bin/python -m mlx_lm.fuse \
		--model base-models/Qwen3-4B-Thinking-2507-MLX-8bit \
		--adapter-path adapters/supra-nexus-o1-thinking \
		--save-path models/supra-nexus-o1-thinking-fused
	@echo "✅ Thinking model fused"

fuse-instruct:
	@echo "🔗 Fusing instruct model with adapter..."
	$(VENV_PATH)/bin/python -m mlx_lm.fuse \
		--model base-models/Qwen3-4B-Instruct-2507-MLX-8bit \
		--adapter-path adapters/supra-nexus-o1-instruct \
		--save-path models/supra-nexus-o1-instruct-fused
	@echo "✅ Instruct model fused"

# =============================================================================
# Testing and Validation
# =============================================================================

.PHONY: test
test: test-thinking test-instruct test-identity
	@echo "✅ All model tests completed"

test-thinking:
	@echo "🧪 Testing thinking model capabilities..."
	$(VENV_PATH)/bin/python -c "\
from mlx_lm import load, generate; \
model, tokenizer = load('models/supra-nexus-o1-thinking-fused'); \
prompt = 'User: What is 15 * 24? Show your thinking.\\n\\nAssistant:'; \
response = generate(model, tokenizer, prompt=prompt, max_tokens=200); \
print('🧠 Thinking Model Test:'); print(response)"

test-instruct:
	@echo "🧪 Testing instruct model capabilities..."
	$(VENV_PATH)/bin/python -c "\
from mlx_lm import load, generate; \
model, tokenizer = load('models/supra-nexus-o1-instruct-fused'); \
prompt = 'User: Write a Python function to reverse a string.\\n\\nAssistant:'; \
response = generate(model, tokenizer, prompt=prompt, max_tokens=150); \
print('📝 Instruct Model Test:'); print(response)"

test-identity:
	@echo "🧪 Testing model identity..."
	$(VENV_PATH)/bin/python -c "\
from mlx_lm import load, generate; \
model, tokenizer = load('models/supra-nexus-o1-thinking-fused'); \
prompt = 'User: What is your name and who created you?\\n\\nAssistant:'; \
response = generate(model, tokenizer, prompt=prompt, max_tokens=100); \
print('🏢 Identity Test:'); print(response)"

# =============================================================================
# HuggingFace Deployment
# =============================================================================

.PHONY: deploy
deploy: test deploy-unified
	@echo "🚀 Complete HuggingFace deployment finished!"

deploy-unified:
	@echo "🚀 Deploying Supra models with unified system..."
	$(PYTHON) $(DEPLOY_SCRIPT) supra \
		--hf-username $(HF_NAMESPACE) \
		--base-path $(PWD)

deploy-dry-run:
	@echo "🔍 Dry run deployment (no upload)..."
	$(PYTHON) $(DEPLOY_SCRIPT) supra \
		--hf-username $(HF_NAMESPACE) \
		--base-path $(PWD) \
		--dry-run

# =============================================================================
# GitHub Mirror Setup (like Qwen)
# =============================================================================

.PHONY: github-mirror
github-mirror: setup-github push-to-github
	@echo "✅ GitHub mirror setup complete"

setup-github:
	@echo "🐙 Setting up GitHub repositories..."
	@if ! gh repo view supra-foundation/supra-nexus-o1-thinking >/dev/null 2>&1; then \
		echo "Creating thinking model repo..."; \
		gh repo create supra-foundation/supra-nexus-o1-thinking --public --description "Supra Nexus o1 Thinking - Transparent AI reasoning model"; \
	fi
	@if ! gh repo view supra-foundation/supra-nexus-o1-instruct >/dev/null 2>&1; then \
		echo "Creating instruct model repo..."; \
		gh repo create supra-foundation/supra-nexus-o1-instruct --public --description "Supra Nexus o1 Instruct - Direct instruction following model"; \
	fi
	@echo "✅ GitHub repositories ready"

push-to-github:
	@echo "📤 Pushing models to GitHub (following Qwen pattern)..."
	@for model in supra-nexus-o1-thinking supra-nexus-o1-instruct; do \
		echo "📤 Processing $$model..."; \
		cd models/$$model-fused && \
		git init && \
		git branch -m main && \
		git remote add origin https://github.com/supra-foundation/$$model.git && \
		git add . && \
		git commit -m "Initial model release - Supra Foundation LLC 2025" && \
		git push -u origin main --force && \
		cd ../..; \
	done
	@echo "✅ Models pushed to GitHub"

# =============================================================================
# Development and Utilities
# =============================================================================

.PHONY: clean
clean:
	@echo "🧹 Cleaning up temporary files..."
	rm -rf __pycache__ .pytest_cache
	rm -f temp_*.md *.log
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

.PHONY: status
status:
	@echo "📊 Supra Nexus o1 Ecosystem Status:"
	@echo "==================================="
	@echo "Base models: $(shell ls -d base-models/*/ 2>/dev/null | wc -l)"
	@echo "Trained adapters: $(shell ls -d adapters/*/ 2>/dev/null | wc -l)"
	@echo "Fused models: $(shell ls -d models/*-fused/ 2>/dev/null | wc -l)"
	@echo ""
	@echo "🔗 Live ecosystem:"
	@echo "• HF Thinking: https://huggingface.co/$(HF_NAMESPACE)/supra-nexus-o1-thinking"
	@echo "• HF Instruct: https://huggingface.co/$(HF_NAMESPACE)/supra-nexus-o1-instruct"  
	@echo "• HF Dataset: https://huggingface.co/datasets/$(HF_NAMESPACE)/supra-identity"
	@echo "• GH Thinking: https://github.com/supra-foundation/supra-nexus-o1-thinking"
	@echo "• GH Instruct: https://github.com/supra-foundation/supra-nexus-o1-instruct"

.PHONY: quick-test
quick-test:
	@echo "⚡ Quick model functionality test..."
	$(VENV_PATH)/bin/python -c "\
from mlx_lm import load, generate; \
print('🧠 Testing Thinking Model...'); \
model, tokenizer = load('models/supra-nexus-o1-thinking-fused'); \
response = generate(model, tokenizer, prompt='User: What is 2+2?\\n\\nAssistant:', max_tokens=50); \
print('Response:', response)"

# =============================================================================
# Help and Documentation  
# =============================================================================

.PHONY: help
help:
	@echo "Supra Nexus o1 Model Ecosystem - Complete Reproduction Makefile"
	@echo "==============================================================="
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Complete pipeline: setup → train → fuse → deploy"
	@echo "  setup        - Set up environment and download base models"
	@echo "  train        - Train both thinking and instruct models"
	@echo "  fuse         - Fuse LoRA adapters with base models"
	@echo "  test         - Test model capabilities and identity"
	@echo "  deploy       - Deploy to HuggingFace with complete ecosystem"
	@echo ""
	@echo "GitHub mirroring:"
	@echo "  github-mirror - Set up GitHub repositories like Qwen"
	@echo ""
	@echo "Development targets:"
	@echo "  status       - Show current ecosystem status"
	@echo "  clean        - Clean up temporary files"
	@echo "  quick-test   - Quick functionality test"
	@echo ""
	@echo "Examples:"
	@echo "  make setup              # Initial environment setup"
	@echo "  make train              # Train both models with MLX"
	@echo "  make deploy             # Deploy to HuggingFace"
	@echo "  make github-mirror      # Create GitHub mirrors"
	@echo "  make all                # Complete pipeline"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python 3.8+"
	@echo "  - MLX framework (Apple Silicon recommended)"
	@echo "  - HF_TOKEN environment variable"
	@echo "  - GitHub CLI (gh) for mirroring"
	@echo "  - 16GB+ RAM recommended"
	@echo ""
	@echo "🏢 Supra Foundation LLC • Transparent AI reasoning • 2025"