# Supra Nexus O1 Training with Zoo AI's Gym

.PHONY: install train-instruct train-thinking train-all test deploy

# Zoo Gym Framework
ZOO_GYM_REPO = https://github.com/zooai/gym
ZOO_GYM_DIR = ./zoo-gym

install:
	@echo "🏋️ Installing Zoo AI's Gym Framework..."
	@if [ ! -d $(ZOO_GYM_DIR) ]; then \
		git clone $(ZOO_GYM_REPO) $(ZOO_GYM_DIR); \
	fi
	@cd $(ZOO_GYM_DIR) && pip install -r requirements.txt
	@pip install transformers datasets accelerate peft
	@echo "✅ Zoo Gym installed"

train-instruct:
	@echo "🏋️ Training instruct model with Zoo Gym..."
	python $(ZOO_GYM_DIR)/train.py \
		--model Qwen/Qwen3-4B-2507 \
		--dataset ./training/supra_instruct.jsonl \
		--output ./models/supra-nexus-o1-instruct \
		--config ./configs/instruct.yaml \
		--framework zoo_gym

train-thinking:
	@echo "🏋️ Training thinking model with Zoo Gym..."
	python $(ZOO_GYM_DIR)/train.py \
		--model Qwen/Qwen3-4B-2507 \
		--dataset ./training/supra_thinking.jsonl \
		--output ./models/supra-nexus-o1-thinking \
		--config ./configs/thinking.yaml \
		--framework zoo_gym

train-v1.1:
	@echo "🏋️ Training v1.1 with recursive data using Zoo Gym..."
	python $(ZOO_GYM_DIR)/train.py \
		--model Qwen/Qwen3-4B-2507 \
		--dataset ./training/nexus-o1-upgrade-1.jsonl \
		--output ./models/supra-nexus-o1-v1.1 \
		--config ./configs/v1_1.yaml \
		--framework zoo_gym \
		--recursive-training

train-all: train-instruct train-thinking train-v1.1
	@echo "✅ All models trained with Zoo Gym"

test:
	@echo "🧪 Running tests..."
	python tests/test_inference.py --model supra-nexus-o1-instruct
	python tests/test_cot.py --model supra-nexus-o1-thinking

deploy:
	@echo "📤 Deploying models trained with Zoo Gym..."
	huggingface-cli upload Supra-Nexus/supra-nexus-o1-instruct ./models/supra-nexus-o1-instruct
	huggingface-cli upload Supra-Nexus/supra-nexus-o1-thinking ./models/supra-nexus-o1-thinking
	@echo "✅ Models deployed (trained with Zoo Gym)"

info:
	@echo "=========================================="
	@echo "  Supra Nexus O1 - Zoo AI's Gym Training"
	@echo "=========================================="
	@echo "  Framework: Zoo AI's Gym"
	@echo "  Repo: $(ZOO_GYM_REPO)"
	@echo "  Base Model: Qwen3-4B-2507"
	@echo "  Parameters: 4.02B"
	@echo "=========================================="
