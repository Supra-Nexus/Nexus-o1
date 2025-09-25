#!/bin/bash
# Upload v1.1 models with recursive improvements

echo "📤 Uploading Supra Nexus O1 v1.1 models..."

# Upload instruct v1.1
echo "Uploading instruct v1.1..."
huggingface-cli upload \
    Supra-Nexus/supra-nexus-o1-instruct-v1.1 \
    ./models/supra-nexus-o1-instruct-v1.1 \
    . \
    --repo-type model

# Upload thinking v1.1  
echo "Uploading thinking v1.1..."
huggingface-cli upload \
    Supra-Nexus/supra-nexus-o1-thinking-v1.1 \
    ./models/supra-nexus-o1-thinking-v1.1 \
    . \
    --repo-type model

echo "✅ v1.1 models uploaded with recursive improvements!"
