#!/usr/bin/env python3
"""Comprehensive inference testing for Zen and Supra models."""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelInferenceTester:
    """Test inference capabilities of language models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {
            "model": model_name,
            "tests": {},
            "metrics": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.results["tests"]["model_loading"] = "PASSED"
    
    def test_basic_generation(self) -> bool:
        """Test basic text generation."""
        prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "def fibonacci(n):",
        ]
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
            latency = time.time() - start_time
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validate output
            assert len(response) > len(prompt), f"No generation for prompt: {prompt}"
            assert response.startswith(prompt), f"Response doesn't include prompt"
            
            self.results["tests"][f"basic_gen_{prompt[:20]}"] = "PASSED"
            self.results["metrics"][f"latency_{prompt[:20]}"] = latency
        
        return True
    
    def test_generation_parameters(self) -> bool:
        """Test different generation parameters."""
        prompt = "Explain quantum computing in simple terms:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        configs = [
            {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 100},
            {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 100},
            {"temperature": 1.0, "top_k": 50, "max_new_tokens": 100},
        ]
        
        for i, config in enumerate(configs):
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                **config
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validate parameter effects
            assert len(response) > len(prompt)
            self.results["tests"][f"param_test_{i}"] = "PASSED"
        
        return True
    
    def test_batch_inference(self) -> bool:
        """Test batch inference capabilities."""
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning:",
            "Define neural networks:",
            "What are transformers in AI?",
        ]
        
        # Batch tokenization
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        batch_latency = time.time() - start_time
        
        # Decode all outputs
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        # Validate batch processing
        assert len(responses) == len(prompts)
        for prompt, response in zip(prompts, responses):
            assert len(response) > len(prompt)
        
        self.results["tests"]["batch_inference"] = "PASSED"
        self.results["metrics"]["batch_latency"] = batch_latency
        self.results["metrics"]["batch_throughput"] = len(prompts) / batch_latency
        
        return True
    
    def test_long_context(self) -> bool:
        """Test long context handling."""
        # Create a long context
        context = "This is a test. " * 500  # ~2000 tokens
        prompt = context + "Summarize the above text:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.results["tests"]["long_context"] = "PASSED"
            return True
        except Exception as e:
            self.results["tests"]["long_context"] = f"FAILED: {str(e)}"
            return False
    
    def measure_throughput(self) -> Dict[str, float]:
        """Measure inference throughput."""
        prompt = "Write a short story about AI:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warmup
        for _ in range(3):
            self.model.generate(**inputs, max_new_tokens=50)
        
        # Measure throughput
        num_runs = 10
        total_tokens = 0
        start_time = time.time()
        
        for _ in range(num_runs):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            total_tokens += outputs.shape[1]
        
        elapsed_time = time.time() - start_time
        tokens_per_second = total_tokens / elapsed_time
        
        self.results["metrics"]["tokens_per_second"] = tokens_per_second
        self.results["metrics"]["avg_latency_per_token"] = elapsed_time / total_tokens
        
        return self.results["metrics"]
    
    def run_all_tests(self) -> Dict:
        """Run all inference tests."""
        print(f"\n{'='*50}")
        print(f"Testing {self.model_name}")
        print('='*50)
        
        tests = [
            ("Basic Generation", self.test_basic_generation),
            ("Generation Parameters", self.test_generation_parameters),
            ("Batch Inference", self.test_batch_inference),
            ("Long Context", self.test_long_context),
            ("Throughput Measurement", lambda: self.measure_throughput() and True),
        ]
        
        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")
            try:
                if test_func():
                    print(f"✅ {test_name} passed")
                else:
                    print(f"❌ {test_name} failed")
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                self.results["tests"][test_name.lower().replace(" ", "_")] = f"FAILED: {str(e)}"
        
        # Summary
        passed = sum(1 for v in self.results["tests"].values() if v == "PASSED")
        total = len(self.results["tests"])
        self.results["summary"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0
        }
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Test model inference")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()
    
    # Map short names to full model paths
    model_map = {
        "supra-nexus-o1-instruct": "Supra-Nexus/supra-nexus-o1-instruct",
        "supra-nexus-o1-thinking": "Supra-Nexus/supra-nexus-o1-thinking",
        "zen-nano-instruct": "ZenLM/zen-nano-instruct",
        "zen-nano-thinking": "ZenLM/zen-nano-thinking",
    }
    
    model_name = model_map.get(args.model, args.model)
    
    # Run tests
    tester = ModelInferenceTester(model_name)
    results = tester.run_all_tests()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.model.replace('/', '_')}_inference.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {output_file}")
    print(f"\n📈 Summary: {results['summary']['passed']}/{results['summary']['total']} tests passed")
    print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
    
    # Exit with error if any tests failed
    if results["summary"]["success_rate"] < 1.0:
        exit(1)


if __name__ == "__main__":
    main()