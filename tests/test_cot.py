#!/usr/bin/env python3
"""Chain-of-thought validation for reasoning models."""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CoTMetrics:
    """Metrics for chain-of-thought evaluation."""
    has_thinking_tags: bool
    num_reasoning_steps: int
    coherence_score: float
    self_corrections: int
    final_answer_present: bool
    thinking_token_ratio: float


class ChainOfThoughtValidator:
    """Validate chain-of-thought reasoning quality."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.results = {
            "model": model_name,
            "test_cases": [],
            "aggregate_metrics": {}
        }
    
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
    
    def extract_thinking(self, text: str) -> Optional[str]:
        """Extract content between <thinking> tags."""
        pattern = r'<thinking>(.*?)</thinking>'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else None
    
    def count_reasoning_steps(self, thinking: str) -> int:
        """Count number of reasoning steps in thinking."""
        if not thinking:
            return 0
        
        # Count various step indicators
        step_patterns = [
            r'step \d+:',
            r'\d+\.',
            r'first,|second,|third,|next,|then,|finally,',
            r'therefore|thus|hence|so,',
        ]
        
        total_steps = 0
        for pattern in step_patterns:
            total_steps += len(re.findall(pattern, thinking, re.IGNORECASE))
        
        # Also count newline-separated thoughts
        lines = [l.strip() for l in thinking.split('\n') if l.strip()]
        total_steps = max(total_steps, len(lines))
        
        return min(total_steps, 20)  # Cap at 20 to avoid outliers
    
    def measure_coherence(self, thinking: str) -> float:
        """Measure coherence of reasoning chain."""
        if not thinking:
            return 0.0
        
        score = 0.0
        
        # Check for logical connectors
        connectors = ['because', 'therefore', 'thus', 'since', 'if', 'then', 'so']
        for connector in connectors:
            if connector in thinking.lower():
                score += 0.1
        
        # Check for structured reasoning
        if re.search(r'\d+[\.:\)]', thinking):
            score += 0.2
        
        # Check for conclusion
        if any(word in thinking.lower() for word in ['conclude', 'answer', 'result', 'final']):
            score += 0.2
        
        # Check for self-reference
        if any(word in thinking.lower() for word in ['i need', 'let me', 'i should']):
            score += 0.1
        
        # Length bonus (longer reasoning often more thorough)
        word_count = len(thinking.split())
        if word_count > 50:
            score += min(0.2, word_count / 500)
        
        return min(score, 1.0)
    
    def count_self_corrections(self, thinking: str) -> int:
        """Count self-correction patterns in thinking."""
        if not thinking:
            return 0
        
        correction_patterns = [
            r'actually',
            r'wait',
            r'no,',
            r'correction:',
            r'mistake',
            r'wrong',
            r'instead',
            r'let me reconsider',
            r'on second thought',
        ]
        
        corrections = 0
        for pattern in correction_patterns:
            corrections += len(re.findall(pattern, thinking, re.IGNORECASE))
        
        return corrections
    
    def analyze_response(self, response: str) -> CoTMetrics:
        """Analyze a model response for CoT quality."""
        thinking = self.extract_thinking(response)
        
        # Calculate metrics
        metrics = CoTMetrics(
            has_thinking_tags=thinking is not None,
            num_reasoning_steps=self.count_reasoning_steps(thinking) if thinking else 0,
            coherence_score=self.measure_coherence(thinking) if thinking else 0.0,
            self_corrections=self.count_self_corrections(thinking) if thinking else 0,
            final_answer_present=bool(re.search(r'(answer|result|conclusion)[:=]', response.lower())),
            thinking_token_ratio=len(thinking.split()) / len(response.split()) if thinking else 0.0
        )
        
        return metrics
    
    def test_math_reasoning(self) -> Dict:
        """Test mathematical reasoning capabilities."""
        problems = [
            {
                "prompt": "Solve step by step: If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed?",
                "expected_answer": 60,
                "type": "math"
            },
            {
                "prompt": "Calculate: A store offers a 25% discount on a $80 item, then adds 8% tax. What is the final price?",
                "expected_answer": 64.8,
                "type": "math"
            },
            {
                "prompt": "Solve: If 3x + 7 = 22, what is the value of x?",
                "expected_answer": 5,
                "type": "algebra"
            }
        ]
        
        results = []
        for problem in problems:
            inputs = self.tokenizer(problem["prompt"], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,  # Low temperature for math
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            metrics = self.analyze_response(response)
            
            # Try to extract numerical answer
            answer_pattern = r'(?:answer|result|equals?)[:\s]*([0-9.]+)'
            answer_match = re.search(answer_pattern, response.lower())
            extracted_answer = float(answer_match.group(1)) if answer_match else None
            
            result = {
                "problem": problem["prompt"][:50] + "...",
                "metrics": metrics.__dict__,
                "correct_answer": extracted_answer == problem["expected_answer"] if extracted_answer else False,
                "extracted_answer": extracted_answer,
                "expected_answer": problem["expected_answer"]
            }
            
            results.append(result)
        
        return results
    
    def test_logic_reasoning(self) -> Dict:
        """Test logical reasoning capabilities."""
        problems = [
            {
                "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain your reasoning.",
                "type": "logic"
            },
            {
                "prompt": "If it's raining, the ground is wet. The ground is wet. Is it necessarily raining? Explain why or why not.",
                "type": "logic"
            }
        ]
        
        results = []
        for problem in problems:
            inputs = self.tokenizer(problem["prompt"], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            metrics = self.analyze_response(response)
            
            result = {
                "problem": problem["prompt"][:50] + "...",
                "metrics": metrics.__dict__,
                "type": problem["type"]
            }
            
            results.append(result)
        
        return results
    
    def test_code_reasoning(self) -> Dict:
        """Test code reasoning capabilities."""
        problems = [
            {
                "prompt": "Debug this code and explain the issue:\n```python\ndef find_max(lst):\n    max_val = 0\n    for num in lst:\n        if num > max_val:\n            max_val = num\n    return max_val\n```\nWhat's wrong with this function?",
                "type": "code"
            }
        ]
        
        results = []
        for problem in problems:
            inputs = self.tokenizer(problem["prompt"], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.2,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            metrics = self.analyze_response(response)
            
            result = {
                "problem": problem["prompt"][:50] + "...",
                "metrics": metrics.__dict__,
                "type": problem["type"]
            }
            
            results.append(result)
        
        return results
    
    def calculate_aggregate_metrics(self, all_results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all tests."""
        total_tests = len(all_results)
        
        if total_tests == 0:
            return {}
        
        # Extract all metrics
        metrics_list = [r["metrics"] for r in all_results]
        
        aggregate = {
            "total_tests": total_tests,
            "thinking_tag_rate": sum(m["has_thinking_tags"] for m in metrics_list) / total_tests,
            "avg_reasoning_steps": sum(m["num_reasoning_steps"] for m in metrics_list) / total_tests,
            "avg_coherence_score": sum(m["coherence_score"] for m in metrics_list) / total_tests,
            "avg_self_corrections": sum(m["self_corrections"] for m in metrics_list) / total_tests,
            "final_answer_rate": sum(m["final_answer_present"] for m in metrics_list) / total_tests,
            "avg_thinking_ratio": sum(m["thinking_token_ratio"] for m in metrics_list) / total_tests,
        }
        
        # Calculate accuracy for math problems
        math_results = [r for r in all_results if "correct_answer" in r]
        if math_results:
            aggregate["math_accuracy"] = sum(r["correct_answer"] for r in math_results) / len(math_results)
        
        # Calculate CoT quality score
        aggregate["cot_quality_score"] = (
            aggregate["thinking_tag_rate"] * 0.3 +
            min(aggregate["avg_reasoning_steps"] / 10, 1.0) * 0.2 +
            aggregate["avg_coherence_score"] * 0.3 +
            aggregate["final_answer_rate"] * 0.2
        )
        
        return aggregate
    
    def run_all_tests(self) -> Dict:
        """Run all chain-of-thought tests."""
        print(f"\n{'='*60}")
        print(f"Chain-of-Thought Validation: {self.model_name}")
        print('='*60)
        
        all_results = []
        
        # Run test suites
        print("\n📐 Testing Math Reasoning...")
        math_results = self.test_math_reasoning()
        all_results.extend(math_results)
        
        print("🧩 Testing Logic Reasoning...")
        logic_results = self.test_logic_reasoning()
        all_results.extend(logic_results)
        
        print("💻 Testing Code Reasoning...")
        code_results = self.test_code_reasoning()
        all_results.extend(code_results)
        
        # Calculate aggregate metrics
        aggregate = self.calculate_aggregate_metrics(all_results)
        
        # Compile final results
        self.results["test_cases"] = all_results
        self.results["aggregate_metrics"] = aggregate
        
        # Print summary
        print(f"\n📊 CoT Validation Summary:")
        print(f"   Thinking Tag Rate: {aggregate.get('thinking_tag_rate', 0):.1%}")
        print(f"   Avg Reasoning Steps: {aggregate.get('avg_reasoning_steps', 0):.1f}")
        print(f"   Avg Coherence Score: {aggregate.get('avg_coherence_score', 0):.2f}")
        print(f"   CoT Quality Score: {aggregate.get('cot_quality_score', 0):.2f}/1.0")
        
        if "math_accuracy" in aggregate:
            print(f"   Math Accuracy: {aggregate['math_accuracy']:.1%}")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Validate chain-of-thought reasoning")
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
    
    # Run validation
    validator = ChainOfThoughtValidator(model_name)
    results = validator.run_all_tests()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.model.replace('/', '_')}_cot.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Exit with error if quality score is too low
    quality_score = results["aggregate_metrics"].get("cot_quality_score", 0)
    if quality_score < 0.5:
        print(f"⚠️  Warning: Low CoT quality score ({quality_score:.2f} < 0.5)")
        exit(1)


if __name__ == "__main__":
    main()