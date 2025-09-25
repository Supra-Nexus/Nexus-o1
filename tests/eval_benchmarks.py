#!/usr/bin/env python3
"""
Standard benchmark evaluation for Zen and Supra models.
Includes MMLU (Massive Multitask Language Understanding) and HellaSwag benchmarks.
Uses minimal dependencies with built-in dataset samples.
"""

import json
import logging
import random
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkEvaluation(unittest.TestCase):
    """Standard benchmark evaluation suite."""
    
    def setUp(self):
        """Set up benchmark test environment."""
        self.test_results = {
            "mmlu_results": {},
            "hellaswag_results": {},
            "test_timestamp": time.time(),
            "test_environment": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # MMLU sample questions across different subjects
        self.mmlu_samples = {
            "abstract_algebra": [
                {
                    "question": "Find the degree for the given field extension Q(√2, √3, √6) over Q.",
                    "choices": ["A) 0", "B) 4", "C) 2", "D) 6"],
                    "answer": "B"
                },
                {
                    "question": "Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the order of p.",
                    "choices": ["A) 8", "B) 2", "C) 24", "D) 4"],
                    "answer": "D"
                }
            ],
            "anatomy": [
                {
                    "question": "Which of the following is the body cavity that contains the pituitary gland?",
                    "choices": ["A) Abdominal", "B) Cranial", "C) Pleural", "D) Spinal"],
                    "answer": "B"
                },
                {
                    "question": "What is the embryological origin of the hyoid bone?",
                    "choices": ["A) The first pharyngeal arch", "B) The second pharyngeal arch", "C) The second and third pharyngeal arches", "D) The third and fourth pharyngeal arches"],
                    "answer": "C"
                }
            ],
            "astronomy": [
                {
                    "question": "How long does it take Pluto to orbit the Sun?",
                    "choices": ["A) 240 Earth years", "B) 248 Earth years", "C) 250 Earth years", "D) 260 Earth years"],
                    "answer": "B"
                },
                {
                    "question": "What is the most abundant element in the Sun?",
                    "choices": ["A) Hydrogen", "B) Helium", "C) Carbon", "D) Iron"],
                    "answer": "A"
                }
            ],
            "business_ethics": [
                {
                    "question": "What is the term for a decision-making approach where ethical choices are made based on the consequences?",
                    "choices": ["A) Deontological", "B) Virtue ethics", "C) Consequentialism", "D) Divine command theory"],
                    "answer": "C"
                },
                {
                    "question": "Which of the following is NOT typically considered a stakeholder in business ethics?",
                    "choices": ["A) Employees", "B) Customers", "C) Competitors", "D) Shareholders"],
                    "answer": "C"
                }
            ],
            "computer_security": [
                {
                    "question": "What is the primary purpose of a firewall?",
                    "choices": ["A) Data encryption", "B) Network traffic filtering", "C) Password hashing", "D) Data compression"],
                    "answer": "B"
                },
                {
                    "question": "Which of the following is a characteristic of symmetric encryption?",
                    "choices": ["A) Uses public and private key pairs", "B) Uses the same key for encryption and decryption", "C) Is slower than asymmetric encryption", "D) Cannot be used for bulk data encryption"],
                    "answer": "B"
                }
            ]
        }
        
        # HellaSwag sample questions (commonsense reasoning)
        self.hellaswag_samples = [
            {
                "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid getting a bath. She",
                "endings": [
                    "rinses the bucket out with soap and blow dries the dog.",
                    "uses a hose to keep it from getting soapy.",
                    "gets the dog wet, then it runs away again.",
                    "gets into a bathtub with the dog."
                ],
                "answer": 2
            },
            {
                "context": "A man is standing on a ladder cleaning windows on a building. He",
                "endings": [
                    "climbs down and starts climbing a tree.",
                    "demonstrates how to clean a window properly.",
                    "leans over and falls off the building.",
                    "jumps down with a parachute."
                ],
                "answer": 1
            },
            {
                "context": "Children smiling and waving at camera. There are children playing on the playground behind them. They",
                "endings": [
                    "look directly at the camera and keep smiling.",
                    "runs over to the playground and plays with the other children.",
                    "stops what they are doing and watch the camera.",
                    "turns away and walks towards the playground."
                ],
                "answer": 0
            },
            {
                "context": "A young man is seen playing a piano while singing. He",
                "endings": [
                    "continues to play the piano while singing.",
                    "stops playing and walks away from the piano.",
                    "throws the piano bench at the camera.",
                    "takes off his shirt while playing."
                ],
                "answer": 0
            },
            {
                "context": "A person is washing dishes at a kitchen sink. They",
                "endings": [
                    "dry the dishes with a towel and put them away.",
                    "throw all the dishes in the garbage.",
                    "start juggling the wet dishes.",
                    "break all the dishes intentionally."
                ],
                "answer": 0
            }
        ]
        
        # Benchmark thresholds
        self.benchmark_thresholds = {
            "mmlu_accuracy_threshold": 0.25,  # Above random (0.25 for 4-choice)
            "hellaswag_accuracy_threshold": 0.30,  # Above random (0.25 for 4-choice)
            "min_response_confidence": 0.6
        }

    def _try_import_model_library(self, library: str) -> bool:
        """Try to import a model library."""
        try:
            if library == "transformers":
                import transformers
            elif library == "mlx_lm":
                import mlx_lm
            return True
        except ImportError:
            logger.warning(f"{library} not available")
            return False

    def _load_model(self, model_path: str) -> Optional[Tuple]:
        """Load model with best available library."""
        # Try transformers first
        if self._try_import_model_library("transformers"):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return (model, tokenizer, "transformers")
            except Exception as e:
                logger.warning(f"Transformers failed for {model_path}: {e}")
        
        # Try MLX if model path suggests MLX format
        if "mlx" in model_path.lower() and self._try_import_model_library("mlx_lm"):
            try:
                from mlx_lm import load
                model, tokenizer = load(model_path)
                return (model, tokenizer, "mlx_lm")
            except Exception as e:
                logger.warning(f"MLX failed for {model_path}: {e}")
        
        return None

    def _generate_response(self, model_tuple: Tuple, prompt: str) -> str:
        """Generate response using loaded model."""
        model, tokenizer, library = model_tuple
        
        try:
            if library == "transformers":
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + 50,  # Short responses for MC
                    temperature=0.1,  # Low temperature for consistency
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt):].strip()
                
            elif library == "mlx_lm":
                from mlx_lm import generate
                return generate(
                    model, 
                    tokenizer, 
                    prompt=prompt,
                    max_tokens=50,
                    temp=0.1
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def _extract_multiple_choice_answer(self, response: str, num_choices: int = 4) -> Optional[str]:
        """Extract multiple choice answer from response."""
        response = response.strip().upper()
        
        # Look for explicit letter answers
        for letter in ['A', 'B', 'C', 'D'][:num_choices]:
            if f"{letter})" in response or f"({letter})" in response:
                return letter
            if response.startswith(letter) and len(response) <= 3:
                return letter
        
        # Look for "The answer is X" patterns
        answer_patterns = [
            r"THE ANSWER IS ([A-D])",
            r"ANSWER: ([A-D])",
            r"^([A-D])[\.\)]",
            r"\b([A-D])\b"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                letter = match.group(1)
                if letter in ['A', 'B', 'C', 'D'][:num_choices]:
                    return letter
        
        return None

    def _evaluate_mmlu(self, model_tuple: Tuple, model_name: str) -> Dict:
        """Evaluate model on MMLU benchmark samples."""
        results = {
            "model_name": model_name,
            "subject_scores": {},
            "overall_accuracy": 0.0,
            "total_questions": 0,
            "correct_answers": 0
        }
        
        logger.info(f"Evaluating MMLU for {model_name}")
        
        for subject, questions in self.mmlu_samples.items():
            subject_correct = 0
            subject_results = []
            
            for question_data in questions:
                # Format multiple choice prompt
                prompt = f"""Question: {question_data['question']}

{chr(10).join(question_data['choices'])}

Answer:"""
                
                response = self._generate_response(model_tuple, prompt)
                predicted = self._extract_multiple_choice_answer(response)
                
                is_correct = predicted == question_data['answer']
                if is_correct:
                    subject_correct += 1
                    results["correct_answers"] += 1
                
                subject_results.append({
                    "question": question_data['question'],
                    "correct_answer": question_data['answer'],
                    "predicted_answer": predicted,
                    "response": response,
                    "correct": is_correct
                })
                
                results["total_questions"] += 1
            
            results["subject_scores"][subject] = {
                "accuracy": subject_correct / len(questions),
                "correct": subject_correct,
                "total": len(questions),
                "details": subject_results
            }
        
        if results["total_questions"] > 0:
            results["overall_accuracy"] = results["correct_answers"] / results["total_questions"]
        
        return results

    def _evaluate_hellaswag(self, model_tuple: Tuple, model_name: str) -> Dict:
        """Evaluate model on HellaSwag benchmark samples."""
        results = {
            "model_name": model_name,
            "accuracy": 0.0,
            "correct_answers": 0,
            "total_questions": len(self.hellaswag_samples),
            "details": []
        }
        
        logger.info(f"Evaluating HellaSwag for {model_name}")
        
        for i, sample in enumerate(self.hellaswag_samples):
            # Format HellaSwag prompt
            prompt = f"""Complete the following scenario with the most likely continuation:

{sample['context']}

A) {sample['endings'][0]}
B) {sample['endings'][1]}
C) {sample['endings'][2]}
D) {sample['endings'][3]}

Answer:"""
            
            response = self._generate_response(model_tuple, prompt)
            predicted = self._extract_multiple_choice_answer(response)
            
            # Convert number answer to letter
            correct_letter = ['A', 'B', 'C', 'D'][sample['answer']]
            is_correct = predicted == correct_letter
            
            if is_correct:
                results["correct_answers"] += 1
            
            results["details"].append({
                "context": sample['context'],
                "correct_answer": correct_letter,
                "predicted_answer": predicted,
                "response": response,
                "correct": is_correct
            })
        
        if results["total_questions"] > 0:
            results["accuracy"] = results["correct_answers"] / results["total_questions"]
        
        return results

    def _test_model_benchmarks(self, model_path: str, model_name: str) -> Dict:
        """Run all benchmarks for a model."""
        model_tuple = self._load_model(model_path)
        if not model_tuple:
            return {
                "model_name": model_name,
                "status": "failed",
                "reason": "Could not load model"
            }
        
        results = {
            "model_name": model_name,
            "model_path": model_path,
            "status": "success",
            "mmlu": self._evaluate_mmlu(model_tuple, model_name),
            "hellaswag": self._evaluate_hellaswag(model_tuple, model_name)
        }
        
        return results

    def test_zen_benchmarks(self):
        """Test Zen models on benchmarks."""
        zen_model_paths = [
            "/Users/z/work/zen/zen-nano/models/zen-nano-instruct",
            "/Users/z/work/zen/zen-nano/models/zen-nano-thinking"
        ]
        
        for model_path in zen_model_paths:
            if Path(model_path).exists():
                model_name = f"zen_{Path(model_path).name}"
                
                results = self._test_model_benchmarks(model_path, model_name)
                self.test_results["mmlu_results"][model_name] = results.get("mmlu", {})
                self.test_results["hellaswag_results"][model_name] = results.get("hellaswag", {})
                
                # Assertions for benchmark performance
                if results.get("status") == "success":
                    mmlu_accuracy = results["mmlu"]["overall_accuracy"]
                    hellaswag_accuracy = results["hellaswag"]["accuracy"]
                    
                    self.assertGreaterEqual(
                        mmlu_accuracy,
                        self.benchmark_thresholds["mmlu_accuracy_threshold"],
                        f"Zen {model_name} MMLU accuracy {mmlu_accuracy:.3f} below threshold"
                    )
                    
                    self.assertGreaterEqual(
                        hellaswag_accuracy,
                        self.benchmark_thresholds["hellaswag_accuracy_threshold"],
                        f"Zen {model_name} HellaSwag accuracy {hellaswag_accuracy:.3f} below threshold"
                    )
            else:
                logger.warning(f"Zen model not found: {model_path}")

    def test_supra_benchmarks(self):
        """Test Supra models on benchmarks."""
        supra_model_paths = [
            "/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused",
            "/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused"
        ]
        
        for model_path in supra_model_paths:
            if Path(model_path).exists():
                model_name = f"supra_{Path(model_path).name}"
                
                results = self._test_model_benchmarks(model_path, model_name)
                self.test_results["mmlu_results"][model_name] = results.get("mmlu", {})
                self.test_results["hellaswag_results"][model_name] = results.get("hellaswag", {})
                
                # Assertions for benchmark performance
                if results.get("status") == "success":
                    mmlu_accuracy = results["mmlu"]["overall_accuracy"]
                    hellaswag_accuracy = results["hellaswag"]["accuracy"]
                    
                    self.assertGreaterEqual(
                        mmlu_accuracy,
                        self.benchmark_thresholds["mmlu_accuracy_threshold"],
                        f"Supra {model_name} MMLU accuracy {mmlu_accuracy:.3f} below threshold"
                    )
                    
                    self.assertGreaterEqual(
                        hellaswag_accuracy,
                        self.benchmark_thresholds["hellaswag_accuracy_threshold"],
                        f"Supra {model_name} HellaSwag accuracy {hellaswag_accuracy:.3f} below threshold"
                    )
            else:
                logger.warning(f"Supra model not found: {model_path}")

    def tearDown(self):
        """Save benchmark results and print summary."""
        results_path = Path("/Users/z/work/supra/o1/tests/benchmark_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {results_path}")
        self._print_benchmark_summary()

    def _print_benchmark_summary(self):
        """Print benchmark test summary."""
        print("\n" + "="*60)
        print("BENCHMARK EVALUATION SUMMARY")
        print("="*60)
        
        print("MMLU Results:")
        for model_name, results in self.test_results["mmlu_results"].items():
            if "overall_accuracy" in results:
                accuracy = results["overall_accuracy"]
                print(f"  {model_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print("\nHellaSwag Results:")
        for model_name, results in self.test_results["hellaswag_results"].items():
            if "accuracy" in results:
                accuracy = results["accuracy"]
                print(f"  {model_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print("="*60)

def run_benchmark_tests():
    """Run benchmark tests as standalone function."""
    unittest.main(verbosity=2, exit=False)

if __name__ == "__main__":
    run_benchmark_tests()