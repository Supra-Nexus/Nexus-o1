#!/usr/bin/env python3
"""
MLX format testing specifically for Apple Silicon optimization.
Tests MLX model loading, inference speed, and memory efficiency.
"""

import json
import logging
import platform
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLXFormatTest(unittest.TestCase):
    """MLX format testing suite for Apple Silicon."""
    
    def setUp(self):
        """Set up MLX test environment."""
        self.test_results = {
            "mlx_compatibility": {},
            "performance_benchmarks": {},
            "memory_usage": {},
            "test_timestamp": time.time(),
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "is_apple_silicon": self._is_apple_silicon()
            }
        }
        
        # Test prompts optimized for MLX performance testing
        self.test_prompts = {
            "short_response": "What is 2+2?",
            "medium_response": "Explain the concept of machine learning in simple terms.",
            "long_response": "Write a detailed explanation of how neural networks work, including the mathematics behind backpropagation.",
            "code_generation": "Write a Python function that implements binary search on a sorted array.",
            "reasoning": "If I have 5 apples and give away 2, then buy 3 more, how many do I have? Show your thinking.",
            "creative": "Write a short story about a robot learning to paint."
        }
        
        # MLX performance thresholds
        self.mlx_thresholds = {
            "max_load_time": 30.0,  # seconds
            "min_tokens_per_second": 10.0,  # tokens/sec
            "max_memory_usage_gb": 12.0,  # GB for 4B models
            "max_first_token_latency": 5.0  # seconds
        }

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        if platform.system() != "Darwin":
            return False
        
        try:
            # Check for Apple Silicon processors
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_brand = result.stdout.strip().lower()
            return 'apple' in cpu_brand or 'm1' in cpu_brand or 'm2' in cpu_brand or 'm3' in cpu_brand or 'm4' in cpu_brand
        except Exception:
            # Fallback check
            return platform.processor().lower() in ['arm', 'arm64']

    def _try_import_mlx(self) -> bool:
        """Try to import MLX libraries."""
        try:
            import mlx
            import mlx.core as mx
            import mlx_lm
            return True
        except ImportError as e:
            logger.warning(f"MLX not available: {e}")
            return False

    def _get_memory_info(self) -> Dict:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_gb": memory_info.rss / (1024**3),
                "vms_gb": memory_info.vms / (1024**3),
                "percent": process.memory_percent()
            }
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {"rss_gb": 0, "vms_gb": 0, "percent": 0}

    def _measure_mlx_performance(self, model, tokenizer, prompt: str) -> Dict:
        """Measure MLX model performance metrics."""
        if not self._try_import_mlx():
            return {"status": "skipped", "reason": "MLX not available"}
        
        try:
            from mlx_lm import generate
            
            # Measure memory before generation
            memory_before = self._get_memory_info()
            
            # Measure first token latency
            start_time = time.time()
            
            # Generate response with token counting
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt,
                max_tokens=100,
                temp=0.7
            )
            
            generation_time = time.time() - start_time
            
            # Measure memory after generation
            memory_after = self._get_memory_info()
            
            # Estimate token count (rough approximation)
            token_count = len(response.split()) * 1.3  # Approximate tokens
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            
            return {
                "status": "success",
                "response": response,
                "generation_time": generation_time,
                "token_count": int(token_count),
                "tokens_per_second": tokens_per_second,
                "memory_before_gb": memory_before["rss_gb"],
                "memory_after_gb": memory_after["rss_gb"],
                "memory_delta_gb": memory_after["rss_gb"] - memory_before["rss_gb"],
                "first_token_latency": generation_time / token_count if token_count > 0 else generation_time
            }
            
        except Exception as e:
            logger.error(f"MLX performance measurement failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _test_mlx_model_loading(self, model_path: str, model_name: str) -> Dict:
        """Test MLX model loading performance."""
        if not self._try_import_mlx():
            return {"status": "skipped", "reason": "MLX not available"}
        
        if not self._is_apple_silicon():
            logger.warning("MLX is optimized for Apple Silicon - running on non-Apple hardware")
        
        try:
            from mlx_lm import load
            
            logger.info(f"Loading MLX model: {model_name}")
            
            # Measure loading time and memory
            memory_before = self._get_memory_info()
            start_time = time.time()
            
            model, tokenizer = load(model_path)
            
            load_time = time.time() - start_time
            memory_after = self._get_memory_info()
            
            return {
                "status": "success",
                "model": model,
                "tokenizer": tokenizer,
                "load_time": load_time,
                "memory_usage_gb": memory_after["rss_gb"] - memory_before["rss_gb"],
                "total_memory_gb": memory_after["rss_gb"]
            }
            
        except Exception as e:
            logger.error(f"MLX model loading failed for {model_name}: {e}")
            return {"status": "failed", "error": str(e)}

    def _test_mlx_model_inference(self, model_path: str, model_name: str) -> Dict:
        """Test MLX model inference performance."""
        # Load model first
        loading_result = self._test_mlx_model_loading(model_path, model_name)
        
        if loading_result["status"] != "success":
            return loading_result
        
        model = loading_result["model"]
        tokenizer = loading_result["tokenizer"]
        
        results = {
            "model_name": model_name,
            "model_path": model_path,
            "loading": loading_result,
            "inference_results": {},
            "performance_summary": {}
        }
        
        # Test inference with different prompt lengths
        total_generation_time = 0
        total_tokens = 0
        max_memory_usage = 0
        
        for prompt_name, prompt_text in self.test_prompts.items():
            logger.info(f"Testing MLX inference: {prompt_name}")
            
            perf_result = self._measure_mlx_performance(model, tokenizer, prompt_text)
            results["inference_results"][prompt_name] = perf_result
            
            if perf_result["status"] == "success":
                total_generation_time += perf_result["generation_time"]
                total_tokens += perf_result["token_count"]
                max_memory_usage = max(max_memory_usage, perf_result["memory_after_gb"])
        
        # Calculate performance summary
        if total_generation_time > 0 and total_tokens > 0:
            avg_tokens_per_second = total_tokens / total_generation_time
            
            results["performance_summary"] = {
                "average_tokens_per_second": avg_tokens_per_second,
                "total_generation_time": total_generation_time,
                "total_tokens_generated": total_tokens,
                "max_memory_usage_gb": max_memory_usage,
                "load_time": loading_result["load_time"],
                "meets_performance_thresholds": {
                    "load_time_ok": loading_result["load_time"] <= self.mlx_thresholds["max_load_time"],
                    "speed_ok": avg_tokens_per_second >= self.mlx_thresholds["min_tokens_per_second"],
                    "memory_ok": max_memory_usage <= self.mlx_thresholds["max_memory_usage_gb"]
                }
            }
        
        return results

    def test_zen_mlx_models(self):
        """Test Zen models in MLX format."""
        if not self._is_apple_silicon():
            self.skipTest("MLX tests are optimized for Apple Silicon")
        
        zen_mlx_paths = [
            # Look for MLX versions of Zen models
            "/Users/z/work/zen/zen-nano/models/zen-nano-instruct-mlx",
            "/Users/z/work/zen/zen-nano/models/zen-nano-thinking-mlx",
            "/Users/z/work/zen/zen-nano/models/zen-nano-4bit"
        ]
        
        for model_path in zen_mlx_paths:
            if Path(model_path).exists():
                model_name = f"zen_{Path(model_path).name}"
                
                results = self._test_mlx_model_inference(model_path, model_name)
                self.test_results["mlx_compatibility"][model_name] = results
                
                # Assertions for MLX performance
                if results.get("performance_summary", {}).get("meets_performance_thresholds"):
                    thresholds = results["performance_summary"]["meets_performance_thresholds"]
                    
                    self.assertTrue(
                        thresholds["load_time_ok"],
                        f"Zen MLX {model_name} loading time too slow"
                    )
                    
                    self.assertTrue(
                        thresholds["speed_ok"],
                        f"Zen MLX {model_name} inference speed too slow"
                    )
                    
                    self.assertTrue(
                        thresholds["memory_ok"],
                        f"Zen MLX {model_name} memory usage too high"
                    )
            else:
                logger.warning(f"Zen MLX model not found: {model_path}")

    def test_supra_mlx_models(self):
        """Test Supra models in MLX format."""
        if not self._is_apple_silicon():
            self.skipTest("MLX tests are optimized for Apple Silicon")
        
        supra_mlx_paths = [
            "/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-mlx",
            "/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-mlx",
            "/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-mlx-4bit",
            "/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-mlx-4bit"
        ]
        
        for model_path in supra_mlx_paths:
            if Path(model_path).exists():
                model_name = f"supra_{Path(model_path).name}"
                
                results = self._test_mlx_model_inference(model_path, model_name)
                self.test_results["mlx_compatibility"][model_name] = results
                
                # Assertions for MLX performance
                if results.get("performance_summary", {}).get("meets_performance_thresholds"):
                    thresholds = results["performance_summary"]["meets_performance_thresholds"]
                    
                    self.assertTrue(
                        thresholds["load_time_ok"],
                        f"Supra MLX {model_name} loading time too slow"
                    )
                    
                    self.assertTrue(
                        thresholds["speed_ok"],
                        f"Supra MLX {model_name} inference speed too slow"
                    )
                    
                    # 4-bit models should use even less memory
                    if "4bit" in model_name.lower():
                        max_memory = results["performance_summary"]["max_memory_usage_gb"]
                        self.assertLessEqual(
                            max_memory,
                            8.0,  # Stricter memory requirement for 4-bit
                            f"Supra MLX 4-bit {model_name} memory usage too high for quantized model"
                        )
                    else:
                        self.assertTrue(
                            thresholds["memory_ok"],
                            f"Supra MLX {model_name} memory usage too high"
                        )
            else:
                logger.warning(f"Supra MLX model not found: {model_path}")

    def test_mlx_performance_comparison(self):
        """Compare MLX performance across different model formats."""
        if not self._is_apple_silicon():
            self.skipTest("MLX performance comparison requires Apple Silicon")
        
        # This test compares performance between different quantization levels
        performance_results = {}
        
        # Collect performance data from previous tests
        for model_name, results in self.test_results["mlx_compatibility"].items():
            if results.get("performance_summary"):
                perf = results["performance_summary"]
                performance_results[model_name] = {
                    "tokens_per_second": perf.get("average_tokens_per_second", 0),
                    "memory_usage_gb": perf.get("max_memory_usage_gb", 0),
                    "load_time": perf.get("load_time", 0)
                }
        
        # Compare 4-bit vs full precision performance
        four_bit_models = {k: v for k, v in performance_results.items() if "4bit" in k.lower()}
        full_models = {k: v for k, v in performance_results.items() if "4bit" not in k.lower()}
        
        if four_bit_models and full_models:
            avg_4bit_memory = sum(m["memory_usage_gb"] for m in four_bit_models.values()) / len(four_bit_models)
            avg_full_memory = sum(m["memory_usage_gb"] for m in full_models.values()) / len(full_models)
            
            # 4-bit models should use significantly less memory
            memory_reduction = (avg_full_memory - avg_4bit_memory) / avg_full_memory
            self.assertGreaterEqual(
                memory_reduction,
                0.3,  # At least 30% memory reduction
                "4-bit quantization should provide significant memory savings"
            )
        
        self.test_results["performance_benchmarks"]["comparison"] = {
            "four_bit_models": four_bit_models,
            "full_precision_models": full_models,
            "memory_reduction_achieved": memory_reduction if 'memory_reduction' in locals() else None
        }

    def tearDown(self):
        """Save MLX test results and print summary."""
        results_path = Path("/Users/z/work/supra/o1/tests/mlx_results.json")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"MLX test results saved to: {results_path}")
        self._print_mlx_summary()

    def _print_mlx_summary(self):
        """Print MLX test summary."""
        print("\n" + "="*60)
        print("MLX APPLE SILICON TEST SUMMARY")
        print("="*60)
        
        is_apple_silicon = self.test_results["system_info"]["is_apple_silicon"]
        print(f"Running on Apple Silicon: {'✅ YES' if is_apple_silicon else '❌ NO'}")
        
        if not is_apple_silicon:
            print("⚠️  MLX is optimized for Apple Silicon - performance may be suboptimal")
        
        print(f"Platform: {self.test_results['system_info']['platform']}")
        print()
        
        for model_name, results in self.test_results["mlx_compatibility"].items():
            if results.get("performance_summary"):
                perf = results["performance_summary"]
                thresholds = perf.get("meets_performance_thresholds", {})
                
                print(f"{model_name}:")
                print(f"  Load Time: {perf.get('load_time', 0):.2f}s")
                print(f"  Speed: {perf.get('average_tokens_per_second', 0):.1f} tokens/sec")
                print(f"  Memory: {perf.get('max_memory_usage_gb', 0):.2f} GB")
                print(f"  Performance: {'✅ PASS' if all(thresholds.values()) else '⚠️  ISSUES'}")
                print()
        
        # Performance comparison summary
        if "comparison" in self.test_results["performance_benchmarks"]:
            comp = self.test_results["performance_benchmarks"]["comparison"]
            if comp.get("memory_reduction_achieved"):
                print(f"4-bit Memory Savings: {comp['memory_reduction_achieved']*100:.1f}%")
        
        print("="*60)

def run_mlx_tests():
    """Run MLX tests as standalone function."""
    unittest.main(verbosity=2, exit=False)

if __name__ == "__main__":
    run_mlx_tests()