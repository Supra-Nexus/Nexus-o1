#!/usr/bin/env python3
"""
Comprehensive benchmark report generator for Zen and Supra models.
Aggregates results from all evaluation tests and generates formatted reports.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkReportGenerator:
    """Generate comprehensive benchmark reports from test results."""
    
    def __init__(self, tests_dir: Path):
        self.tests_dir = Path(tests_dir)
        self.report_data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator_version": "1.0.0",
                "tests_directory": str(tests_dir)
            },
            "test_results": {},
            "summary": {},
            "recommendations": []
        }
        
        # Result file mapping
        self.result_files = {
            "inference": "inference_results.json",
            "cot": "cot_results.json",
            "benchmarks": "benchmark_results.json",
            "gsm8k": "gsm8k_results.json",
            "mlx": "mlx_results.json"
        }

    def load_test_results(self) -> bool:
        """Load all available test results."""
        logger.info("Loading test results from JSON files...")
        
        loaded_count = 0
        for test_name, filename in self.result_files.items():
            filepath = self.tests_dir / filename
            
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.report_data["test_results"][test_name] = data
                    loaded_count += 1
                    logger.info(f"✅ Loaded {test_name} results")
                except Exception as e:
                    logger.error(f"❌ Failed to load {test_name} results: {e}")
                    self.report_data["test_results"][test_name] = {"error": str(e)}
            else:
                logger.warning(f"⚠️  {test_name} results not found: {filepath}")
                self.report_data["test_results"][test_name] = {"status": "not_run"}
        
        logger.info(f"Loaded {loaded_count}/{len(self.result_files)} test result files")
        return loaded_count > 0

    def _extract_model_scores(self) -> Dict[str, Dict]:
        """Extract and normalize scores for all models across tests."""
        model_scores = {}
        
        # Process inference results
        if "inference" in self.report_data["test_results"]:
            inference_data = self.report_data["test_results"]["inference"]
            for model_family in ["zen_models", "supra_models"]:
                if model_family in inference_data:
                    for model_name, model_data in inference_data[model_family].items():
                        if model_name not in model_scores:
                            model_scores[model_name] = {"family": model_family.replace("_models", "")}
                        
                        # Calculate inference success rate
                        if "transformers" in model_data and model_data["transformers"].get("status") == "success":
                            responses = model_data["transformers"].get("responses", {})
                            passed_checks = sum(1 for r in responses.values() if r.get("passes_quality_check"))
                            total_checks = len(responses)
                            model_scores[model_name]["inference_success_rate"] = passed_checks / total_checks if total_checks > 0 else 0
                        else:
                            model_scores[model_name]["inference_success_rate"] = 0
        
        # Process CoT results
        if "cot" in self.report_data["test_results"]:
            cot_data = self.report_data["test_results"]["cot"].get("cot_evaluation", {})
            for model_name, model_data in cot_data.items():
                if model_name not in model_scores:
                    model_scores[model_name] = {"family": "unknown"}
                
                summary = model_data.get("summary", {})
                model_scores[model_name]["cot_score"] = summary.get("average_cot_score", 0)
                model_scores[model_name]["cot_passed"] = summary.get("passed_threshold", False)
        
        # Process benchmark results
        if "benchmarks" in self.report_data["test_results"]:
            benchmark_data = self.report_data["test_results"]["benchmarks"]
            
            # MMLU scores
            for model_name, model_data in benchmark_data.get("mmlu_results", {}).items():
                if model_name not in model_scores:
                    model_scores[model_name] = {"family": "unknown"}
                model_scores[model_name]["mmlu_accuracy"] = model_data.get("overall_accuracy", 0)
            
            # HellaSwag scores
            for model_name, model_data in benchmark_data.get("hellaswag_results", {}).items():
                if model_name not in model_scores:
                    model_scores[model_name] = {"family": "unknown"}
                model_scores[model_name]["hellaswag_accuracy"] = model_data.get("accuracy", 0)
        
        # Process GSM8K results
        if "gsm8k" in self.report_data["test_results"]:
            gsm8k_data = self.report_data["test_results"]["gsm8k"].get("gsm8k_results", {})
            for model_name, model_data in gsm8k_data.items():
                if model_name not in model_scores:
                    model_scores[model_name] = {"family": "unknown"}
                
                if model_data.get("status") == "success":
                    model_scores[model_name]["gsm8k_accuracy"] = model_data.get("accuracy", 0)
                    model_scores[model_name]["gsm8k_reasoning_pct"] = model_data.get("reasoning_percentage", 0)
                else:
                    model_scores[model_name]["gsm8k_accuracy"] = 0
                    model_scores[model_name]["gsm8k_reasoning_pct"] = 0
        
        # Process MLX performance results
        if "mlx" in self.report_data["test_results"]:
            mlx_data = self.report_data["test_results"]["mlx"].get("mlx_compatibility", {})
            for model_name, model_data in mlx_data.items():
                if model_name not in model_scores:
                    model_scores[model_name] = {"family": "unknown"}
                
                perf_summary = model_data.get("performance_summary", {})
                if perf_summary:
                    model_scores[model_name]["mlx_tokens_per_sec"] = perf_summary.get("average_tokens_per_second", 0)
                    model_scores[model_name]["mlx_memory_gb"] = perf_summary.get("max_memory_usage_gb", 0)
                    model_scores[model_name]["mlx_load_time"] = perf_summary.get("load_time", 0)
                    
                    # Overall MLX performance score
                    thresholds = perf_summary.get("meets_performance_thresholds", {})
                    passed_thresholds = sum(1 for passed in thresholds.values() if passed)
                    total_thresholds = len(thresholds)
                    model_scores[model_name]["mlx_performance_score"] = passed_thresholds / total_thresholds if total_thresholds > 0 else 0
        
        return model_scores

    def _calculate_overall_scores(self, model_scores: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate overall performance scores for each model."""
        for model_name, scores in model_scores.items():
            # Define score components with weights
            score_components = [
                ("inference_success_rate", 0.15),
                ("cot_score", 0.20),
                ("mmlu_accuracy", 0.20),
                ("hellaswag_accuracy", 0.15),
                ("gsm8k_accuracy", 0.20),
                ("mlx_performance_score", 0.10)
            ]
            
            total_score = 0
            total_weight = 0
            
            for score_key, weight in score_components:
                if score_key in scores:
                    total_score += scores[score_key] * weight
                    total_weight += weight
            
            # Normalize to available components
            scores["overall_score"] = total_score / total_weight if total_weight > 0 else 0
            
            # Calculate grade based on overall score
            if scores["overall_score"] >= 0.8:
                scores["grade"] = "A"
            elif scores["overall_score"] >= 0.7:
                scores["grade"] = "B"
            elif scores["overall_score"] >= 0.6:
                scores["grade"] = "C"
            elif scores["overall_score"] >= 0.5:
                scores["grade"] = "D"
            else:
                scores["grade"] = "F"
        
        return model_scores

    def _generate_summary_statistics(self, model_scores: Dict[str, Dict]) -> Dict:
        """Generate summary statistics across all models."""
        zen_models = {k: v for k, v in model_scores.items() if v.get("family") == "zen"}
        supra_models = {k: v for k, v in model_scores.items() if v.get("family") == "supra"}
        
        def calculate_family_stats(models: Dict) -> Dict:
            if not models:
                return {"count": 0, "avg_overall_score": 0, "grade_distribution": {}}
            
            overall_scores = [m.get("overall_score", 0) for m in models.values()]
            grades = [m.get("grade", "F") for m in models.values()]
            
            from collections import Counter
            grade_dist = Counter(grades)
            
            return {
                "count": len(models),
                "avg_overall_score": sum(overall_scores) / len(overall_scores),
                "grade_distribution": dict(grade_dist),
                "best_model": max(models.keys(), key=lambda k: models[k].get("overall_score", 0)) if models else None,
                "avg_scores": {
                    "inference": sum(m.get("inference_success_rate", 0) for m in models.values()) / len(models),
                    "cot": sum(m.get("cot_score", 0) for m in models.values()) / len(models),
                    "mmlu": sum(m.get("mmlu_accuracy", 0) for m in models.values()) / len(models),
                    "hellaswag": sum(m.get("hellaswag_accuracy", 0) for m in models.values()) / len(models),
                    "gsm8k": sum(m.get("gsm8k_accuracy", 0) for m in models.values()) / len(models)
                }
            }
        
        return {
            "zen_family": calculate_family_stats(zen_models),
            "supra_family": calculate_family_stats(supra_models),
            "total_models_tested": len(model_scores),
            "test_completion": {
                "inference": len([m for m in model_scores.values() if "inference_success_rate" in m]),
                "cot": len([m for m in model_scores.values() if "cot_score" in m]),
                "benchmarks": len([m for m in model_scores.values() if "mmlu_accuracy" in m]),
                "gsm8k": len([m for m in model_scores.values() if "gsm8k_accuracy" in m]),
                "mlx": len([m for m in model_scores.values() if "mlx_performance_score" in m])
            }
        }

    def _generate_recommendations(self, model_scores: Dict[str, Dict], summary: Dict) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Model performance recommendations
        zen_avg = summary["zen_family"]["avg_overall_score"]
        supra_avg = summary["supra_family"]["avg_overall_score"]
        
        if supra_avg > zen_avg + 0.1:
            recommendations.append("🎯 Supra models show superior overall performance. Consider prioritizing Supra model development.")
        elif zen_avg > supra_avg + 0.1:
            recommendations.append("🎯 Zen models show superior overall performance. Consider prioritizing Zen model development.")
        else:
            recommendations.append("⚖️ Both model families show similar performance. Continue balanced development.")
        
        # Specific capability recommendations
        zen_gsm8k = summary["zen_family"]["avg_scores"]["gsm8k"]
        supra_gsm8k = summary["supra_family"]["avg_scores"]["gsm8k"]
        
        if zen_gsm8k < 0.3:
            recommendations.append("📊 Zen models need improvement in mathematical reasoning (GSM8K). Consider additional math training data.")
        
        if supra_gsm8k < 0.3:
            recommendations.append("📊 Supra models need improvement in mathematical reasoning (GSM8K). Consider additional math training data.")
        
        # CoT recommendations
        thinking_models = [name for name, scores in model_scores.items() if "thinking" in name.lower()]
        if thinking_models:
            thinking_cot_scores = [model_scores[name].get("cot_score", 0) for name in thinking_models]
            avg_thinking_cot = sum(thinking_cot_scores) / len(thinking_cot_scores) if thinking_cot_scores else 0
            
            if avg_thinking_cot < 0.6:
                recommendations.append("🧠 Thinking models show low chain-of-thought quality. Consider improving reasoning transparency training.")
        
        # MLX optimization recommendations
        mlx_tested = summary["test_completion"]["mlx"]
        if mlx_tested > 0:
            mlx_models = {k: v for k, v in model_scores.items() if "mlx_performance_score" in v}
            poor_mlx_models = [k for k, v in mlx_models.items() if v["mlx_performance_score"] < 0.7]
            
            if poor_mlx_models:
                recommendations.append(f"⚡ {len(poor_mlx_models)} models have suboptimal MLX performance. Consider MLX-specific optimizations.")
        
        # Benchmark-specific recommendations
        mmlu_scores = [v.get("mmlu_accuracy", 0) for v in model_scores.values() if "mmlu_accuracy" in v]
        if mmlu_scores and sum(mmlu_scores) / len(mmlu_scores) < 0.3:
            recommendations.append("📚 Overall MMLU performance is low. Consider broader knowledge training or curriculum improvements.")
        
        hellaswag_scores = [v.get("hellaswag_accuracy", 0) for v in model_scores.values() if "hellaswag_accuracy" in v]
        if hellaswag_scores and sum(hellaswag_scores) / len(hellaswag_scores) < 0.4:
            recommendations.append("🎭 HellaSwag (commonsense reasoning) performance needs improvement. Consider scenario-based training data.")
        
        return recommendations

    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report."""
        if not self.load_test_results():
            logger.error("No test results available for report generation")
            return {}
        
        logger.info("Extracting model scores...")
        model_scores = self._extract_model_scores()
        
        logger.info("Calculating overall scores...")
        model_scores = self._calculate_overall_scores(model_scores)
        
        logger.info("Generating summary statistics...")
        summary = self._generate_summary_statistics(model_scores)
        
        logger.info("Generating recommendations...")
        recommendations = self._generate_recommendations(model_scores, summary)
        
        # Update report data
        self.report_data["summary"] = summary
        self.report_data["model_scores"] = model_scores
        self.report_data["recommendations"] = recommendations
        
        return self.report_data

    def save_json_report(self, output_path: Path) -> bool:
        """Save comprehensive JSON report."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.report_data, f, indent=2)
            logger.info(f"✅ JSON report saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save JSON report: {e}")
            return False

    def save_markdown_report(self, output_path: Path) -> bool:
        """Save human-readable Markdown report."""
        try:
            with open(output_path, 'w') as f:
                f.write(self._generate_markdown_content())
            logger.info(f"✅ Markdown report saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save Markdown report: {e}")
            return False

    def _generate_markdown_content(self) -> str:
        """Generate Markdown content for the report."""
        md_content = []
        
        # Header
        md_content.append("# Zen & Supra Models - Comprehensive Benchmark Report")
        md_content.append(f"Generated: {self.report_data['metadata']['generated_at']}")
        md_content.append("")
        
        # Executive Summary
        md_content.append("## Executive Summary")
        md_content.append("")
        summary = self.report_data["summary"]
        
        md_content.append(f"**Total Models Tested:** {summary['total_models_tested']}")
        md_content.append(f"**Zen Models:** {summary['zen_family']['count']} (Avg Score: {summary['zen_family']['avg_overall_score']:.3f})")
        md_content.append(f"**Supra Models:** {summary['supra_family']['count']} (Avg Score: {summary['supra_family']['avg_overall_score']:.3f})")
        md_content.append("")
        
        # Best performers
        zen_best = summary["zen_family"].get("best_model")
        supra_best = summary["supra_family"].get("best_model")
        
        if zen_best or supra_best:
            md_content.append("### Top Performers")
            if zen_best:
                zen_score = self.report_data["model_scores"][zen_best]["overall_score"]
                md_content.append(f"- **Best Zen Model:** {zen_best} (Score: {zen_score:.3f})")
            if supra_best:
                supra_score = self.report_data["model_scores"][supra_best]["overall_score"]
                md_content.append(f"- **Best Supra Model:** {supra_best} (Score: {supra_score:.3f})")
            md_content.append("")
        
        # Detailed Results Table
        md_content.append("## Detailed Model Performance")
        md_content.append("")
        md_content.append("| Model | Family | Overall | Grade | Inference | CoT | MMLU | HellaSwag | GSM8K | MLX |")
        md_content.append("|-------|--------|---------|-------|-----------|-----|------|-----------|-------|-----|")
        
        for model_name in sorted(self.report_data["model_scores"].keys()):
            scores = self.report_data["model_scores"][model_name]
            
            row = [
                model_name,
                scores.get("family", "unknown"),
                f"{scores.get('overall_score', 0):.3f}",
                scores.get("grade", "F"),
                f"{scores.get('inference_success_rate', 0):.2f}",
                f"{scores.get('cot_score', 0):.2f}",
                f"{scores.get('mmlu_accuracy', 0):.2f}",
                f"{scores.get('hellaswag_accuracy', 0):.2f}",
                f"{scores.get('gsm8k_accuracy', 0):.2f}",
                f"{scores.get('mlx_performance_score', 0):.2f}" if "mlx_performance_score" in scores else "N/A"
            ]
            md_content.append("| " + " | ".join(row) + " |")
        
        md_content.append("")
        
        # Test Coverage
        md_content.append("## Test Coverage")
        md_content.append("")
        completion = summary["test_completion"]
        for test_name, count in completion.items():
            md_content.append(f"- **{test_name.title()}:** {count} models tested")
        md_content.append("")
        
        # Recommendations
        if self.report_data["recommendations"]:
            md_content.append("## Recommendations")
            md_content.append("")
            for rec in self.report_data["recommendations"]:
                md_content.append(f"- {rec}")
            md_content.append("")
        
        # Technical Details
        md_content.append("## Technical Details")
        md_content.append("")
        md_content.append("### Scoring Methodology")
        md_content.append("Overall scores are calculated as weighted averages:")
        md_content.append("- Inference Success Rate: 15%")
        md_content.append("- Chain-of-Thought Quality: 20%")
        md_content.append("- MMLU Accuracy: 20%")
        md_content.append("- HellaSwag Accuracy: 15%")
        md_content.append("- GSM8K Accuracy: 20%")
        md_content.append("- MLX Performance: 10%")
        md_content.append("")
        
        md_content.append("### Grade Scale")
        md_content.append("- A: ≥0.8 (Excellent)")
        md_content.append("- B: ≥0.7 (Good)")
        md_content.append("- C: ≥0.6 (Average)")
        md_content.append("- D: ≥0.5 (Below Average)")
        md_content.append("- F: <0.5 (Poor)")
        md_content.append("")
        
        return "\n".join(md_content)

    def print_summary(self):
        """Print a concise summary to console."""
        if not self.report_data["summary"]:
            print("❌ No report data available")
            return
        
        print("\n" + "="*70)
        print("ZEN & SUPRA MODELS - BENCHMARK REPORT SUMMARY")
        print("="*70)
        
        summary = self.report_data["summary"]
        
        print(f"Total Models Tested: {summary['total_models_tested']}")
        print()
        
        # Family comparison
        zen_stats = summary["zen_family"]
        supra_stats = summary["supra_family"]
        
        print("Family Performance:")
        print(f"  Zen Models:   {zen_stats['count']} models, avg score {zen_stats['avg_overall_score']:.3f}")
        print(f"  Supra Models: {supra_stats['count']} models, avg score {supra_stats['avg_overall_score']:.3f}")
        print()
        
        # Top performers
        zen_best = zen_stats.get("best_model")
        supra_best = supra_stats.get("best_model")
        
        print("Best Performers:")
        if zen_best:
            zen_score = self.report_data["model_scores"][zen_best]["overall_score"]
            print(f"  Zen:   {zen_best} ({zen_score:.3f})")
        if supra_best:
            supra_score = self.report_data["model_scores"][supra_best]["overall_score"]
            print(f"  Supra: {supra_best} ({supra_score:.3f})")
        print()
        
        # Key recommendations
        if self.report_data["recommendations"]:
            print("Key Recommendations:")
            for i, rec in enumerate(self.report_data["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
        
        print("="*70)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate comprehensive benchmark reports")
    parser.add_argument("--tests-dir", default="/Users/z/work/supra/o1/tests",
                       help="Directory containing test result JSON files")
    parser.add_argument("--output-dir", default="/Users/z/work/supra/o1/reports",
                       help="Directory to save generated reports")
    parser.add_argument("--json-only", action="store_true",
                       help="Generate only JSON report")
    parser.add_argument("--markdown-only", action="store_true",
                       help="Generate only Markdown report")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate report
    generator = BenchmarkReportGenerator(args.tests_dir)
    report_data = generator.generate_report()
    
    if not report_data:
        logger.error("Report generation failed")
        sys.exit(1)
    
    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success_count = 0
    
    if not args.markdown_only:
        json_path = output_dir / f"benchmark_report_{timestamp}.json"
        if generator.save_json_report(json_path):
            success_count += 1
    
    if not args.json_only:
        md_path = output_dir / f"benchmark_report_{timestamp}.md"
        if generator.save_markdown_report(md_path):
            success_count += 1
    
    # Print summary
    generator.print_summary()
    
    if success_count > 0:
        print(f"\n✅ Reports generated successfully in: {output_dir}")
    else:
        print("\n❌ Report generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()