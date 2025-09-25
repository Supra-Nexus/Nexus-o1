#!/usr/bin/env python3
"""
Test runner script to execute all evaluation tests and generate reports.
Runs the complete evaluation suite for both Zen and Supra models.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """Orchestrate all evaluation tests."""
    
    def __init__(self, tests_dir: Path, scripts_dir: Path):
        self.tests_dir = Path(tests_dir)
        self.scripts_dir = Path(scripts_dir)
        self.results = {
            "test_execution_results": {},
            "start_time": time.time(),
            "end_time": None,
            "total_duration": None
        }
        
        # Test files to run in order
        self.test_files = [
            "test_inference.py",
            "test_cot.py", 
            "eval_benchmarks.py",
            "eval_gsm8k.py",
            "test_mlx.py"
        ]

    def run_single_test(self, test_file: str) -> Dict:
        """Run a single test file and return results."""
        test_path = self.tests_dir / test_file
        
        if not test_path.exists():
            logger.error(f"Test file not found: {test_path}")
            return {"status": "failed", "reason": "file_not_found"}
        
        logger.info(f"🧪 Running {test_file}...")
        start_time = time.time()
        
        try:
            # Run the test with minimal output
            result = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=str(self.tests_dir),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout per test
            )
            
            duration = time.time() - start_time
            
            return {
                "status": "completed",
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {test_file} timed out after 30 minutes")
            return {
                "status": "timeout", 
                "duration": time.time() - start_time,
                "success": False
            }
        except Exception as e:
            logger.error(f"❌ {test_file} failed with exception: {e}")
            return {
                "status": "error",
                "duration": time.time() - start_time,
                "error": str(e),
                "success": False
            }

    def run_all_tests(self) -> Dict:
        """Run all tests in sequence."""
        logger.info("🚀 Starting comprehensive evaluation suite...")
        logger.info(f"Tests directory: {self.tests_dir}")
        logger.info(f"Running {len(self.test_files)} test files")
        print()
        
        for test_file in self.test_files:
            result = self.run_single_test(test_file)
            self.results["test_execution_results"][test_file] = result
            
            # Log results
            if result["success"]:
                logger.info(f"✅ {test_file} completed successfully ({result['duration']:.1f}s)")
            else:
                logger.error(f"❌ {test_file} failed ({result.get('duration', 0):.1f}s)")
                if result.get("stderr"):
                    logger.error(f"   Error: {result['stderr'][:200]}...")
        
        self.results["end_time"] = time.time()
        self.results["total_duration"] = self.results["end_time"] - self.results["start_time"]
        
        return self.results

    def generate_report(self) -> bool:
        """Generate comprehensive benchmark report."""
        logger.info("📊 Generating comprehensive benchmark report...")
        
        report_script = self.scripts_dir / "generate_benchmark_report.py"
        
        if not report_script.exists():
            logger.error(f"Report generator not found: {report_script}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(report_script)],
                cwd=str(self.scripts_dir),
                timeout=300  # 5 minutes for report generation
            )
            
            if result.returncode == 0:
                logger.info("✅ Benchmark report generated successfully")
                return True
            else:
                logger.error(f"❌ Report generation failed (return code: {result.returncode})")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Report generation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return False

    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*70)
        print("EVALUATION SUITE EXECUTION SUMMARY")
        print("="*70)
        
        total_tests = len(self.test_files)
        successful_tests = sum(1 for result in self.results["test_execution_results"].values() 
                             if result.get("success", False))
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Total Duration: {self.results.get('total_duration', 0):.1f} seconds")
        print()
        
        print("Individual Test Results:")
        for test_file, result in self.results["test_execution_results"].items():
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            duration = result.get("duration", 0)
            print(f"  {test_file:<25} {status} ({duration:.1f}s)")
        
        print()
        
        if successful_tests == total_tests:
            print("🎉 All tests completed successfully!")
        else:
            print(f"⚠️  {total_tests - successful_tests} tests failed or had issues")
        
        print("="*70)

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete evaluation suite")
    parser.add_argument("--tests-dir", default="/Users/z/work/supra/o1/tests",
                       help="Directory containing test files")
    parser.add_argument("--scripts-dir", default="/Users/z/work/supra/o1/scripts", 
                       help="Directory containing scripts")
    parser.add_argument("--skip-report", action="store_true",
                       help="Skip report generation")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = TestRunner(args.tests_dir, args.scripts_dir)
    
    try:
        # Run all tests
        results = runner.run_all_tests()
        
        # Generate report unless skipped
        if not args.skip_report:
            runner.generate_report()
        
        # Print summary
        runner.print_summary()
        
        # Exit with appropriate code
        successful_tests = sum(1 for result in results["test_execution_results"].values() 
                             if result.get("success", False))
        total_tests = len(runner.test_files)
        
        if successful_tests == total_tests:
            logger.info("🎯 Evaluation suite completed successfully")
            sys.exit(0)
        else:
            logger.error("🚫 Evaluation suite completed with failures")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Evaluation suite interrupted by user")
        runner.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Evaluation suite failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()