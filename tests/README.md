# Zen & Supra Model Evaluation Suite

Complete evaluation framework for both Zen (ZenLM) and Supra (Supra-Nexus) model families. This suite provides comprehensive testing across inference quality, reasoning capabilities, benchmark performance, and format compatibility.

## Test Files

### Core Evaluation Tests

1. **`test_inference.py`** - Basic inference testing for all model formats
   - Tests Transformers and MLX model loading
   - Validates response quality and identity alignment
   - Measures inference speed and response lengths
   - Supports batch processing for efficiency

2. **`test_cot.py`** - Chain-of-thought reasoning quality validation
   - Evaluates step-by-step reasoning transparency
   - Measures logical flow and reasoning indicators
   - Tests mathematical and logical problem solving
   - Validates thinking process visibility

3. **`eval_benchmarks.py`** - Standard benchmark evaluation (MMLU, HellaSwag)
   - MMLU: Massive Multitask Language Understanding
   - HellaSwag: Commonsense reasoning evaluation
   - Multiple choice answer extraction
   - Cross-domain knowledge assessment

4. **`eval_gsm8k.py`** - Mathematical reasoning evaluation
   - Grade School Math 8K dataset
   - Word problem solving capability
   - Numerical answer extraction and validation
   - Step-by-step reasoning analysis

5. **`test_mlx.py`** - MLX format testing (Apple Silicon optimized)
   - MLX model loading and compatibility
   - Performance benchmarking on Apple Silicon
   - Memory usage and inference speed measurement
   - Quantization comparison (4-bit vs full precision)

### Report Generation

6. **`scripts/generate_benchmark_report.py`** - Comprehensive report generator
   - Aggregates results from all evaluation tests
   - Generates JSON and Markdown reports
   - Calculates overall performance scores
   - Provides actionable recommendations

7. **`scripts/run_all_tests.py`** - Test orchestration runner
   - Executes complete evaluation suite
   - Handles test sequencing and error recovery
   - Generates final comprehensive reports
   - Provides execution summaries

## Quick Start

### Run Individual Tests

```bash
# Basic inference testing
cd /Users/z/work/supra/o1/tests
python test_inference.py

# Chain-of-thought evaluation
python test_cot.py

# Benchmark evaluation
python eval_benchmarks.py

# Math reasoning test
python eval_gsm8k.py

# MLX format testing (Apple Silicon only)
python test_mlx.py
```

### Run Complete Evaluation Suite

```bash
# Run all tests and generate reports
cd /Users/z/work/supra/o1/scripts
python run_all_tests.py

# Run tests only (skip report generation)
python run_all_tests.py --skip-report
```

### Generate Reports from Existing Results

```bash
# Generate comprehensive report
cd /Users/z/work/supra/o1/scripts
python generate_benchmark_report.py

# Custom output directory
python generate_benchmark_report.py --output-dir /path/to/reports

# Generate only JSON or Markdown
python generate_benchmark_report.py --json-only
python generate_benchmark_report.py --markdown-only
```

## Requirements

### Dependencies

```bash
# Core dependencies (installed with supra-nexus)
pip install torch transformers pyyaml tqdm

# Optional dependencies for enhanced functionality
pip install mlx mlx-lm        # For MLX testing (Apple Silicon)
pip install psutil           # For memory monitoring
```

### Model Locations

The evaluation suite automatically detects models in these locations:

**Zen Models:**
- `/Users/z/work/zen/zen-nano/models/zen-nano-instruct`
- `/Users/z/work/zen/zen-nano/models/zen-nano-thinking`
- `/Users/z/work/zen/zen-nano/models/zen-nano-4bit`

**Supra Models:**
- `/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused`
- `/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused`
- `/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-mlx`
- `/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-mlx`

## Output Files

### Test Results (JSON)
- `inference_results.json` - Basic inference test results
- `cot_results.json` - Chain-of-thought evaluation results
- `benchmark_results.json` - MMLU and HellaSwag results
- `gsm8k_results.json` - Mathematical reasoning results
- `mlx_results.json` - MLX performance and compatibility results

### Reports
- `reports/benchmark_report_YYYYMMDD_HHMMSS.json` - Comprehensive JSON report
- `reports/benchmark_report_YYYYMMDD_HHMMSS.md` - Human-readable Markdown report

## Quality Thresholds

### Performance Baselines
- **Inference Success Rate**: >80% for core functionality
- **Chain-of-Thought Quality**: >60% transparency score for thinking models
- **MMLU Accuracy**: >25% (above random chance for 4-choice questions)
- **HellaSwag Accuracy**: >30% (above random for commonsense reasoning)
- **GSM8K Accuracy**: >30% for 4B models (mathematical reasoning)
- **MLX Performance**: >10 tokens/sec, <12GB memory usage

### Model-Specific Requirements
- **Thinking Models**: Must demonstrate reasoning transparency in CoT tests
- **Instruct Models**: Must maintain identity alignment and response quality
- **MLX Models**: Must meet Apple Silicon optimization thresholds
- **4-bit Models**: Must achieve >30% memory reduction vs full precision

## CI Integration

Test results are formatted for continuous integration:

```bash
# Run in CI environment with strict assertions
python test_inference.py --verbose

# Generate machine-readable results
python generate_benchmark_report.py --json-only

# Exit codes: 0 = success, 1 = failures detected
echo $?
```

## Troubleshooting

### Common Issues

1. **Model Not Found**: Verify model paths in test files
2. **MLX Import Error**: Install MLX libraries for Apple Silicon testing
3. **Memory Issues**: Reduce batch sizes or test subset of models
4. **Timeout Errors**: Increase timeout values for slower hardware

### Debug Mode

```bash
# Run with verbose logging
python test_inference.py --verbose

# Run single test method
python test_inference.py TestClass.test_method_name

# Check dependencies
python -c "import transformers, torch; print('Dependencies OK')"
```

## Contributing

When adding new evaluation tests:

1. Follow the existing pattern of test classes inheriting from `unittest.TestCase`
2. Include proper error handling and logging
3. Generate JSON results for report integration
4. Add quality thresholds and assertions
5. Support both Zen and Supra model families
6. Document test methodology and expected outputs

## Architecture

The evaluation suite follows a modular design:

```
tests/
├── test_inference.py     # Core inference testing
├── test_cot.py          # Reasoning quality validation
├── eval_benchmarks.py   # Standard benchmarks (MMLU, HellaSwag)
├── eval_gsm8k.py        # Math reasoning (GSM8K)
├── test_mlx.py          # MLX format testing
└── README.md            # This documentation

scripts/
├── generate_benchmark_report.py  # Report aggregation
├── run_all_tests.py              # Test orchestration
└── [output]
    └── reports/         # Generated reports directory
```

Each test module is self-contained and can be run independently, with results saved to JSON files for later aggregation into comprehensive reports.