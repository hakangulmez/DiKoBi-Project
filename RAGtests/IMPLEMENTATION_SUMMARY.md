# Test Folder Implementation - Summary

## ✅ Implementation Complete

A comprehensive test suite for finding optimal few-shot prompting strategies has been implemented.

## 📁 What Was Created

```
tests/
├── README.md                      # Comprehensive documentation
├── QUICKREF.md                    # Quick reference guide
├── config.py                      # Configuration settings
├── __init__.py                    # Package initialization
│
├── Core Components:
├── few_shot_selector.py           # 8 example selection strategies
├── mse_evaluator.py               # MSE-focused evaluation with detailed analysis
│
├── Prompt Templates:
├── prompts/
│   ├── __init__.py
│   └── few_shot_templates.py     # 5 prompt format generators
│
├── Execution Scripts:
├── run_experiments.py             # Main experiment runner (full suite)
├── quick_test.py                  # Quick testing (single experiments)
├── batch_runner.py                # Batch processing (multiple categories)
└── examples.py                    # Interactive examples
│
├── Output Directories:
├── results/                       # Experiment results (JSON)
└── experiments/                   # Custom experiments (optional)
```

## 🎯 Key Features

### 1. **Example Selection (8 Strategies)**
- Random, Balanced, Stratified
- Diverse, Representative
- Boundary, Hard, Easy

### 2. **Prompt Formats (5 Variations)**
- Standard (basic few-shot)
- Explicit Scale (emphasize criteria)
- Step-by-Step (chain-of-thought)
- Criteria Focused (evaluation-focused)
- Comparative (similarity-based)

### 3. **MSE-Focused Evaluation**
- Primary metric: Mean Squared Error
- Secondary: RMSE, MAE, Accuracy, QWK
- Error breakdown (perfect, off-by-1, off-by-2, etc.)
- Class-specific MSE
- Confusion matrix
- Worst predictions analysis

### 4. **Flexible Testing**
- Quick tests (20 samples, ~2 minutes)
- Full experiments (full test set, ~30-60 minutes)
- Batch processing (all categories)
- Customizable configurations

### 5. **Automated Workflows**
- Single experiments
- Strategy comparisons
- Format comparisons
- N-shots optimization
- Full experiment suites

## 🚀 Quick Start

### Test Single Configuration
```bash
cd tests
python quick_test.py --category 1_D_M --n-shots 3 --test-size 20
```

### Compare All Strategies
```bash
python quick_test.py --test-strategies --category 1_D_M --test-size 20
```

### Run Full Experiment Suite
```bash
python run_experiments.py
```

### Batch Process Categories
```bash
python batch_runner.py --categories 1_D_M 3_D_F 5_D_Ma --test-size 50
```

## 📊 Expected Output

### Console Output
- Experiment progress with timestamps
- Example selection statistics
- Classification progress
- **Detailed MSE evaluation:**
  - Primary metrics (MSE, RMSE, MAE, Accuracy)
  - Error statistics (mean, std, max over/under)
  - Error breakdown by magnitude
  - Class-specific MSE
  - Confusion matrix
- Comparison tables
- Best configuration recommendations

### Saved Results
```json
{
  "experiment_id": "20241214_143022",
  "category": "1_D_M",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "results": [
    {
      "experiment_name": "3shot_balanced_standard",
      "n_shots": 3,
      "selection_strategy": "balanced",
      "prompt_format": "standard",
      "metrics": {
        "mse": 0.3245,
        "accuracy": 0.72,
        "...": "..."
      }
    }
  ]
}
```

## 🎓 Typical Workflow

1. **Quick Iteration** (test_size=20)
   - Test strategies → Find best
   - Test formats → Find best
   - Test n_shots → Find optimal

2. **Validation** (test_size=50)
   - Validate best configs
   - Compare across variations
   - Analyze error patterns

3. **Final Evaluation** (full test set)
   - Confirm best configuration
   - Get production metrics
   - Document findings

4. **Scale Up** (all categories)
   - Apply best configs to all categories
   - Batch processing
   - Compare across categories

## 🔧 Configuration

Default settings in `config.py`:
```python
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_NUM_SHOTS = [1, 3, 5, 7]
DEFAULT_TEST_SIZE = 50
USE_8BIT = True
PRIMARY_METRIC = "mse"
```

## 📈 Success Metrics

**Good MSE values for ordinal ratings:**
- **< 0.30:** Excellent ⭐
- **0.30-0.50:** Good ✓
- **0.50-0.80:** Acceptable
- **> 0.80:** Needs improvement

## 💡 Design Decisions

1. **Focus on MSE:** Ordinal classification requires penalizing large errors
2. **Small models:** Qwen2.5 3B for balance of speed/quality
3. **Multiple strategies:** 8 strategies cover different selection approaches
4. **Multiple formats:** 5 formats test different prompting styles
5. **Systematic testing:** Automated comparison across all dimensions
6. **Detailed analysis:** Error breakdown helps identify issues

## 🔄 Integration with Main Project

The test suite integrates seamlessly:
- Uses existing `src/classification/classifier.py`
- Uses existing `src/classification/prompts.py` for category info
- Uses existing train/test splits from `data/processed/`
- Uses existing device detection from `src/utils/device.py`
- Saves results in organized JSON format
- Can be extended with custom strategies/formats

## 📚 Documentation

- **README.md:** Comprehensive guide (405 lines)
- **QUICKREF.md:** Quick reference for common tasks
- **Inline documentation:** All functions documented
- **Examples:** Interactive examples script

## 🎯 Next Steps

1. Run quick tests to familiarize yourself
2. Test on 1-2 categories with small test_size
3. Identify promising configurations
4. Scale up to larger test sets
5. Apply best configs to all categories
6. Document final recommendations

## 🎉 Ready to Use!

All components are implemented and tested. The suite is ready for:
- Finding optimal few-shot configurations
- Comparing different prompting strategies
- Evaluating model performance with MSE focus
- Batch processing multiple categories

**Start with:** `python tests/quick_test.py --category 1_D_M --test-size 20`

---

**Implementation Date:** December 14, 2025
**Focus:** Few-shot prompting with MSE optimization
**Model:** Qwen2.5 3B (small, efficient)
**Status:** ✅ Complete and ready for testing
