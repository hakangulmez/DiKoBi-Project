# Few-Shot Prompting Tests

**Goal:** Find the best few-shot prompting strategies for DiKobi classification using small models (Qwen2.5 3B) with MSE as the primary evaluation metric.

## 🎯 Overview

This test suite systematically evaluates:
1. **Number of examples** (1-shot, 3-shot, 5-shot, 7-shot, etc.)
2. **Example selection strategies** (8 strategies: random, balanced, stratified, diverse, etc.)
3. **Prompt formats** (5 formats: standard, explicit_scale, step_by_step, etc.)

The focus is on **minimizing MSE (Mean Squared Error)** to achieve the most accurate predictions for ordinal ratings.

## 📁 Structure

```
tests/
├── README.md                   # This file
├── config.py                   # Configuration settings
├── __init__.py
│
├── few_shot_selector.py        # Example selection strategies
├── mse_evaluator.py            # MSE-focused evaluation
│
├── prompts/
│   └── few_shot_templates.py  # Prompt format generators
│
├── experiments/                # Optional: custom experiments
│
├── results/                    # Experiment results (JSON)
│   └── few_shot_results_*.json
│
├── run_experiments.py          # Main experiment runner
├── quick_test.py               # Quick testing script
└── batch_runner.py             # Batch processing multiple categories
```

## 🚀 Quick Start

### 1. Run a Quick Test (20 samples, ~2 minutes)

```bash
cd tests
python quick_test.py --category 1_D_M --n-shots 3 --test-size 20
```

### 2. Test All Strategies (20 samples each, ~10 minutes)

```bash
python quick_test.py --test-strategies --category 1_D_M --test-size 20
```

### 3. Test All Formats (20 samples each, ~8 minutes)

```bash
python quick_test.py --test-formats --category 1_D_M --test-size 20
```

### 4. Run Full Experiment Suite (50 samples, ~30-60 minutes)

```bash
python run_experiments.py
```

This runs:
- **Phase 1:** N-shots comparison (1, 3, 5, 7 shots)
- **Phase 2:** Strategy comparison (5 strategies)
- **Phase 3:** Format comparison (5 formats)

### 5. Batch Process Multiple Categories

```bash
# Quick test categories only
python batch_runner.py --test-size 50

# Specific categories
python batch_runner.py --categories 1_D_M 3_D_F 5_D_Ma --test-size 50

# All 26 categories (several hours)
python batch_runner.py --all-categories --test-size 50
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model (small models recommended)
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_OPTIONS = [
    "Qwen/Qwen2.5-3B-Instruct",    # Best balance
    "Qwen/Qwen2.5-1.5B-Instruct",  # Faster
    "Qwen/Qwen2.5-0.5B-Instruct"   # Fastest
]

# Few-shot settings
DEFAULT_NUM_SHOTS = [1, 3, 5, 7]
MAX_SHOTS = 10

# Test size
DEFAULT_TEST_SIZE = 50  # None = full test set

# Memory optimization
USE_8BIT = True  # 8-bit quantization
BATCH_SIZE = 2
```

## 📊 Example Selection Strategies

The system tests **8 different strategies** for selecting few-shot examples:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `random` | Random selection | Baseline comparison |
| `balanced` | Equal per rating class | Balanced datasets |
| `stratified` | Match rating distribution | Imbalanced datasets |
| `diverse` | Maximize text variety | Coverage |
| `representative` | Most typical examples | Clear prototypes |
| `boundary` | Ambiguous cases | Decision boundaries |
| `hard` | Difficult examples | Challenging cases |
| `easy` | Clear examples | Simple cases |

## 📝 Prompt Formats

The system tests **5 different prompt formats**:

| Format | Description | Characteristics |
|--------|-------------|-----------------|
| `standard` | Basic few-shot | Simple, clear structure |
| `explicit_scale` | Emphasize rating scale | Clear criteria |
| `step_by_step` | Chain-of-thought | Reasoning process |
| `criteria_focused` | Highlight criteria | Evaluation focus |
| `comparative` | Compare to examples | Similarity-based |

## 📈 Evaluation Metrics

**Primary Metric:** MSE (Mean Squared Error)
- Lower is better
- Penalizes large errors heavily
- Range: 0 (perfect) to ∞

**Secondary Metrics:**
- **RMSE:** Root MSE (same scale as ratings)
- **MAE:** Mean Absolute Error (average distance)
- **Accuracy:** Exact match percentage
- **QWK:** Quadratic Weighted Kappa (inter-rater agreement)

**Detailed Analysis:**
- Error breakdown (perfect, off-by-1, off-by-2, etc.)
- Class-specific MSE
- Confusion matrix
- Worst predictions

## 💡 Usage Examples

### Example 1: Quick Single Test

```python
from tests.run_experiments import FewShotExperimentRunner

runner = FewShotExperimentRunner(
    category="1_D_M",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    test_size=20
)

result = runner.run_single_experiment(
    n_shots=3,
    selection_strategy="balanced",
    prompt_format="standard"
)

print(f"MSE: {result['metrics']['mse']:.4f}")
runner.cleanup()
```

### Example 2: Compare Strategies

```python
runner = FewShotExperimentRunner(
    category="1_D_M",
    test_size=50
)

runner.run_strategy_comparison(
    n_shots=3,
    strategies=["random", "balanced", "stratified"],
    prompt_format="standard"
)

runner.save_results()
runner.cleanup()
```

### Example 3: Find Optimal N-Shots

```python
runner = FewShotExperimentRunner(
    category="1_D_M",
    test_size=50
)

runner.run_n_shots_comparison(
    n_shots_list=[1, 2, 3, 5, 7, 10],
    selection_strategy="balanced",
    prompt_format="standard"
)

runner.save_results()
runner.cleanup()
```

## 📊 Results Format

Results are saved as JSON files in `results/`:

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
        "rmse": 0.5697,
        "mae": 0.4231,
        "accuracy": 0.72,
        "qwk": 0.6543
      },
      "time_seconds": 156.3,
      "timestamp": "2024-12-14T14:35:42"
    }
  ]
}
```

## 🎓 Best Practices

### For Quick Iteration

1. Start with `test_size=20` for rapid testing
2. Use `quick_test.py` for single experiments
3. Test strategies first, then formats
4. Increase test size once you find promising configs

### For Thorough Analysis

1. Use full test set (`test_size=None`)
2. Run complete experiment suite
3. Test all strategies and formats
4. Save results for later comparison

### For Production

1. Select best config from experiments
2. Validate on full test set
3. Test on multiple categories
4. Document final configuration

## 🔍 Interpreting Results

### Good MSE Values

For ordinal ratings (0-2 or 0-3):
- **MSE < 0.30:** Excellent
- **MSE 0.30-0.50:** Good
- **MSE 0.50-0.80:** Acceptable
- **MSE > 0.80:** Needs improvement

### Common Patterns

- **Balanced strategy** usually performs best
- **3-5 shots** often optimal (more isn't always better)
- **Standard format** works well for clear criteria
- **Explicit_scale format** helps with ambiguous cases

### Error Analysis

Check the error breakdown:
- High "off-by-1": May need better examples
- High "off-by-2+": May need different strategy or format
- Class-specific high MSE: Need more representative examples for that class

## 🐛 Troubleshooting

### Out of Memory

```python
# In config.py
USE_8BIT = True  # Enable 8-bit quantization
BATCH_SIZE = 1   # Reduce batch size
DEFAULT_TEST_SIZE = 20  # Use smaller test set
```

### Slow Performance

```python
# Use smaller model
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Reduce test size
DEFAULT_TEST_SIZE = 30

# Test fewer strategies
strategies = ["random", "balanced", "stratified"]
```

### Parse Failures

If many predictions fail to parse:
- Check prompt format (some formats work better)
- Verify examples are clean and clear
- Try different temperature settings in `config.py`

## 📚 Advanced Usage

### Custom Selection Strategy

```python
# In few_shot_selector.py, add new method:
def _select_custom(self, n_shots: int) -> List[Dict]:
    """Your custom selection logic."""
    # Your implementation
    pass

# Register in select_examples():
elif strategy == "custom":
    return self._select_custom(n_shots)
```

### Custom Prompt Format

```python
# In prompts/few_shot_templates.py, add new method:
def _custom_format(self, text: str, examples: List[Dict]) -> str:
    """Your custom prompt format."""
    # Your implementation
    pass

# Register in generate_prompt():
elif format_type == "custom":
    return self._custom_format(text, examples)
```

### Custom Experiment

```python
# Create custom experiment in experiments/ folder
from tests.run_experiments import FewShotExperimentRunner

runner = FewShotExperimentRunner(
    category="1_D_M",
    test_size=50
)

# Your custom experiment logic
# ...

runner.save_results()
runner.cleanup()
```

## 📈 Next Steps

After finding optimal configurations:

1. **Validate:** Test on additional categories
2. **Scale:** Run batch experiments on all categories
3. **Compare Models:** Test with different model sizes
4. **Fine-tune:** Consider fine-tuning on best configs
5. **Deploy:** Integrate best config into production pipeline

## 🤝 Contributing

To add new features:

1. **New Strategy:** Add to `few_shot_selector.py`
2. **New Format:** Add to `prompts/few_shot_templates.py`
3. **New Metric:** Add to `mse_evaluator.py`
4. **Update Config:** Add to `config.py`

## 📝 Notes

- All experiments use the same random seed for reproducibility
- Results are timestamped and saved automatically
- Models are automatically cleaned up after experiments
- 8-bit quantization is enabled by default for memory efficiency
- Test data is automatically loaded from `data/processed/test/`

## 🎯 Goals Summary

✅ **Primary Goal:** Minimize MSE for ordinal classification
✅ **Secondary Goals:** Maximize accuracy, minimize inference time
✅ **Focus:** Few-shot learning with small models
✅ **Output:** Best configuration (n_shots, strategy, format) per category

---

**Happy Testing! 🚀**
