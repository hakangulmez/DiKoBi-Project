# Quick Reference Guide - Few-Shot Testing

## 🚀 Quick Commands

### Basic Testing
```bash
# Quick test (20 samples, ~2 min)
cd tests
python quick_test.py --category 1_D_M --n-shots 3 --test-size 20

# Test all strategies
python quick_test.py --test-strategies --category 1_D_M --test-size 20

# Test all formats
python quick_test.py --test-formats --category 1_D_M --test-size 20
```

### Full Experiments
```bash
# Full experiment suite (~30-60 min)
python run_experiments.py

# Batch process multiple categories
python batch_runner.py --test-size 50

# All categories (several hours)
python batch_runner.py --all-categories --test-size 50
```

### Interactive Examples
```bash
# Run guided examples
python examples.py
```

## 📊 Key Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration settings |
| `run_experiments.py` | Main experiment runner |
| `quick_test.py` | Quick testing script |
| `batch_runner.py` | Batch processing |
| `examples.py` | Interactive examples |
| `few_shot_selector.py` | Example selection |
| `prompts/few_shot_templates.py` | Prompt formats |
| `mse_evaluator.py` | Evaluation metrics |

## 🎯 Selection Strategies (8)

1. **random** - Random selection
2. **balanced** - Equal per class ⭐ (usually best)
3. **stratified** - Match distribution
4. **diverse** - Maximize variety
5. **representative** - Most typical
6. **boundary** - Ambiguous cases
7. **hard** - Difficult examples
8. **easy** - Clear examples

## 📝 Prompt Formats (5)

1. **standard** - Basic few-shot ⭐ (good default)
2. **explicit_scale** - Emphasize criteria
3. **step_by_step** - Chain-of-thought
4. **criteria_focused** - Highlight evaluation
5. **comparative** - Compare to examples

## 📈 Metrics

**Primary:** MSE (Mean Squared Error)
- Lower is better
- < 0.30 = Excellent
- 0.30-0.50 = Good
- 0.50-0.80 = Acceptable

**Secondary:**
- RMSE, MAE, Accuracy, QWK
- Error breakdown
- Class-specific MSE
- Confusion matrix

## 🔧 Configuration Tips

### For Speed
```python
# config.py
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TEST_SIZE = 20
```

### For Accuracy
```python
# config.py
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TEST_SIZE = None  # Full test set
```

### For Memory
```python
# config.py
USE_8BIT = True
BATCH_SIZE = 1
```



## 🎓 Typical Workflow

1. **Quick test** (20 samples) → Find promising configs
2. **Strategy comparison** (50 samples) → Select best strategy
3. **Format comparison** (50 samples) → Select best format
4. **Full evaluation** (full test set) → Validate
5. **Batch testing** → Apply to all categories

## 💡 Pro Tips

- Start with `balanced` strategy and `standard` format
- Test 3-5 shots first (often optimal)
- Use small test_size for iteration
- Check error breakdown for insights
- Save results regularly
- Compare across categories

## 🐛 Common Issues

**Out of memory?**
→ Enable 8-bit, reduce batch size, use smaller model

**Slow?**
→ Reduce test size, use 1.5B model

**Parse failures?**
→ Try different format, check examples

**Poor MSE?**
→ Try more shots, different strategy, better examples

## 📞 Quick Help

```python
# In Python
from tests.config import *
help(FewShotExperimentRunner)

# Check available strategies
from tests.few_shot_selector import FewShotSelector
# See SELECTION_STRATEGIES in config.py

# Check available formats
from tests.prompts.few_shot_templates import FewShotPromptGenerator
# See PROMPT_FORMATS in config.py
```

## 🎯 Goal

Find configuration that **minimizes MSE** for each category:
- Optimal n_shots (1, 3, 5, 7, 10)
- Best selection strategy
- Best prompt format

Then use in production! 🚀
