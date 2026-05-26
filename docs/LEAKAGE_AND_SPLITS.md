# Understanding Data Leakage & Grouped Splits

**Why this pipeline uses grouped splits instead of random splits, and why it matters for protein structures.**

---

## Quick Answer

**TL;DR**: 
- ❌ Random splits can mix structures from the same kinase between train/test
- ✅ Grouped splits keep all structures of one kinase together
- 📊 Result: Honest metrics that reflect real generalization to NEW kinases

---

## What is Data Leakage?

### Simple Definition

**Data leakage** = Information from the test set "leaks" into training, artificially inflating performance metrics.

### School Exam Analogy

Imagine two scenarios:

**❌ WITH LEAKAGE:**
- Teacher teaches you general concepts in class
- Teacher gives you THE EXACT EXAM to practice at home
- You get 95% on the exam
- **Problem**: You didn't learn concepts, you memorized answers

**✅ WITHOUT LEAKAGE:**
- Teacher teaches you general concepts
- You take a NEW exam with different but related problems
- You get 70% on the exam
- **Good**: This reflects real learning

---

## Data Leakage in Your Original Pipeline

### The Problem

Random stratified split:
```python
train, test = train_test_split(df, stratify=df['conformational_state'])
```

**What happened:**
```
Dataset:
├─ EGFR structure A → TRAIN (randomly)
├─ EGFR structure B → TEST  (randomly)  ← LEAKAGE!
├─ BRAF structure X → TRAIN
├─ BRAF structure Y → TEST  ← LEAKAGE!
└─ ...
```

**Why is this bad?**

1. Model sees EGFR in training
2. Model also sees EGFR in test (different structure, same protein)
3. Model learned "how to recognize EGFR" in train
4. Model recognizes EGFR in test → accuracy artificially high
5. When you deploy on a NEW kinase (never seen), model fails

### Consequence: Inflated Metrics

```
❌ WITH LEAKAGE:
  Train Accuracy: 92%
  Test Accuracy:  87%  ← Misleading! You've seen these kinases
  
  Interpretation: "Great model!"
  Reality: Model just memorized EGFR, BRAF patterns
  
✅ WITHOUT LEAKAGE:
  Train Accuracy: 92%
  Test Accuracy:  63%  ← Lower but HONEST
  
  Interpretation: "Model generalizes 63% to new kinases"
  Reality: This is what you'll get on truly new proteins
```

---

## Grouped Splits: The Solution

### How It Works

```python
# ✅ Group all structures of SAME kinase in ONE split

Distribution:
├── TRAIN (70%)
│   ├─ ALL EGFR structures      } Complete per kinase
│   ├─ ALL BRAF structures      } No overlap with other splits
│   ├─ ALL ABL1 structures
│   └─ ... (9 kinases total)
│
├── VAL (15%)
│   ├─ ALL CDK4 structures      } Different kinases
│   ├─ ALL KIT structures
│   └─ ... (2 kinases total)
│
└── TEST (15%)
    ├─ ALL PDGFRA structures    } Completely new kinases
    ├─ ALL FGFR1 structures
    └─ ... (2 kinases total)
```

### Guarantees

✅ **Zero kinase overlap** - Each kinase appears in exactly ONE split  
✅ **Simulates real scenario** - Test kinases are completely unseen  
✅ **Honest metrics** - Performance reflects true generalization  
✅ **Reproducible** - `random_state=42` gives same split every time  

---

## Why This Matters for Protein Structures

### Protein Uniqueness

Each kinase has:
- Unique 3D structure
- Unique amino acid sequence
- Unique folding patterns
- Unique chemical environment

**Problem**: With random splits, model easily confuses:
- "Feature of EGFR protein" ← memorization
- vs "Feature of active/inactive state" ← real learning

### Real Use Case

In production:
- You study: EGFR, BRAF, ABL1 (training data)
- You encounter: NOVO_KINASE (never seen before)
- Your model must predict: Is NOVO_KINASE active or inactive?

**Random split approach**: Model never learned general principles, only EGFR/BRAF patterns → fails on NOVO_KINASE

**Grouped split approach**: Model learned "these patterns = active in ANY kinase" → works on NOVO_KINASE

---

## The Trade-off: ~13 Kinases Limit

### The Challenge

You have 13 unique kinases. For grouped splits:
- 70% train → ~9 kinases
- 15% val → ~2 kinases  
- 15% test → ~2 kinases

### The Dilemma

```
Option A: Perfect ACTIVE/INACTIVE balance
├─ Each split has 50% active, 50% inactive
└─ But kinase assignment is arbitrary

Option B: Group separation (no leakage)
├─ Completely separate kinases between splits
└─ But balance might be 70% active, 30% inactive
```

### The Decision: Prioritize Leakage Prevention

```yaml
# In config.yaml:
splits:
  prioritize_group_separation: true  # Leakage prevention > perfect balance
```

**Meaning**: "Avoid leakage FIRST, perfect balance SECOND"

### Impact on Training

If splits are imbalanced (e.g., 70% active in train, 30% in test):

**Solution**: Use weighted loss in training
```python
pos_weight = n_inactive / n_active  # Weight minority class more
loss = BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## Expected Accuracy Drop

### Before vs After Refactoring

```
BEFORE (with leakage):          AFTER (without leakage):
├─ Train: 92%                   ├─ Train: 90%
├─ Val:   88%                   ├─ Val:   65%
└─ Test:  85%  ❌ FALSE         └─ Test:  62%  ✅ HONEST
                                    (new kinases in test)
```

### Interpretation

**❌ WRONG**: "My refactoring broke the model! Accuracy dropped 20%!"

**✅ CORRECT**: "I was measuring wrong before. The real accuracy on new kinases is 62%."

### Why Lower is Actually Better

```
Scenario A: Test Accuracy 85% (with leakage)
└─ You deploy on new protein → Fails with 40% accuracy
└─ Your metrics were lying

Scenario B: Test Accuracy 62% (without leakage)
└─ You deploy on new protein → Gets ~60% accuracy
└─ Your metrics predicted reality
```

---

## Validation: Zero Leakage Guarantee

### What Gets Checked

The pipeline validates:

```python
# ✅ Check 1: No kinase appears twice
assert train_kinases & val_kinases == ∅
assert train_kinases & test_kinases == ∅
assert val_kinases & test_kinases == ∅

# ✅ Check 2: Full coverage
assert total_structures_in_splits == total_structures_in_metadata

# ✅ Check 3: Reasonable proportions
assert train_proportion ≈ 70% (within reason)
assert val_proportion ≈ 15%
assert test_proportion ≈ 15%
```

### Logs Show Details

```
🧬 KINASE_NAMES PER SPLIT
  TRAIN (9): ABL1, AKT1, BRAF, CDK4, CDK6, EGFR, ERBB2, FGFR1, KIT
  VAL   (2): MET, PDGFRA
  TEST  (2): ALK, PIK3CA

✅ VALIDATION OF LEAKAGE
   ✓ No overlap between TRAIN, VAL, TEST
   ✓ Kinase separation guaranteed
   ✓ Coverage: 523 structures distributed
   ✓ Final proportions: Train 70.0%, Val 15.3%, Test 14.7%
```

---

## Configuration

In `config.yaml`:

```yaml
splits:
  # Strategy: "grouped" (recommended) or "stratified" (legacy, has leakage)
  strategy: "grouped"
  
  # Group by: which column defines groups (e.g., kinase_name)
  group_by: "kinase_name"
  
  # Proportions (approximate with ~13 groups)
  train: 0.70
  validation: 0.15
  test: 0.15
  
  # Reproducibility
  random_state: 42
  
  # Safety
  prevent_leakage: true  # Validation will fail if overlap detected
  verbose_reporting: true  # Show exact kinases per split
```

---

## Troubleshooting

### Q: "Balance is off (70% active in train, 30% in test)"

**A**: Normal with grouped splits. Solution:

```python
# In training, use weighted loss:
from torch.nn import BCEWithLogitsLoss

pos_weight = 180 / 343  # Adjust based on your actual distribution
loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

### Q: "Test accuracy is much lower than validation"

**A**: Expected! Test has new kinases. This is honest measurement.

```
Train (seen kinases): 90% → Overfits to EGFR/BRAF patterns
Val (new kinases):    65% → More realistic
Test (new kinases):   62% → Consistent with Val, shows true generalization
```

### Q: "Can I use strategy='stratified' instead?"

**A**: Yes, but...

```python
# config.yaml
strategy: "stratified"  # ⚠️ WARNING: Can cause leakage!
```

The pipeline will warn:
```
⚠️  USING STRATIFIED STRATEGY (LEGACY)
   This CAN CAUSE LEAKAGE STRUCTURAL
   Recommended: use strategy='grouped'
```

---

## Concepts Reference

| Term | Meaning |
|------|---------|
| **Data Leakage** | Test information "leaks" into training, inflating metrics |
| **Overfitting** | Model memorizes training data instead of learning patterns |
| **Distribution Shift** | Train and test come from different distributions |
| **Stratification** | Keep class balance (active/inactive) across splits |
| **Grouped Stratification** | Keep class balance AND ensure groups don't overlap |
| **Generalization** | Model's ability to perform on unseen data |

---

## Summary

| Aspect | Random Split | Grouped Split |
|--------|--------------|---------------|
| **Leakage** | ❌ Possible | ✅ Impossible |
| **Realism** | ❌ Unrealistic | ✅ Real use case |
| **Metrics** | ❌ Inflated | ✅ Honest |
| **Implementation** | ✅ Simple | ✅ Implemented |
| **Your Pipeline** | ❌ Old approach | ✅ Current approach |

**Result**: Your metrics might look lower, but they're **honest** and **actionable**.

---

## Further Reading

- See `README.md` for quick start
- See `TROUBLESHOOTING.md` for common errors
- See `API_REFERENCE.md` for technical details
