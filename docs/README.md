# Documentation Index

**Quick guide to find what you need.**

---

## I want to...

### 🚀 **Get started quickly**
→ Go to root [README.md](../README.md)
- 2-3 minute overview
- Step-by-step commands
- Expected outputs

### 🐛 **Fix an error**
→ See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Common problems
- Solutions
- Debugging tips

### 🔒 **Understand data leakage & grouped splits**
→ See [LEAKAGE_AND_SPLITS.md](LEAKAGE_AND_SPLITS.md)
- Why accuracy dropped after refactoring
- Data leakage explanation
- How grouped splits work
- When to use each strategy

### 🧬 **Use with FoldFlow**
→ See [FOLDFLOW_INTEGRATION.md](FOLDFLOW_INTEGRATION.md)
- Dataset preparation for FoldFlow
- Conditioning module code
- Training loop example
- Inference examples

### 📊 **Understand the API & scripts**
→ See [API_REFERENCE.md](API_REFERENCE.md)
- Detailed documentation of each script
- Class methods and parameters
- Data formats
- Code examples

---

## Document Purposes

| Document | Purpose | Read if... |
|----------|---------|-----------|
| `../README.md` | Main entry point | You're new to the project |
| `TROUBLESHOOTING.md` | Error solutions | Something broke |
| `LEAKAGE_AND_SPLITS.md` | Technical ML details | You want to understand the methodology |
| `FOLDFLOW_INTEGRATION.md` | Integration guide | You're using FoldFlow for generation |
| `API_REFERENCE.md` | Technical API docs | You're modifying scripts or need details |

---

## Quick Decision Tree

```
START HERE
    ↓
Is this my first time?
├─ YES → Read ../README.md (2 min)
│       Follow step-by-step instructions
│       Run: make pipeline
│       Done!
│
└─ NO, I have a problem
    ├─ Something errored?
    │  └─ See TROUBLESHOOTING.md
    │
    ├─ Want to understand the methodology?
    │  └─ See LEAKAGE_AND_SPLITS.md
    │
    ├─ Using FoldFlow?
    │  └─ See FOLDFLOW_INTEGRATION.md
    │
    └─ Need technical details?
       └─ See API_REFERENCE.md
```

---

## Common Paths

### Path 1: I just want to run the pipeline

1. Read: `../README.md` (Quick Start section)
2. Run: `make pipeline`
3. Done!

**Time**: 5 minutes reading, 1-2 hours running

### Path 2: I'm getting an error

1. Read: `TROUBLESHOOTING.md` (find your error)
2. Follow solution
3. Run again
4. Still not working? Check the validation steps

**Time**: 10-15 minutes

### Path 3: I want to understand the data

1. Read: `../README.md` (Dataset section)
2. Read: `LEAKAGE_AND_SPLITS.md` (full explanation)
3. Run: `make explore-dataset`
4. Explore: `jupyter notebook notebooks/dataset_visualization.ipynb`

**Time**: 30 minutes reading, 5 minutes running

### Path 4: I'm using FoldFlow

1. Read: `../README.md` (overview)
2. Run: `make pipeline`
3. Read: `FOLDFLOW_INTEGRATION.md` (integration guide)
4. Adapt code examples to your needs

**Time**: 30 minutes reading, 1-2 hours running

### Path 5: I'm modifying the scripts

1. Read: `../README.md` (get context)
2. Read: `API_REFERENCE.md` (understand structure)
3. Look at the actual code
4. Modify carefully

**Time**: 1-2 hours

---

## Document Details

### ../README.md
**Length**: ~500 lines  
**Audience**: Anyone (non-technical overview + technical details)  
**Contains**:
- Project overview
- Quick start (2-3 min)
- Step-by-step instructions
- Directory structure
- Configuration guide
- Key dataset properties
- Expected outputs
- Next steps

### TROUBLESHOOTING.md
**Length**: ~400 lines  
**Audience**: Users encountering issues  
**Contains**:
- Common errors and solutions
- Installation issues
- KLIFS API issues
- Dataset download issues
- Preprocessing issues
- Split issues
- Training issues
- Data validation
- Performance optimization
- Recovery procedures

### LEAKAGE_AND_SPLITS.md
**Length**: ~400 lines  
**Audience**: ML practitioners, researchers  
**Contains**:
- Data leakage definition
- Why it's dangerous
- Your pipeline's original problem
- Grouped splits solution
- Why better for proteins
- Limitations with ~13 kinases
- Expected accuracy changes
- Validation procedures
- Configuration
- Troubleshooting
- Concepts reference

### FOLDFLOW_INTEGRATION.md
**Length**: ~300 lines  
**Audience**: Developers using FoldFlow  
**Contains**:
- Overview/architecture
- What is conditioning
- Dataset preparation
- Custom Dataset class
- Conditioning module
- Training loop code
- Inference examples
- Interpolation examples
- Advanced techniques
- Integration checklist
- Performance tips

### API_REFERENCE.md
**Length**: ~400 lines  
**Audience**: Developers, researchers  
**Contains**:
- Scripts overview table
- Class/method documentation
- Configuration details
- Data format specifications
- Biological concepts
- Usage examples
- Code snippets
- Performance notes

---

## When to Update

**Update README.md** when:
- New steps added to pipeline
- User interface changes
- New features

**Update TROUBLESHOOTING.md** when:
- New common errors discovered
- Solutions change
- Workarounds found

**Update LEAKAGE_AND_SPLITS.md** when:
- Methodology changes
- New split strategies
- Important insights

**Update FOLDFLOW_INTEGRATION.md** when:
- FoldFlow API changes
- New conditioning methods
- Better examples

**Update API_REFERENCE.md** when:
- Scripts modified
- New parameters
- Data formats change

---

## Structure Rationale

**Why separate docs/?**
- Root README is for "how to run"
- docs/ is for "how it works" (deeper dives)
- Keeps main entry point clean and minimal

**Why 5 documents, not 1 giant one?**
- Users only read what they need (faster)
- Easier to maintain and update
- Each has a specific audience
- Better for cross-referencing

**Why not wiki or external docs?**
- Everything stays in repo
- No external dependencies
- Works offline
- Version controlled

---

## Navigation Tips

- Use Ctrl+F (or Cmd+F) to search within documents
- Most documents have a table of contents at top
- Links between documents use [filename](path.md) format
- Search GitHub repo for keywords

---

## Feedback

If documentation is unclear:
1. Open the document
2. Note which section confused you
3. Suggest improvement (if you can)
4. Report issue/confusion

---

**Last updated**: 2026-05-25
