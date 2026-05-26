# Repository Cleanup & Migration Guide

**Explanation of documentation reorganization.**

---

## What Changed

The repository documentation was reorganized from **9 scattered .md files** to a **clean, minimal structure**:

### Before (Cluttered)
```
├── README.md                    # Obsolete, high-level overview
├── QUICKSTART.md               # Duplicate of README
├── IMPLEMENTATION_SUMMARY.md   # Outdated
├── CHEATSHEET.md              # Commands in README
├── TROUBLESHOOTING.md         # Root level
├── FOLDFLOW_INTEGRATION.md    # Root level
├── LEAKAGE_AND_SPLITS_GUIDE.md # Newly added (long)
├── REFACTORING_SUMMARY.md     # Reference only
└── scripts/README.md          # Buried
```

**Problem**: Too many entry points, redundancy, confusing for new users.

### After (Clean)
```
├── README.md                   # ✅ NEW: Single entry point
├── config.yaml                # Configuration
├── Makefile                   # Commands
└── docs/
    ├── README.md             # ✅ NEW: Navigation guide
    ├── TROUBLESHOOTING.md    # ✅ Consolidated
    ├── LEAKAGE_AND_SPLITS.md # ✅ Moved (improved)
    ├── FOLDFLOW_INTEGRATION.md # ✅ Moved
    └── API_REFERENCE.md      # ✅ NEW: From scripts/README.md
```

**Benefit**: Clear hierarchy, single entry point, specific documentation for each use case.

---

## What Happened to Each File

### ✅ Kept & Updated

| File | Status | Changes |
|------|--------|---------|
| `README.md` | ✅ **UPDATED** | Complete rewrite - now main entry point |
| `config.yaml` | ✅ **KEPT** | No changes needed |
| `Makefile` | ✅ **KEPT** | No changes needed |

### 📦 Moved to docs/

| Old Location | New Location | Changes |
|---|---|---|
| `LEAKAGE_AND_SPLITS_GUIDE.md` | `docs/LEAKAGE_AND_SPLITS.md` | Renamed (shorter), same content |
| `TROUBLESHOOTING.md` | `docs/TROUBLESHOOTING.md` | Translated to English, expanded |
| `FOLDFLOW_INTEGRATION.md` | `docs/FOLDFLOW_INTEGRATION.md` | Translated to English, improved |
| `scripts/README.md` | `docs/API_REFERENCE.md` | Renamed, expanded with examples |

### ⚠️ Deprecated (kept for reference)

| File | Status | Why |
|---|---|---|
| `QUICKSTART.md` | ⚠️ **DEPRECATED** | Content now in main `README.md` |
| `IMPLEMENTATION_SUMMARY.md` | ⚠️ **DEPRECATED** | Outdated (pre-refactoring) |
| `CHEATSHEET.md` | ⚠️ **DEPRECATED** | Commands now in `README.md` |
| `REFACTORING_SUMMARY.md` | ⚠️ **DEPRECATED** | Reference only, see `docs/LEAKAGE_AND_SPLITS.md` |

**Note**: These files are still in the repo but no longer updated. They may be deleted in future cleanup.

---

## Migration Path for Users

### If you had bookmarks to...

**→ QUICKSTART.md**
```
NOW: Use README.md (Quick Start section)
```

**→ CHEATSHEET.md**
```
NOW: Use README.md (Step-by-Step section or see Makefile)
```

**→ IMPLEMENTATION_SUMMARY.md**
```
NOW: Use README.md (expected outputs section)
```

**→ TROUBLESHOOTING.md**
```
NOW: Use docs/TROUBLESHOOTING.md (same location if you had it, or moved to docs/)
```

**→ LEAKAGE_AND_SPLITS_GUIDE.md**
```
NOW: Use docs/LEAKAGE_AND_SPLITS.md
```

**→ FOLDFLOW_INTEGRATION.md**
```
NOW: Use docs/FOLDFLOW_INTEGRATION.md
```

---

## New Documentation Structure

### Entry Point: README.md

**What it contains:**
- 2-3 minute project overview
- Step-by-step execution (with commands)
- Directory structure
- Configuration guide
- Dataset properties
- Troubleshooting link
- Next steps

**Who should read**: Everyone (first)

### Navigation: docs/README.md

**What it contains:**
- Index of all documentation
- Decision tree (what to read based on your goal)
- Common learning paths
- Document purposes

**Who should read**: Anyone looking for specific information

### Technical Docs: docs/*.md

**LEAKAGE_AND_SPLITS.md**: Methodology & ML concepts  
**TROUBLESHOOTING.md**: Error solutions  
**FOLDFLOW_INTEGRATION.md**: Using with FoldFlow  
**API_REFERENCE.md**: Script details & code examples  

**Who should read**: Depends on need

---

## How to Find Information

### As a New User
1. Start with `README.md` (top level)
2. Follow step-by-step instructions
3. Run `make pipeline`

### As an Experienced User
1. Go to `docs/README.md`
2. Use decision tree to find relevant doc
3. Jump to specific section

### If Something Breaks
1. Go to `docs/TROUBLESHOOTING.md`
2. Find your error
3. Follow solution

### If You Want Deep Dive
1. Go to `docs/README.md`
2. Choose topic from index
3. Read relevant document

---

## What This Means

### ✅ Better For Users
- Single clear entry point (README.md)
- Non-cluttered root directory
- Specific documentation for specific needs
- Clear navigation

### ✅ Better For Maintainers
- Each file has single purpose
- No duplication
- Easier to update
- Clear organization

### ✅ Better For Repo Appearance
- Professional structure
- Clean root directory
- Organized `/docs/` folder
- Standard conventions

---

## Future Cleanup

### Will Eventually Be Deleted
```
QUICKSTART.md                 # Content moved to README.md
IMPLEMENTATION_SUMMARY.md     # Outdated
CHEATSHEET.md                # Commands in README.md
REFACTORING_SUMMARY.md       # Reference info in docs/LEAKAGE_AND_SPLITS.md
```

**When**: At next major version or when confirmed no one uses them

### Will Stay
```
README.md                     # Main entry point
config.yaml                   # Configuration
Makefile                      # Build system
docs/                         # Documentation folder
  ├── README.md              # Navigation
  ├── TROUBLESHOOTING.md     # Errors
  ├── LEAKAGE_AND_SPLITS.md  # Methodology
  ├── FOLDFLOW_INTEGRATION.md # Integration
  └── API_REFERENCE.md       # Technical details
```

---

## Quick Reference

### Where to Find Commands
- **Install**: README.md → Quick Start → Installation
- **Download**: README.md → Quick Start → Step 2
- **Run everything**: `make pipeline`

### Where to Find Explanations
- **Why accuracy dropped**: docs/LEAKAGE_AND_SPLITS.md
- **How to use with FoldFlow**: docs/FOLDFLOW_INTEGRATION.md
- **How scripts work**: docs/API_REFERENCE.md

### Where to Find Help
- **Something errored**: docs/TROUBLESHOOTING.md
- **Not sure what to read**: docs/README.md

---

## Backward Compatibility

### Broken Links?
If old docs had links like:
```
See QUICKSTART.md for details
```

**New equivalent:**
```
See README.md (Quick Start section)
```

### Old Commands Still Work?
Yes, all Makefile targets unchanged:
```bash
make pipeline    # Still works
make download-dataset  # Still works
```

### Old Config Still Works?
Yes, config.yaml unchanged.

---

## FAQ

**Q: Why move things to docs/?**  
A: Keeps root clean, clear hierarchy, follows GitHub conventions.

**Q: Will old files be deleted?**  
A: Eventually yes, but staying for now to avoid breaking old bookmarks.

**Q: Where do I start?**  
A: Read `/README.md` (you are probably there already).

**Q: The docs are too long**  
A: Read only what you need! Use `docs/README.md` to find specific doc.

**Q: I liked the CHEATSHEET**  
A: Commands are now in README.md and also in Makefile.

---

## Summary

**Old**: 9 scattered .md files, unclear navigation  
**New**: 1 main README.md + 5 focused docs/ files

**Result**: Easier to understand, clearer navigation, professional structure.

**For you**: Just start with `README.md` and follow the steps. Everything else is optional reading.

---

**Last updated**: 2026-05-25
**Cleanup completed**: 2026-05-25
