# CLAUDEME Guardrails

## File Size Limits

| Limit Type | Lines | Action |
|------------|-------|--------|
| HARD LIMIT | 500 | MUST split file |
| SOFT TARGET | 300 | SHOULD split file |
| IDEAL | 200 | Target for new files |

## Function and Class Limits

| Element | Max Lines |
|---------|-----------|
| Function | 50 |
| Class | 200 |
| Module | 500 |

## Module Structure Requirements

Every Python module MUST have:

1. **Module docstring** - Describes purpose and contents
2. **RECEIPT_SCHEMA export** - Documents receipt types
3. **__all__ list** - Explicit public API
4. **Clean imports** - At top of file, grouped

### RECEIPT_SCHEMA Template

```python
RECEIPT_SCHEMA = {
    "module": "src.module.name",
    "receipt_types": ["receipt_type_1", "receipt_type_2"],
    "version": "1.0.0",
}
```

## __init__.py Requirements

Every package __init__.py MUST:

1. Have module docstring
2. Import and re-export public API
3. NOT contain implementation code

### __init__.py Template

```python
"""Package description."""

from .module1 import function1
from .module2 import Class1

__all__ = ["function1", "Class1"]
```

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `fractal_layers.py` |
| Functions | snake_case | `compute_alpha()` |
| Classes | PascalCase | `StopRule` |
| Constants | UPPER_SNAKE | `TENANT_ID` |
| Private | _prefix | `_internal_func()` |

## Extraction Triggers

Split a file when:

1. Lines > 300 (soft trigger)
2. Lines > 500 (hard trigger)
3. Multiple unrelated concerns
4. > 10 public functions
5. > 5 classes

## Pre-Commit Checklist

- [ ] No file > 500 lines
- [ ] All imports resolve
- [ ] All tests pass
- [ ] Lint clean (ruff)
- [ ] Every directory has __init__.py
- [ ] Every module has RECEIPT_SCHEMA

## Directory Structure

```
src/
├── core/                 # Receipts, validation, constants
├── fractal/
│   └── depths/           # D1-D14 separated
├── cfd/                  # Reynolds regimes separated
├── paths/
│   ├── agi/
│   │   ├── defenses/     # One file per defense
│   │   └── zk/           # One file per ZK system
│   ├── multiplanet/
│   │   ├── moons/        # One file per moon
│   │   ├── planets/      # One file per planet
│   │   └── hubs/         # One file per hub
│   └── mars/             # ISRU, MOXIE, etc.
├── entropy/              # Entropy computations
└── utils/                # Common utilities
```

## Running Guardrails

```bash
# Check file sizes
./scripts/check_file_size.sh

# Full audit
./scripts/audit_codebase.sh

# Import verification
python -c "from src import *"
python -c "from src.fractal import recursive_fractal"
python -c "from src.cfd import stokes_settling"
```
