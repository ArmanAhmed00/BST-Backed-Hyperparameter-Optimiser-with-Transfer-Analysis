# BST-Backed Hyperparameter Optimiser with Transfer Analysis

A hyperparameter optimisation framework using Binary Search Trees (BST) for efficient trial management and transfer learning analysis between machine learning tasks.

## Project Team

- **Arman Ahmed**
- **Maroune Benhimoud**
- **Prodipta Paul**

> ⚠️ **Branch Policy**: Do not push to `main` branch. Use the `review` branch for all contributions.

---

## 📁 Project Structure

```
BST-Backed-Hyperparameter-Optimiser-with-Transfer-Analysis/
├── main.py                          # Entry point
├── pyproject.toml                   # Project configuration
├── requirements.txt                 # Dependencies
├── capstone_project/
│   ├── bst_toolkit/                 # BST implementation
│   │   ├── bst.py                   # Binary Search Tree class
│   │   ├── node.py                  # TrialNode dataclass
│   │   ├── registry.py              # Trial registry (TODO)
│   │   ├── rebuild.py               # Tree rebuild utilities (TODO)
│   │   └── __init__.py
│   ├── ml_toolkit/                  # ML utilities
│   │   ├── grid_search.py           # Grid search implementation (TODO)
│   │   ├── transfer.py              # Transfer learning analysis (TODO)
│   │   └── __init__.py
│   ├── data/                        # Data handling
│   │   ├── data.py                  # Data download & loading
│   │   └── data/                    # Downloaded datasets
│   │       ├── wdbc.data            # Breast Cancer dataset
│   │       └── data_banknote_authentication.txt
│   ├── benchmarks/                  # Performance benchmarking
│   │   ├── timer.py                 # Timing utilities
│   │   └── __init__.py
│   └── setup.py                     # Package setup
└── .venv/                           # Virtual environment
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project root
cd BST-Backed-Hyperparameter-Optimiser-with-Transfer-Analysis

# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

### 2. Download Datasets

```bash
cd capstone_project/data
uv run data.py
```

This downloads:
- **WDBC**: Wisconsin Diagnostic Breast Cancer (569 samples, 32 features)
- **Banknote Authentication**: Banknote dataset (1372 samples, 5 features)

### 3. Run the BST Module

```bash
# From project root
uv run python -m capstone_project.bst_toolkit.bst
```

---

## 🏗️ Architecture

### BST Toolkit (`bst_toolkit/`)

| File | Description |
|------|-------------|
| `bst.py` | Core BST implementation with insert, delete, search, and traversal operations |
| `node.py` | `TrialNode` dataclass representing a single hyperparameter trial |
| `registry.py` | Trial registry for tracking all experiments (TODO) |
| `rebuild.py` | Utilities for rebuilding BST from saved state (TODO) |

### ML Toolkit (`ml_toolkit/`)

| File | Description |
|------|-------------|
| `grid_search.py` | Grid search hyperparameter tuning (TODO) |
| `transfer.py` | Transfer learning analysis between tasks (TODO) |

### Data Module (`data/`)

- Downloads datasets from UCI Machine Learning Repository
- Handles automatic extraction and caching
- Provides column names for proper data loading

---

## 🔧 Usage

### Importing the BST

```python
from capstone_project.bst_toolkit.bst import BST
from capstone_project.bst_toolkit.node import TrialNode

# Create BST
bst = BST()

# Insert trials (score = evaluation metric, params = hyperparameters)
bst.insert(0.85, {"lr": 0.01, "epochs": 100})
bst.insert(0.92, {"lr": 0.001, "epochs": 50})
bst.insert(0.78, {"lr": 0.1, "epochs": 200})

# Query trials
best = bst.find_max()      # Highest score
worst = bst.find_min()     # Lowest score

# Traverse (sorted by score)
sorted_trials = bst.inorder()

# Check tree properties
print(f"Tree height: {bst.height()}")
print(f"Is balanced: {bst.is_balanced()}")
print(f"Total trials: {len(bst)}")
```

---

## 📊 Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        HYPERPARAMETER OPTIMISATION              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Define Search Space                                         │
│     - Learning rate: [0.001, 0.01, 0.1]                         │
│     - Batch size: [16, 32, 64]                                  │
│     - Hidden units: [64, 128, 256]                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Run Trials & Collect Results                                │
│     - Train model with hyperparameters                          │
│     - Evaluate (accuracy, F1, AUC, etc.)                        │
│     - Store score + params in BST                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. BST Operations                                              │
│     - Insert: Add new trial (O(log n))                         │
│     - Search: Find specific score                               │
│     - Find Min/Max: Get best/worst trial                        │
│     - Inorder: Sorted traversal (best → worst)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Transfer Learning Analysis (TODO)                           │
│     - Compare performance across datasets                       │
│     - Identify transferable hyperparameters                    │
│     - Build knowledge transfer pipeline                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Development Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies |
| `uv pip install -e .` | Install package in editable mode |
| `uv run python -m capstone_project.bst_toolkit.bst` | Run BST module |
| `cd capstone_project/data && uv run data.py` | Download datasets |
| `uv run main.py` | Run main entry point |

---

## 📋 TODO List

- [ ] Implement `registry.py` - Trial registry for tracking experiments
- [ ] Implement `rebuild.py` - Rebuild BST from saved state
- [ ] Implement `grid_search.py` - Grid search hyperparameter tuning
- [ ] Implement `transfer.py` - Transfer learning analysis
- [ ] Add unit tests
- [ ] Add CI/CD pipeline
- [ ] Document API

---

## 📚 Dependencies

- **Python**: 3.12+
- **pandas**: Data handling
- **requests**: HTTP requests
- **numpy**: Numerical computing
- **torch**: Machine learning (optional)

---

## 📄 License

This project is for academic purposes as part of a capstone project.

---

## 🔗 References

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [WDBC Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
