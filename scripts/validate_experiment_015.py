"""
Validation script for experiment 015 setup.

Checks:
1. All required imports are available
2. Config files are valid
3. Code supports all parameters
4. No obvious runtime issues
"""

import sys
import yaml
import traceback
from pathlib import Path

print("="*70)
print("VALIDATION SCRIPT FOR EXPERIMENT 015")
print("="*70)

errors = []
warnings = []

# Test 1: Check required imports
print("\n[TEST 1] Checking required imports...")
required_modules = [
    ('torch', 'PyTorch'),
    ('torch_geometric', 'PyTorch Geometric'),
    ('yaml', 'PyYAML'),
    ('numpy', 'NumPy'),
]

for module, name in required_modules:
    try:
        __import__(module)
        print(f"  ✓ {name} available")
    except ImportError:
        errors.append(f"Missing required module: {name} ({module})")
        print(f"  ✗ {name} NOT available")

# Test 2: Check torch-scatter (critical for GATv2)
print("\n[TEST 2] Checking torch-scatter (required for GATv2)...")
try:
    import torch_scatter
    print(f"  ✓ torch-scatter available (version: {torch_scatter.__version__})")
except ImportError:
    warnings.append("torch-scatter not installed - will be auto-installed by run_experiment_015.sh")
    print("  ⚠ torch-scatter not installed (will be auto-installed)")

# Test 3: Validate config files
print("\n[TEST 3] Validating config files...")
config_files = [
    'configs/experiment_015_gatv2_rewired.yaml',
    'configs/rewiring_015.yaml'
]

for config_path in config_files:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ {config_path} - Valid YAML")

        # Check specific requirements
        if 'experiment_015' in config_path:
            # Main experiment config
            if config.get('model', {}).get('structural_stream', {}).get('layer_type') != 'gatv2':
                warnings.append(f"{config_path}: layer_type should be 'gatv2'")
                print(f"    ⚠ Warning: layer_type is not 'gatv2'")

            if not config.get('phylogenetic', {}).get('use_rewired_graph', False):
                warnings.append(f"{config_path}: use_rewired_graph is False")
                print(f"    ⚠ Warning: use_rewired_graph is False")

    except FileNotFoundError:
        errors.append(f"Config file not found: {config_path}")
        print(f"  ✗ {config_path} - NOT FOUND")
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in {config_path}: {e}")
        print(f"  ✗ {config_path} - INVALID YAML")

# Test 4: Check if new implementations are importable
print("\n[TEST 4] Checking new implementations...")
implementations = [
    ('src.layers.denoising_gate', 'DenoisingGate'),
    ('src.layers.gatv2', 'GATv2Conv'),
    ('src.phylogenetic.link_prediction', 'LinkPredictor'),
    ('src.phylogenetic.train_link_predictor', 'LinkPredictionTrainer'),
    ('src.phylogenetic.graph_rewiring', 'rewire_graph_with_link_predictor'),
]

for module_path, obj_name in implementations:
    try:
        module = __import__(module_path, fromlist=[obj_name])
        obj = getattr(module, obj_name)
        print(f"  ✓ {module_path}.{obj_name} available")
    except ImportError as e:
        errors.append(f"Cannot import {module_path}.{obj_name}: {e}")
        print(f"  ✗ {module_path}.{obj_name} - IMPORT ERROR")
        traceback.print_exc()
    except AttributeError as e:
        errors.append(f"Module {module_path} has no {obj_name}: {e}")
        print(f"  ✗ {module_path}.{obj_name} - NOT FOUND")

# Test 5: Check DualStreamPhylogeneticTransformer with new params
print("\n[TEST 5] Checking DualStreamPhylogeneticTransformer compatibility...")
try:
    from src.models.dual_stream import DualStreamPhylogeneticTransformer

    # Create minimal config
    test_config = {
        'embedding': {'embedding_dim': 1024},
        'model': {
            'semantic_stream': {
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.4,
                'activation': 'gelu'
            },
            'structural_stream': {
                'hidden_dim': 256,
                'num_gnn_layers': 2,
                'dropout': 0.4,
                'layer_type': 'gatv2',
                'num_heads': 4,
                'use_residual': False,
                'use_denoising_gate': False
            },
            'cross_attention': {
                'hidden_dim': 256,
                'num_heads': 4,
                'num_layers': 1,
                'dropout': 0.4
            },
            'classifier': {
                'hidden_dim': 128,
                'num_classes': 2,
                'dropout': 0.5,
                'classifier_type': 'simple'
            }
        }
    }

    model = DualStreamPhylogeneticTransformer(test_config)
    print(f"  ✓ Model instantiation successful")
    print(f"  ✓ StructuralStream layer_type: {model.structural_stream.layer_type}")
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    errors.append(f"Model instantiation failed: {e}")
    print(f"  ✗ Model instantiation FAILED")
    traceback.print_exc()

# Test 6: Check run_graph_rewiring.py
print("\n[TEST 6] Checking run_graph_rewiring.py...")
rewiring_script = Path('run_graph_rewiring.py')
if rewiring_script.exists():
    print(f"  ✓ run_graph_rewiring.py exists")

    # Check if it's executable
    import stat
    if rewiring_script.stat().st_mode & stat.S_IXUSR:
        print(f"  ✓ run_graph_rewiring.py is executable")
    else:
        warnings.append("run_graph_rewiring.py is not executable")
        print(f"  ⚠ run_graph_rewiring.py is not executable (will still work with python)")
else:
    errors.append("run_graph_rewiring.py not found")
    print(f"  ✗ run_graph_rewiring.py NOT FOUND")

# Test 7: Check run_experiment_015.sh
print("\n[TEST 7] Checking run_experiment_015.sh...")
script_path = Path('run_experiment_015.sh')
if script_path.exists():
    print(f"  ✓ run_experiment_015.sh exists")

    # Check if it's executable
    if script_path.stat().st_mode & stat.S_IXUSR:
        print(f"  ✓ run_experiment_015.sh is executable")
    else:
        warnings.append("run_experiment_015.sh is not executable")
        print(f"  ⚠ run_experiment_015.sh needs chmod +x")

    # Read and validate structure
    content = script_path.read_text()
    required_steps = [
        'torch-scatter',
        'graph_data.pt',
        'run_graph_rewiring.py',
        'rewired_graph.pt',
    ]

    for step in required_steps:
        if step in content:
            print(f"  ✓ Script includes {step}")
        else:
            warnings.append(f"Script may be missing step: {step}")
            print(f"  ⚠ Script may be missing: {step}")

else:
    errors.append("run_experiment_015.sh not found")
    print(f"  ✗ run_experiment_015.sh NOT FOUND")

# Test 8: Check datasets
print("\n[TEST 8] Checking datasets...")
datasets = ['datasets/train.csv', 'datasets/test.csv']
for dataset in datasets:
    if Path(dataset).exists():
        print(f"  ✓ {dataset} exists")
    else:
        warnings.append(f"Dataset not found: {dataset}")
        print(f"  ⚠ {dataset} NOT FOUND (required for execution)")

# Test 9: Syntax check on critical files
print("\n[TEST 9] Python syntax validation...")
python_files = [
    'src/layers/denoising_gate.py',
    'src/layers/gatv2.py',
    'src/phylogenetic/link_prediction.py',
    'src/phylogenetic/train_link_predictor.py',
    'src/phylogenetic/graph_rewiring.py',
    'src/models/dual_stream.py',
    'run_graph_rewiring.py'
]

import py_compile

for py_file in python_files:
    try:
        py_compile.compile(py_file, doraise=True)
        print(f"  ✓ {py_file} - Syntax OK")
    except py_compile.PyCompileError as e:
        errors.append(f"Syntax error in {py_file}: {e}")
        print(f"  ✗ {py_file} - SYNTAX ERROR")

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

if errors:
    print(f"\n❌ ERRORS ({len(errors)}):")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
else:
    print("\n✅ No errors found!")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
else:
    print("\n✅ No warnings!")

print("\n" + "="*70)
if errors:
    print("❌ VALIDATION FAILED - Fix errors before running experiment")
    sys.exit(1)
elif warnings:
    print("⚠️  VALIDATION PASSED WITH WARNINGS - Review warnings before running")
    sys.exit(0)
else:
    print("✅ VALIDATION PASSED - Ready to run experiment!")
    sys.exit(0)
