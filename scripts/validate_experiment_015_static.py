"""
Static validation for experiment 015 (no imports needed).

Checks code logic, parameter compatibility, and configuration consistency.
"""

import re
import ast
import yaml
from pathlib import Path

print("="*70)
print("STATIC VALIDATION FOR EXPERIMENT 015")
print("="*70)

errors = []
warnings = []
info = []

# Load configs
print("\n[1] Loading configurations...")
with open('configs/experiment_015_gatv2_rewired.yaml') as f:
    exp_config = yaml.safe_load(f)

with open('configs/rewiring_015.yaml') as f:
    rewire_config = yaml.safe_load(f)

print("  ✓ Configs loaded")

# Check 1: Validate experiment config
print("\n[2] Validating experiment config...")

# Check layer_type
layer_type = exp_config.get('model', {}).get('structural_stream', {}).get('layer_type')
if layer_type != 'gatv2':
    errors.append(f"layer_type should be 'gatv2', got '{layer_type}'")
    print(f"  ✗ layer_type = '{layer_type}' (should be 'gatv2')")
else:
    print(f"  ✓ layer_type = 'gatv2'")

# Check num_heads
num_heads = exp_config.get('model', {}).get('structural_stream', {}).get('num_heads')
if not num_heads:
    warnings.append("num_heads not specified (will use default=4)")
    print(f"  ⚠ num_heads not specified")
else:
    print(f"  ✓ num_heads = {num_heads}")

# Check use_rewired_graph
use_rewired = exp_config.get('phylogenetic', {}).get('use_rewired_graph', False)
if not use_rewired:
    warnings.append("use_rewired_graph is False - rewiring will not be used")
    print(f"  ⚠ use_rewired_graph = False")
else:
    print(f"  ✓ use_rewired_graph = True")

# Check rewired_graph_path
rewired_path = exp_config.get('phylogenetic', {}).get('rewired_graph_path')
if not rewired_path:
    errors.append("rewired_graph_path not specified")
    print(f"  ✗ rewired_graph_path not specified")
else:
    print(f"  ✓ rewired_graph_path = {rewired_path}")

    # Check if it matches the expected output from rewiring
    expected_path = "results/experiment_015_gatv2_rewired/rewiring/rewired_graph.pt"
    if rewired_path != expected_path:
        warnings.append(f"rewired_graph_path mismatch: {rewired_path} vs {expected_path}")
        print(f"  ⚠ Path mismatch: expected {expected_path}")

# Check denoising gate (should be disabled for rewired graph)
use_denoising = exp_config.get('model', {}).get('structural_stream', {}).get('use_denoising_gate', False)
if use_denoising:
    warnings.append("use_denoising_gate is True - not needed with rewired graph")
    print(f"  ⚠ use_denoising_gate = True (not recommended with rewiring)")
else:
    print(f"  ✓ use_denoising_gate = False (good for rewired scenario)")

# Check 2: Validate rewiring config
print("\n[3] Validating rewiring config...")

# Check graph_data_path matches
graph_data_path = rewire_config.get('graph_data_path')
expected_graph_data = "results/experiment_015_gatv2_rewired/graph_data.pt"
if graph_data_path != expected_graph_data:
    warnings.append(f"graph_data_path mismatch: {graph_data_path} vs {expected_graph_data}")
    print(f"  ⚠ graph_data_path = {graph_data_path}")
else:
    print(f"  ✓ graph_data_path = {graph_data_path}")

# Check output_dir matches
output_dir = rewire_config.get('output_dir')
expected_output = "results/experiment_015_gatv2_rewired/rewiring"
if output_dir != expected_output:
    warnings.append(f"output_dir mismatch: {output_dir} vs {expected_output}")
    print(f"  ⚠ output_dir = {output_dir}")
else:
    print(f"  ✓ output_dir = {output_dir}")

# Check k value
k = rewire_config.get('rewiring', {}).get('k', 10)
num_neighbors = exp_config.get('model', {}).get('structural_stream', {}).get('num_neighbors', 10)
if k != num_neighbors:
    warnings.append(f"k mismatch: rewiring k={k} vs model num_neighbors={num_neighbors}")
    print(f"  ⚠ k={k} but num_neighbors={num_neighbors}")
else:
    print(f"  ✓ k = num_neighbors = {k}")

# Check 3: Analyze StructuralStream code for parameter support
print("\n[4] Checking StructuralStream parameter support...")

with open('src/models/dual_stream.py', 'r') as f:
    dual_stream_code = f.read()

# Parse AST
try:
    tree = ast.parse(dual_stream_code)

    # Find StructuralStream.__init__
    structural_init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'StructuralStream':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    structural_init = item
                    break

    if structural_init:
        # Extract parameter names
        params = [arg.arg for arg in structural_init.args.args]

        required_params = [
            'layer_type',
            'num_heads',
            'use_residual',
            'use_denoising_gate',
            'denoising_gate_type',
            'denoising_gate_mode',
            'denoising_hard_threshold',
            'denoising_neighbor_dropout'
        ]

        for param in required_params:
            if param in params:
                print(f"  ✓ Parameter '{param}' supported")
            else:
                errors.append(f"StructuralStream missing parameter: {param}")
                print(f"  ✗ Parameter '{param}' NOT supported")

        # Check if gatv2 is handled
        if "'gatv2'" in dual_stream_code or '"gatv2"' in dual_stream_code:
            print(f"  ✓ GATv2 handling implemented")
        else:
            errors.append("GATv2 handling not found in code")
            print(f"  ✗ GATv2 handling NOT found")

    else:
        warnings.append("Could not parse StructuralStream.__init__")
        print(f"  ⚠ Could not parse StructuralStream.__init__")

except SyntaxError as e:
    errors.append(f"Syntax error in dual_stream.py: {e}")
    print(f"  ✗ Syntax error: {e}")

# Check 4: Verify GATv2 implementation
print("\n[5] Checking GATv2 implementation...")

gatv2_file = Path('src/layers/gatv2.py')
if gatv2_file.exists():
    with open(gatv2_file, 'r') as f:
        gatv2_code = f.read()

    # Check for GATv2Conv class
    if 'class GATv2Conv' in gatv2_code:
        print(f"  ✓ GATv2Conv class found")

        # Check for key GATv2 feature: LeakyReLU AFTER linear
        if 'leaky_relu' in gatv2_code.lower() or 'LeakyReLU' in gatv2_code:
            print(f"  ✓ LeakyReLU activation found (key GATv2 feature)")
        else:
            warnings.append("LeakyReLU not found in GATv2Conv")
            print(f"  ⚠ LeakyReLU not found")

        # Check for torch_scatter dependency
        if 'torch_scatter' in gatv2_code or 'scatter_add' in gatv2_code:
            print(f"  ✓ torch_scatter usage found")
        else:
            warnings.append("torch_scatter not used in GATv2Conv")
            print(f"  ⚠ torch_scatter not used")

    else:
        errors.append("GATv2Conv class not found in gatv2.py")
        print(f"  ✗ GATv2Conv class NOT found")
else:
    errors.append("gatv2.py file not found")
    print(f"  ✗ gatv2.py NOT found")

# Check 5: Verify denoising gate implementation
print("\n[6] Checking denoising gate implementation...")

denoising_file = Path('src/layers/denoising_gate.py')
if denoising_file.exists():
    with open(denoising_file, 'r') as f:
        denoising_code = f.read()

    classes = ['DenoisingGate', 'AdaptiveDenoisingGate', 'DenoisingGateWithNeighborDropout']
    for cls in classes:
        if f'class {cls}' in denoising_code:
            print(f"  ✓ {cls} class found")
        else:
            warnings.append(f"{cls} class not found")
            print(f"  ⚠ {cls} class NOT found")
else:
    errors.append("denoising_gate.py file not found")
    print(f"  ✗ denoising_gate.py NOT found")

# Check 6: Verify link prediction modules
print("\n[7] Checking link prediction modules...")

link_pred_files = {
    'src/phylogenetic/link_prediction.py': ['LinkPredictor', 'LinkPredictionEncoder', 'LinkPredictionDecoder'],
    'src/phylogenetic/train_link_predictor.py': ['LinkPredictionTrainer', 'train_link_predictor_from_graph'],
    'src/phylogenetic/graph_rewiring.py': ['rewire_graph_with_link_predictor', 'save_rewired_graph', 'load_rewired_graph']
}

for file_path, expected_items in link_pred_files.items():
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            code = f.read()

        for item in expected_items:
            if f'class {item}' in code or f'def {item}' in code:
                print(f"  ✓ {file_path}: {item} found")
            else:
                warnings.append(f"{item} not found in {file_path}")
                print(f"  ⚠ {file_path}: {item} NOT found")
    else:
        errors.append(f"{file_path} not found")
        print(f"  ✗ {file_path} NOT found")

# Check 7: Validate run_experiment_015.sh logic
print("\n[8] Validating run_experiment_015.sh logic...")

script_path = Path('run_experiment_015.sh')
if script_path.exists():
    script = script_path.read_text()

    # Check step sequence
    steps = [
        ('torch-scatter installation', 'torch-scatter'),
        ('First pass training', 'main.py.*--config'),
        ('Graph export', 'graph_data.pt'),
        ('Graph rewiring', 'run_graph_rewiring.py'),
        ('Rewired graph check', 'rewired_graph.pt'),
        ('Final training', 'main.py.*--config')
    ]

    for step_name, pattern in steps:
        if re.search(pattern, script):
            print(f"  ✓ {step_name}")
        else:
            warnings.append(f"Missing step in script: {step_name}")
            print(f"  ⚠ {step_name} - NOT found")

    # Check error handling
    if 'set -euo pipefail' in script:
        print(f"  ✓ Error handling enabled (set -euo pipefail)")
    else:
        warnings.append("Error handling not enabled")
        print(f"  ⚠ Error handling NOT enabled")

    # Check if it fails when rewired graph is missing
    if 'if [ ! -f "$REWIRED_PT" ]' in script:
        print(f"  ✓ Rewired graph existence check")
    else:
        warnings.append("No check for rewired graph existence")
        print(f"  ⚠ No rewired graph check")

else:
    errors.append("run_experiment_015.sh not found")
    print(f"  ✗ run_experiment_015.sh NOT found")

# Check 8: Path consistency
print("\n[9] Checking path consistency...")

paths_to_check = {
    'Experiment output': exp_config.get('data', {}).get('output_dir'),
    'Rewiring input': rewire_config.get('graph_data_path'),
    'Rewiring output': rewire_config.get('output_dir'),
    'Rewired graph': exp_config.get('phylogenetic', {}).get('rewired_graph_path'),
}

base_dir = "results/experiment_015_gatv2_rewired"
for name, path in paths_to_check.items():
    if path and path.startswith(base_dir):
        print(f"  ✓ {name}: {path}")
    else:
        warnings.append(f"{name} path may not be consistent: {path}")
        print(f"  ⚠ {name}: {path} (not under {base_dir})")

# Check 9: Training parameters
print("\n[10] Checking training parameters...")

num_epochs = exp_config.get('training', {}).get('num_epochs')
if num_epochs and num_epochs <= 10:
    print(f"  ✓ num_epochs = {num_epochs} (good for validation)")
else:
    info.append(f"num_epochs = {num_epochs} (might be long for validation)")
    print(f"  ℹ num_epochs = {num_epochs}")

batch_size = exp_config.get('training', {}).get('batch_size', 64)
print(f"  ℹ batch_size = {batch_size}")

# Check link predictor training epochs
lp_epochs = rewire_config.get('link_predictor', {}).get('num_epochs', 50)
if lp_epochs <= 20:
    print(f"  ✓ Link predictor epochs = {lp_epochs} (good for validation)")
else:
    info.append(f"Link predictor epochs = {lp_epochs} (might be long)")
    print(f"  ℹ Link predictor epochs = {lp_epochs}")

# Summary
print("\n" + "="*70)
print("STATIC VALIDATION SUMMARY")
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

if info:
    print(f"\nℹ️  INFO ({len(info)}):")
    for i, item in enumerate(info, 1):
        print(f"  {i}. {item}")

print("\n" + "="*70)
if errors:
    print("❌ VALIDATION FAILED - Fix errors before running")
    print("\nRecommended actions:")
    for error in errors[:3]:
        print(f"  • {error}")
elif warnings:
    print("⚠️  VALIDATION PASSED WITH WARNINGS")
    print("\nConsider reviewing:")
    for warning in warnings[:3]:
        print(f"  • {warning}")
    print("\n✅ Should be safe to run, but review warnings")
else:
    print("✅ VALIDATION PASSED - All checks OK!")
    print("\n🚀 Ready to run: ./run_experiment_015.sh")
print("="*70)
