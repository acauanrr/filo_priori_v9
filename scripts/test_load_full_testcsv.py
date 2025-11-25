"""
Teste rápido para validar carregamento do test.csv completo

Este script testa APENAS o carregamento dos dados, sem treinar modelo.
Objetivo: Verificar se conseguimos carregar os 277 builds corretamente.
"""

import yaml
import pandas as pd
from src.preprocessing import DataLoader

# Load config
with open('configs/experiment_012_best_practices.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("TESTE: Carregamento do test.csv completo")
print("=" * 80)

# Create data loader
data_loader = DataLoader(config)

# Load FULL test dataset
print("\n1. Carregando test.csv completo...")
test_df_full = data_loader.load_full_test_dataset()

print(f"\n✅ Loaded test.csv:")
print(f"   Total samples: {len(test_df_full)}")
print(f"   Total builds: {test_df_full['Build_ID'].nunique()}")

# Check builds with Fail
builds_with_fail = test_df_full[test_df_full['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
print(f"   Builds with 'Fail': {builds_with_fail}")

# Validate
print("\n" + "=" * 80)
print("VALIDAÇÃO")
print("=" * 80)

if len(test_df_full) == 31333:
    print("✅ Total samples: 31333 (CORRETO)")
else:
    print(f"❌ Total samples: {len(test_df_full)} (ESPERADO: 31333)")

if test_df_full['Build_ID'].nunique() == 1365:
    print("✅ Total builds: 1365 (CORRETO)")
else:
    print(f"⚠️  Total builds: {test_df_full['Build_ID'].nunique()} (ESPERADO: 1365)")

if builds_with_fail == 277:
    print("✅ Builds com Fail: 277 (CORRETO!)")
else:
    print(f"❌ Builds com Fail: {builds_with_fail} (ESPERADO: 277)")

# Show sample of builds with Fail
print(f"\n📊 Primeiras 10 builds com Fail:")
builds_fail_df = test_df_full[test_df_full['TE_Test_Result'] == 'Fail']
unique_builds = builds_fail_df['Build_ID'].unique()[:10]
print(unique_builds)

print("\n" + "=" * 80)
print("TESTE CONCLUÍDO!")
print("=" * 80)
