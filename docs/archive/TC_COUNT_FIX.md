# Correção da Regra de Negócio: count_tc=1 → APFD=1.0

## Problema Identificado

Durante a análise do `experiment_017_ranking_corrected_03`, identificamos que builds com apenas 1 test case (count_tc=1) estavam recebendo APFD=0.5 em vez de APFD=1.0.

### Regra de Negócio (segundo master_vini)

```python
# Special case: single test case always returns 1.0
# With only one test, there's no ordering to optimize
if n == 1:
    return 1.0
```

**Justificativa:** Quando há apenas 1 test case em um build, não há otimização de ordem possível. O teste será executado de qualquer forma, então o APFD deve ser 1.0 (perfeito).

## Análise dos Dados

### ANTES da Correção

```
Total builds: 277
Builds com count_tc=1: 23
Mean APFD: 0.555148
Builds com APFD=1.0: 0
```

**Builds afetados (count_tc=1 com APFD=0.5):**
- T1TGN33.60-23, T1TZ33.3-60, T2SE33.73-8, T3TD33.10
- U1SJ34.2-54, U1TD34.100-7, U1TT34.126, U1UD34.16-7
- U2UAN34.72-35, U2UM34.27-8, U2UU34.17, U2UU34.40-5
- U2UUI34.40-6, U3TZ34.2-58, U3UX34.30, U3UX34.9
- UTPN34.141, UTR34.116, UTR34.173, UTT34.104
- UUG34.20, UUU34.21, UUU34.25

Total: **23 builds**

### DEPOIS da Correção

```
Total builds: 277
Builds com count_tc=1: 23
Mean APFD: 0.596664
Builds com APFD=1.0: 23
```

**Melhoria:**
- Mean APFD: 0.555148 → 0.596664 (+0.041516)
- Aumento percentual: **+7.48%**
- Todos os 23 builds com count_tc=1 agora têm APFD=1.0 ✓

## Implementação da Correção

### Script de Recalculo

Criado: `recalculate_apfd_fix_count_tc_1.py`

**Lógica principal:**
```python
if count_tc == 1:
    apfd = 1.0  # Regra de negócio: count_tc=1 → APFD=1.0
else:
    apfd = calculate_apfd_single_build(ranks, labels)
```

### Arquivos Atualizados

1. **Arquivo original (backup):**
   - `results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv_OLD.csv`
   - Mean APFD: 0.555148

2. **Arquivo corrigido (atual):**
   - `results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv.csv`
   - Mean APFD: 0.596664

## Verificação

```bash
# Verificar todos os builds com count_tc=1 agora têm APFD=1.0
awk -F',' 'NR>1 && $4==1 {print $2,$4,$6}' \
  results/experiment_017_ranking_corrected_03/apfd_per_build_FULL_testcsv.csv

# Output (todos com APFD=1.0):
T1TGN33.60-23 1 1.0
T1TZ33.3-60 1 1.0
T2SE33.73-8 1 1.0
...
UUU34.21 1 1.0
UUU34.25 1 1.0
```

✓ **Verificado:** Todos os 23 builds com count_tc=1 agora têm APFD=1.0

## Impacto nos Resultados

### Estatísticas Comparativas

| Métrica | ANTES | DEPOIS | Diferença |
|---------|-------|--------|-----------|
| Mean APFD | 0.555148 | 0.596664 | +0.041516 (+7.48%) |
| Median APFD | - | - | - |
| Builds APFD=1.0 | 0 | 23 | +23 |
| Builds APFD≥0.7 | - | - | - |
| Builds APFD≥0.5 | - | - | - |

### Distribuição de APFD

A correção moveu 23 builds de APFD=0.5 para APFD=1.0, melhorando significativamente a média geral do experimento.

## Conclusão

A correção foi aplicada com sucesso, alinhando o cálculo de APFD com a regra de negócio definida no master_vini. A melhoria de 7.48% no Mean APFD reflete corretamente o comportamento esperado para builds com apenas 1 test case.

## Próximos Passos

1. ✓ Aplicar correção no experiment_017_ranking_corrected_03
2. Verificar outros experimentos que possam ter o mesmo problema
3. Atualizar código base em src/evaluation/apfd.py para garantir aplicação correta em futuras execuções
4. Documentar em guias de uso

---

**Data da Correção:** 2025-11-06
**Experimento Afetado:** experiment_017_ranking_corrected_03
**Script:** recalculate_apfd_fix_count_tc_1.py
