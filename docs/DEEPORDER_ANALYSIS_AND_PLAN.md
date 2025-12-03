# Análise Científica do DeepOrder e Plano de Implementação para Filo-Priori V9

## 1. Análise Aprofundada do Paper DeepOrder

### 1.1 Insight Fundamental: Regressão vs Classificação

O DeepOrder trata a priorização de testes como um **problema de REGRESSÃO**, não de classificação.

```
DEEPORDER:
  Input: 14 features → Output: Priority Value p(ti) ∈ (0,1)
  Loss: MSE (Mean Squared Error)

FILO-PRIORI:
  Input: 1546 features → Output: P(Fail), P(Pass)
  Loss: Weighted Focal Loss
```

**Por que isso é crucial:**

| Aspecto | Classificação (Filo-Priori) | Regressão (DeepOrder) |
|---------|----------------------------|-----------------------|
| Objetivo | Separar classes (Fail/Pass) | Ordenar por prioridade |
| Otimiza | Fronteira de decisão | Ordem relativa |
| Problema | Pode classificar bem mas ordenar mal | Ordena diretamente |
| APFD | Indireto (via probabilidades) | Direto (via priority score) |

### 1.2 Fórmula de Prioridade do DeepOrder

```
p(ti) = Σ(j=1..m) wj × max(ES(i,j), 0)

Onde:
  - ES(i,j) ∈ {1 = falhou, 0 = passou, -1 = não executado}
  - wj = peso para ciclo j (ciclos recentes têm peso maior)
  - Σwj = 1
  - p(ti) ∈ (0, 1)
```

**Interpretação:**
- Um teste que falhou recentemente tem prioridade alta
- Um teste que nunca falhou tem prioridade ~0
- Um teste com falhas antigas tem prioridade média-baixa
- A fórmula cria um **ranking contínuo** que é o label de treino

### 1.3 Arquitetura do DeepOrder

```
Entrada (14 features):
├── ExecutionStatus[1..10]  (últimos 10 ciclos)
├── Duration                (tempo médio de execução)
├── LastRun                 (quando foi executado por último)
├── Distance                (diferença entre status mais antigo e mais recente)
└── ChangeInStatus          (quantas transições pass→fail)

Hidden Layers:
├── Layer 1: 10 neurônios
├── Layer 2: 20 neurônios
└── Layer 3: 15 neurônios

Ativação: Mish
Saída: 1 neurônio (priority value)
Total: 631 parâmetros
```

### 1.4 Tratamento de Desbalanceamento: SMOGN

O DeepOrder usa **SMOGN** (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise), não SMOTE comum:

- SMOTE: Para classificação
- SMOGN: Para regressão (gera pontos sintéticos em regiões sub-representadas do target contínuo)

**Taxas de falha nos datasets:**
- Cisco: 0.43%
- Paint Control: 0.19%
- IOF/ROL: 0.28%
- Google: 0.0025%

### 1.5 Achado Crítico: Histórico Longo Melhora APFD

O paper prova que usar mais de 4 ciclos de histórico **sempre** melhora o APFD:

```
Dataset      | 4 ciclos APFD | >4 ciclos APFD | Melhoria
-------------|---------------|----------------|----------
Cisco        | 0.42          | 0.52           | +24%
IOF/ROL      | 0.52          | 0.67           | +29%
Paint Control| 0.68          | 0.76           | +12%
GSDTSR       | 0.82          | 0.94           | +15%
```

---

## 2. Comparação Detalhada: DeepOrder vs Filo-Priori V9

### 2.1 Arquitetura

| Componente | DeepOrder | Filo-Priori V9 |
|------------|-----------|----------------|
| **Tipo** | MLP simples | Dual-Stream + GAT |
| **Parâmetros** | 631 | ~500,000+ |
| **Features** | 14 (histórico) | 1,546 (1536 semântico + 10 estrutural) |
| **Grafo** | Nenhum | Co-failure graph com GAT |
| **Semântica** | Nenhuma | SBERT all-mpnet-base-v2 |

### 2.2 Features Comparadas

**DeepOrder (14 features):**
1. ExecutionStatus[1-10] - últimos 10 ciclos
2. Duration - tempo médio
3. LastRun - última execução
4. Distance - |status_antigo - status_recente|
5. ChangeInStatus - transições pass→fail

**Filo-Priori V9 (1,546 features):**
- Semântico (1,536): SBERT embeddings de summary + steps + commits
- Estrutural (10): test_age, failure_rate, recent_failure_rate, flakiness_rate, consecutive_failures, max_consecutive_failures, failure_trend, commit_count, test_novelty, cr_count

### 2.3 O Que Filo-Priori TEM que DeepOrder NÃO TEM

✅ **Informação Semântica**: Descrições de testes, mensagens de commit
✅ **Grafo de Co-falhas**: Relações entre testes
✅ **Embeddings Ricos**: 768 dims para TC + 768 dims para Commits
✅ **Attention Mechanism**: GAT para propagar informação pelo grafo

### 2.4 O Que DeepOrder TEM que Filo-Priori NÃO TEM

❌ **Formulação de Regressão**: Label contínuo de prioridade
❌ **Histórico de Execuções**: Status dos últimos N ciclos
❌ **Distance Feature**: Captura mudança de comportamento
❌ **ChangeInStatus**: Captura flakiness de forma diferente
❌ **SMOGN**: Data augmentation para regressão

---

## 3. Por Que as Implementações Anteriores Falharam

### 3.1 Proposta #1: Stratified Sampling (APFD: 0.6413 → 0.5772)

**Problema identificado:**
```python
# Bug no StratifiedBuildSampler:
self.n_batches = n_pos // min_positives  # Limitou amostras!

# Com n_pos=37 e min_positives=4:
# n_batches = 37 // 4 = 9 batches
# Amostras por época = 9 × 32 = 288 (em vez de ~1400)
```

**Resultado:** Modelo viu apenas 20% dos dados por época → underfitting severo.

### 3.2 Proposta #2 e #3: Ranking Loss + LightGBM (APFD: 0.6413 → 0.6204)

**Problemas:**
1. **Conflito de Losses**: Focal loss puxa para 0/1, ranking loss puxa para ordenação
2. **Batch Composition**: Com ~1 falha por batch, não há pares suficientes para ranking
3. **LightGBM Stacking**: Se predições neurais são ruins, LightGBM não conserta

### 3.3 Lição Aprendida

> **As propostas tentaram ADICIONAR complexidade ao modelo atual. O DeepOrder mostra que SIMPLICIDADE com a FORMULAÇÃO CORRETA é mais eficaz.**

---

## 4. Plano de Implementação Baseado no DeepOrder

### Filosofia do Plano

Em vez de adicionar mais componentes ao modelo complexo, vamos:
1. **Reformular o problema** como regressão (parcialmente)
2. **Adicionar features inspiradas no DeepOrder**
3. **Manter as vantagens do Filo-Priori** (semântica, grafo)

### FASE 1: Adicionar Priority Score como Label Auxiliar (ALTA PRIORIDADE)

**Objetivo:** Criar labels de prioridade contínuos baseados no histórico, similar ao DeepOrder.

**Implementação:**

```python
# src/preprocessing/priority_score_generator.py

import numpy as np
import pandas as pd
from typing import Dict, List

class PriorityScoreGenerator:
    """
    Generates DeepOrder-style priority scores from test execution history.

    Priority formula: p(ti) = Σ wj × max(ES(i,j), 0)
    Where wj decays exponentially for older cycles.
    """

    def __init__(
        self,
        num_cycles: int = 10,
        decay_type: str = 'exponential',  # 'exponential', 'linear', 'inverse'
        decay_factor: float = 0.8
    ):
        self.num_cycles = num_cycles
        self.decay_type = decay_type
        self.decay_factor = decay_factor
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute cycle weights (more recent = higher weight)."""
        if self.decay_type == 'exponential':
            # w_j = decay^(m-j) where m is most recent
            raw_weights = np.array([
                self.decay_factor ** (self.num_cycles - 1 - j)
                for j in range(self.num_cycles)
            ])
        elif self.decay_type == 'linear':
            raw_weights = np.linspace(0.1, 1.0, self.num_cycles)
        elif self.decay_type == 'inverse':
            raw_weights = 1.0 / np.arange(self.num_cycles, 0, -1)
        else:
            raw_weights = np.ones(self.num_cycles)

        # Normalize so sum = 1
        return raw_weights / raw_weights.sum()

    def compute_priority(
        self,
        execution_history: List[int]
    ) -> float:
        """
        Compute priority score for a test case.

        Args:
            execution_history: List of execution statuses
                               [1=fail, 0=pass, -1=not_executed]
                               Ordered from oldest to most recent

        Returns:
            Priority score in (0, 1)
        """
        # Pad or truncate to num_cycles
        history = execution_history[-self.num_cycles:]
        if len(history) < self.num_cycles:
            # Pad with -1 (not executed) for missing old cycles
            history = [-1] * (self.num_cycles - len(history)) + history

        history = np.array(history)

        # Apply formula: p = Σ wj × max(ES, 0)
        # max(ES, 0) converts -1 to 0 (ignore not-executed)
        contribution = np.maximum(history, 0)
        priority = np.dot(self.weights, contribution)

        return float(priority)

    def compute_priorities_for_dataset(
        self,
        df: pd.DataFrame,
        build_col: str = 'Build_ID',
        tc_col: str = 'TC_Name',
        result_col: str = 'TE_Test_Result'
    ) -> pd.DataFrame:
        """
        Compute priority scores for all test cases in dataset.

        This processes the data chronologically and computes
        priority for each test case in each build.
        """
        df = df.copy()
        df['priority_score'] = 0.0

        # Track execution history per test case
        tc_history: Dict[str, List[int]] = {}

        # Process builds chronologically
        builds = df[build_col].unique()

        for build_id in builds:
            build_mask = df[build_col] == build_id
            build_df = df[build_mask]

            for idx, row in build_df.iterrows():
                tc_name = row[tc_col]

                # Get history for this TC
                history = tc_history.get(tc_name, [])

                # Compute priority based on PAST history (before this execution)
                if len(history) > 0:
                    priority = self.compute_priority(history)
                else:
                    priority = 0.0  # New test, no history

                df.loc[idx, 'priority_score'] = priority

                # Update history with current result
                result = row[result_col]
                if result == 'Fail':
                    status = 1
                elif result == 'Pass':
                    status = 0
                else:
                    status = -1

                if tc_name not in tc_history:
                    tc_history[tc_name] = []
                tc_history[tc_name].append(status)

        return df
```

**Integração no main.py:**

```python
# Em prepare_data(), após carregar os dados:
from preprocessing.priority_score_generator import PriorityScoreGenerator

priority_generator = PriorityScoreGenerator(
    num_cycles=10,
    decay_type='exponential',
    decay_factor=0.8
)

# Adicionar priority_score aos DataFrames
df_train = priority_generator.compute_priorities_for_dataset(df_train)
df_val = priority_generator.compute_priorities_for_dataset(df_val)
df_test = priority_generator.compute_priorities_for_dataset(df_test)

# Usar como label auxiliar
train_data['priority_scores'] = df_train['priority_score'].values
```

**Impacto Esperado:** Este é o insight mais importante do DeepOrder. Criar labels que representam diretamente o objetivo de ranking.

---

### FASE 2: Dual-Head Model (Classificação + Regressão)

**Objetivo:** Adicionar uma segunda saída de regressão para prever priority_score.

**Implementação:**

```python
# src/models/dual_head_classifier.py

import torch
import torch.nn as nn

class DualHeadClassifier(nn.Module):
    """
    Classifier with two heads:
    1. Classification head: Predicts P(Fail), P(Pass)
    2. Regression head: Predicts priority score (0-1)

    The regression head directly optimizes for ranking.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Regression head (priority prediction)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in (0, 1)
        )

    def forward(self, x):
        """
        Returns:
            cls_logits: [B, num_classes] classification logits
            priority: [B, 1] priority scores
        """
        shared_features = self.shared(x)
        cls_logits = self.cls_head(shared_features)
        priority = self.reg_head(shared_features)
        return cls_logits, priority
```

**Nova Loss Function:**

```python
# src/training/dual_loss.py

class DualTaskLoss(nn.Module):
    """
    Combined loss for classification + priority regression.

    Loss = α × FocalLoss(cls) + β × MSE(priority)

    The key insight from DeepOrder: MSE on priority directly
    optimizes for ranking, which is what APFD measures.
    """

    def __init__(
        self,
        cls_weight: float = 0.5,
        reg_weight: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

        # Focal loss for classification
        self.focal_loss = WeightedFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            weight=class_weights
        )

        # MSE for priority regression
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        priority_pred: torch.Tensor,
        labels: torch.Tensor,
        priority_target: torch.Tensor
    ):
        cls_loss = self.focal_loss(cls_logits, labels)
        reg_loss = self.mse_loss(priority_pred.squeeze(), priority_target)

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total_loss, cls_loss, reg_loss
```

**Impacto Esperado:** +5-10% APFD ao otimizar diretamente para ranking.

---

### FASE 3: Adicionar Features Inspiradas no DeepOrder

**Objetivo:** Incorporar as features do DeepOrder que o Filo-Priori não tem.

**Novas features a adicionar:**

```python
# Adicionar ao structural_feature_extractor_v2_5.py:

NEW_DEEPORDER_FEATURES = [
    'execution_status_last_1',     # Status no ciclo mais recente
    'execution_status_last_2',     # Status 2 ciclos atrás
    'execution_status_last_3',     # Status 3 ciclos atrás
    'execution_status_last_5',     # Status 5 ciclos atrás
    'execution_status_last_10',    # Status 10 ciclos atrás
    'distance',                    # |status_antigo - status_recente|
    'status_changes',              # Número de transições pass↔fail
    'cycles_since_last_fail',      # Ciclos desde última falha
    'fail_rate_last_10_cycles',    # Taxa de falha nos últimos 10
]
```

**Implementação:**

```python
def extract_deeporder_features(
    tc_history: Dict[str, List[int]],
    tc_name: str
) -> np.ndarray:
    """Extract DeepOrder-inspired features for a test case."""
    history = tc_history.get(tc_name, [])

    features = []

    # Execution statuses at specific points
    for offset in [1, 2, 3, 5, 10]:
        if len(history) >= offset:
            features.append(history[-offset])
        else:
            features.append(-1)  # Not executed

    # Distance: difference between oldest and newest
    if len(history) >= 2:
        distance = abs(history[-1] - history[0])
    else:
        distance = 0
    features.append(distance)

    # Status changes (transitions)
    changes = 0
    for i in range(1, len(history)):
        if history[i] != history[i-1] and history[i] != -1 and history[i-1] != -1:
            changes += 1
    features.append(changes)

    # Cycles since last fail
    cycles_since_fail = 0
    for status in reversed(history):
        if status == 1:  # Fail
            break
        cycles_since_fail += 1
    features.append(min(cycles_since_fail, 100))  # Cap at 100

    # Fail rate in last 10 cycles
    recent = [s for s in history[-10:] if s != -1]
    if len(recent) > 0:
        fail_rate = sum(1 for s in recent if s == 1) / len(recent)
    else:
        fail_rate = 0
    features.append(fail_rate)

    return np.array(features)
```

**Impacto Esperado:** +2-4% APFD ao capturar padrões temporais.

---

### FASE 4: Simplificação da Arquitetura (Opcional)

**Objetivo:** Testar se uma arquitetura mais simples performa melhor.

O DeepOrder mostra que 631 parâmetros podem superar modelos complexos. Vale testar:

```python
class SimplePriorityMLP(nn.Module):
    """
    DeepOrder-inspired simple MLP for priority prediction.

    Combines:
    - Filo-Priori's rich features (semantic + structural)
    - DeepOrder's simple architecture and regression formulation
    """

    def __init__(
        self,
        semantic_dim: int = 1536,
        structural_dim: int = 10,
        deeporder_dim: int = 9,  # New DeepOrder features
        hidden_dims: List[int] = [256, 128, 64]
    ):
        super().__init__()

        input_dim = semantic_dim + structural_dim + deeporder_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Mish(),  # DeepOrder usa Mish
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, semantic, structural, deeporder_features):
        x = torch.cat([semantic, structural, deeporder_features], dim=-1)
        return self.network(x)
```

**Impacto Esperado:** Baseline para comparação - pode ser similar ou melhor que modelo complexo.

---

## 5. Cronograma de Implementação

### Semana 1: FASE 1 + Testes

| Dia | Tarefa |
|-----|--------|
| 1-2 | Implementar `PriorityScoreGenerator` |
| 3 | Integrar no pipeline de dados |
| 4-5 | Treinar modelo baseline com priority_score como feature |
| 6-7 | Avaliar impacto e ajustar |

### Semana 2: FASE 2 + FASE 3

| Dia | Tarefa |
|-----|--------|
| 1-2 | Implementar `DualHeadClassifier` |
| 3 | Implementar `DualTaskLoss` |
| 4-5 | Adicionar DeepOrder features ao extrator |
| 6-7 | Treinar e avaliar |

### Semana 3: Refinamento

| Dia | Tarefa |
|-----|--------|
| 1-3 | Hyperparameter tuning |
| 4-5 | Ablation study |
| 6-7 | Documentação e análise |

---

## 6. Métricas de Sucesso

### Objetivo Primário
- **APFD ≥ 0.70** (baseline atual: 0.6344)

### Métricas Secundárias
- F1-macro ≥ 0.55
- Precision@10 (top-10 contém falhas?)
- First-Fault-Time (tempo até primeira falha)

### Critérios de Validação

1. **Cada mudança deve ser testada isoladamente** (uma de cada vez)
2. **Manter baseline estável** antes de cada experimento
3. **Rollback imediato** se APFD cair >2%

---

## 7. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Priority scores não melhoram ranking | Média | Alto | Usar como feature adicional, não substituto |
| Dual-head instável no treino | Média | Médio | Warmup no regression head |
| Overfitting com mais features | Baixa | Médio | Regularização e dropout |
| Tempo de treino aumenta muito | Baixa | Baixo | Batch size optimization |

---

## 8. Conclusão

O DeepOrder oferece insights valiosos:

1. **Regressão > Classificação** para ranking
2. **Simplicidade pode superar complexidade** quando bem formulada
3. **Histórico longo é importante** (>4 ciclos)
4. **Priority score como label** é mais diretamente conectado ao APFD

O plano proposto mantém as vantagens do Filo-Priori (semântica, grafo) enquanto incorpora a formulação superior do DeepOrder (regressão de prioridade).

**Recomendação:** Começar pela FASE 1 (Priority Score) pois é a mudança mais fundamentada cientificamente e com menor risco de regressão.
