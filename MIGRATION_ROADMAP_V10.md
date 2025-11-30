# Roadmap de Migração: Filo-Priori V9 → V10

## Visão Geral da Migração

Este documento detalha o plano de migração do Filo-Priori V9 (atual) para o V10 (proposto), implementando as três inovações críticas do Plano de Desenvolvimento Estratégico.

---

## Sumário Executivo

| Componente | V9 (Atual) | V10 (Proposto) | Impacto Esperado |
|------------|------------|----------------|------------------|
| Encoder Semântico | SBERT (all-mpnet) | CodeBERT + Co-Attention | Melhor captura de semântica de código |
| Grafo | Multi-edge estático | Time-Decay dinâmico | Captura "burstiness" de falhas |
| Função de Perda | Weighted Focal Loss | LambdaRank | Otimização direta do APFD |
| Arquitetura | Dual-Stream | Hybrid Residual | Aproveita força das heurísticas |

**Meta**: Superar baseline "Recently-Failed" no RTPTorrent de forma estatisticamente significativa.

---

## Fase 1: Infraestrutura e Preparação (Semana 1)

### 1.1 Atualização de Dependências

```bash
# Novas dependências necessárias
pip install transformers>=4.35.0  # CodeBERT
pip install torch-geometric>=2.4.0  # GAT atualizado
pip install lmdb  # Cache de embeddings
pip install gitpython  # Mineração de Git
```

**Arquivo**: `requirements_v10.txt`

### 1.2 Estrutura de Diretórios V10

```
src/
├── v10/                           # Novo módulo V10
│   ├── __init__.py
│   ├── encoders/
│   │   ├── codebert_encoder.py    # CodeBERT wrapper
│   │   ├── co_attention.py        # Camada de co-atenção
│   │   └── tokenizer.py           # CamelCase splitter
│   ├── graphs/
│   │   ├── time_decay_builder.py  # Grafo com decaimento
│   │   ├── co_change_miner.py     # Mineração de Git
│   │   └── temporal_gat.py        # GAT temporal
│   ├── ranking/
│   │   ├── lambda_rank.py         # LambdaRank loss
│   │   ├── lambda_loss.py         # Variante moderna
│   │   └── ndcg_utils.py          # Utilidades NDCG
│   ├── features/
│   │   ├── heuristic_features.py  # Features explícitas
│   │   └── recency_transform.py   # Transformação de recência
│   └── models/
│       ├── hybrid_model.py        # Arquitetura híbrida
│       └── residual_fusion.py     # Fusão residual
```

### 1.3 Configuração Base V10

```yaml
# configs/experiment_v10_base.yaml
experiment:
  name: "filo_priori_v10"
  version: "10.0"

# Novo: CodeBERT
encoder:
  type: "codebert"
  model_name: "microsoft/codebert-base"
  use_co_attention: true
  co_attention_heads: 8

# Novo: Time-Decay Graph
graph:
  type: "time_decay"
  decay_lambda: 0.1  # Hiperparâmetro de decaimento
  min_co_changes: 2
  lookback_days: 365

# Novo: LambdaRank
training:
  loss:
    type: "lambda_rank"
    sigma: 1.0
    ndcg_at_k: 10

# Novo: Residual Learning
model:
  type: "hybrid_residual"
  use_heuristic_bias: true
  heuristic_weight: 0.3  # Peso inicial das heurísticas
```

---

## Fase 2: Time-Decay Graph Builder (Semana 2)

### 2.1 Objetivo

Substituir o grafo multi-edge estático por um grafo dinâmico onde os pesos das arestas decaem exponencialmente com o tempo.

### 2.2 Fórmula Matemática

$$W_{ij}(t) = \sum_{k \in \text{Commits}} \mathbb{1}(\text{co\_change}_k(i,j)) \cdot e^{-\lambda (t - t_k)}$$

Onde:
- $t$ = tempo atual (build atual)
- $t_k$ = tempo do commit $k$
- $\lambda$ = taxa de decaimento (hiperparâmetro)
- $\mathbb{1}$ = indicador de co-mudança

### 2.3 Implementação

```python
# src/v10/graphs/time_decay_builder.py

class TimeDecayGraphBuilder:
    """
    Constrói grafo de co-mudança com decaimento temporal.

    Diferente do V9 (pesos estáticos), este grafo:
    1. Minera histórico de commits do Git
    2. Calcula pesos baseados em co-mudanças recentes
    3. Aplica decaimento exponencial
    """

    def __init__(self, decay_lambda=0.1, lookback_days=365):
        self.decay_lambda = decay_lambda
        self.lookback_days = lookback_days

    def compute_edge_weight(self, co_change_times, current_time):
        """
        W_ij(t) = Σ exp(-λ * (t - t_k))
        """
        weights = []
        for t_k in co_change_times:
            delta = (current_time - t_k).days
            weight = math.exp(-self.decay_lambda * delta)
            weights.append(weight)
        return sum(weights)
```

### 2.4 Integração com V9

| V9 (Atual) | V10 (Novo) | Migração |
|------------|------------|----------|
| `multi_edge_graph_builder.py` | `time_decay_builder.py` | Substituição |
| Pesos fixos: 1.0, 0.5, 0.3 | Pesos dinâmicos: $e^{-\lambda t}$ | Nova fórmula |
| Baseado em execuções | Baseado em commits | Nova fonte de dados |

### 2.5 Testes de Validação

```python
# Teste: Decaimento correto
def test_time_decay():
    builder = TimeDecayGraphBuilder(decay_lambda=0.1)

    # Co-mudança de ontem deve ter peso ~0.9
    weight_1day = builder.compute_edge_weight([yesterday], today)
    assert 0.89 < weight_1day < 0.91

    # Co-mudança de 30 dias deve ter peso ~0.05
    weight_30days = builder.compute_edge_weight([month_ago], today)
    assert weight_30days < 0.1
```

---

## Fase 3: CodeBERT Encoder com Co-Attention (Semana 3)

### 3.1 Objetivo

Substituir SBERT por CodeBERT para melhor captura de semântica de código, com camada de co-atenção para focar nas partes relevantes.

### 3.2 Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODEBERT + CO-ATTENTION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────┐              │
│  │   Test Code      │        │   Changed Code   │              │
│  │   Tokens         │        │   Tokens         │              │
│  └────────┬─────────┘        └────────┬─────────┘              │
│           │                           │                        │
│           ▼                           ▼                        │
│  ┌──────────────────┐        ┌──────────────────┐              │
│  │    CodeBERT      │        │    CodeBERT      │              │
│  │    Encoder       │        │    Encoder       │              │
│  │    [CLS] + seq   │        │    [CLS] + seq   │              │
│  └────────┬─────────┘        └────────┬─────────┘              │
│           │                           │                        │
│           │ h_test [768]              │ h_code [768]           │
│           │                           │                        │
│           └───────────┬───────────────┘                        │
│                       ▼                                        │
│           ┌───────────────────────┐                            │
│           │    CO-ATTENTION       │                            │
│           │                       │                            │
│           │  Q = h_test           │                            │
│           │  K, V = h_code        │                            │
│           │                       │                            │
│           │  Attention weights    │                            │
│           │  identify relevant    │                            │
│           │  code sections        │                            │
│           └───────────┬───────────┘                            │
│                       │                                        │
│                       ▼                                        │
│           ┌───────────────────────┐                            │
│           │  h_semantic [768]     │                            │
│           │  (attended output)    │                            │
│           └───────────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementação

```python
# src/v10/encoders/codebert_encoder.py

class CodeBERTEncoder(nn.Module):
    """
    Encoder baseado em CodeBERT para código-fonte.

    Diferente do SBERT (V9):
    - Pré-treinado em código (CodeSearchNet)
    - Entende sintaxe de programação
    - Melhor para identificadores Java
    """

    def __init__(self, model_name="microsoft/codebert-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.output_dim = 768

    def forward(self, text_batch):
        inputs = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token
```

```python
# src/v10/encoders/co_attention.py

class CoAttention(nn.Module):
    """
    Camada de co-atenção entre teste e código modificado.

    Permite que o modelo "foque" nas partes do teste
    que são mais relevantes para as linhas alteradas.
    """

    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_test, h_code):
        # Test "pergunta" ao código
        attended, weights = self.attention(
            query=h_test.unsqueeze(1),
            key=h_code.unsqueeze(1),
            value=h_code.unsqueeze(1)
        )
        return self.norm(h_test + attended.squeeze(1)), weights
```

### 3.4 CamelCase Tokenizer

```python
# src/v10/encoders/tokenizer.py

class CamelCaseSplitter:
    """
    Divide identificadores CamelCase em tokens.

    Exemplo:
        "AbstractFactoryTest" → ["Abstract", "Factory", "Test"]
        "PaymentProcessingTest" → ["Payment", "Processing", "Test"]
    """

    def split(self, identifier):
        # Regex para CamelCase
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', identifier)
        return [t.lower() for t in tokens]
```

### 3.5 Migração de SBERT → CodeBERT

| Aspecto | SBERT (V9) | CodeBERT (V10) |
|---------|------------|----------------|
| Modelo | all-mpnet-base-v2 | microsoft/codebert-base |
| Pré-treino | Sentenças naturais | Código-fonte |
| Dimensão | 768 | 768 |
| Tokenização | WordPiece | BPE + CamelCase |
| Co-Attention | Não | Sim |

---

## Fase 4: LambdaRank Loss (Semana 4)

### 4.1 Objetivo

Substituir Weighted Focal Loss por LambdaRank para otimizar diretamente a métrica APFD/NDCG.

### 4.2 Problema com Focal Loss

```
Focal Loss: Otimiza classificação binária (Pass/Fail)
            ↓
            Não considera a ORDEM relativa
            ↓
            Subótimo para APFD (métrica de ranking)
```

### 4.3 LambdaRank: Intuição

```
LambdaRank: Para cada par (i, j) onde i é mais relevante que j:

            Se modelo erra a ordem (j > i):
                Penalidade = |ΔNDCG| × gradiente de cross-entropy

            Resultado: Gradientes proporcionais ao impacto no APFD
```

### 4.4 Fórmula Matemática

Para um par de documentos $(i, j)$ onde $y_i > y_j$ (i é mais relevante):

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta \text{NDCG}_{ij}|$$

Onde:
- $s_i, s_j$ = scores preditos
- $\sigma$ = hiperparâmetro de escala
- $\Delta\text{NDCG}_{ij}$ = mudança no NDCG se trocarmos $i$ e $j$

### 4.5 Implementação

```python
# src/v10/ranking/lambda_rank.py

class LambdaRankLoss(nn.Module):
    """
    Função de perda LambdaRank para Learning-to-Rank.

    Diferente de Focal Loss (V9):
    - Otimiza diretamente NDCG/APFD
    - Considera ordem relativa, não apenas classificação
    - Melhor para esparsidade extrema
    """

    def __init__(self, sigma=1.0, ndcg_at_k=None):
        super().__init__()
        self.sigma = sigma
        self.ndcg_at_k = ndcg_at_k

    def forward(self, scores, relevances, mask=None):
        """
        Args:
            scores: [batch, num_items] - scores preditos
            relevances: [batch, num_items] - 0/1 (pass/fail)
            mask: [batch, num_items] - máscara de padding

        Returns:
            loss: escalar
        """
        batch_size, num_items = scores.shape
        total_loss = 0.0

        for b in range(batch_size):
            s = scores[b]  # [num_items]
            y = relevances[b]  # [num_items]

            # Encontra pares onde y_i > y_j (falhas vs passes)
            fail_indices = (y == 1).nonzero().squeeze(-1)
            pass_indices = (y == 0).nonzero().squeeze(-1)

            if len(fail_indices) == 0 or len(pass_indices) == 0:
                continue

            # Para cada par (fail, pass)
            for i in fail_indices:
                for j in pass_indices:
                    # Diferença de scores
                    s_diff = s[i] - s[j]

                    # Delta NDCG
                    delta_ndcg = self._compute_delta_ndcg(y, i, j)

                    # Lambda gradient
                    sigmoid = torch.sigmoid(-self.sigma * s_diff)
                    lambda_ij = self.sigma * sigmoid * abs(delta_ndcg)

                    # Acumula perda
                    total_loss += lambda_ij * F.softplus(-s_diff)

        return total_loss / batch_size

    def _compute_delta_ndcg(self, relevances, i, j):
        """
        Calcula mudança no NDCG se trocarmos posições i e j.
        """
        # Implementação simplificada
        # NDCG = DCG / IDCG
        # Delta = |DCG(swap) - DCG(original)| / IDCG

        n = len(relevances)
        pos_i = i + 1  # 1-indexed
        pos_j = j + 1

        # Ganho: 2^rel - 1
        gain_i = (2 ** relevances[i].item()) - 1
        gain_j = (2 ** relevances[j].item()) - 1

        # Desconto: 1 / log2(pos + 1)
        disc_i = 1.0 / math.log2(pos_i + 1)
        disc_j = 1.0 / math.log2(pos_j + 1)

        # Delta DCG
        delta = abs((gain_i - gain_j) * (disc_i - disc_j))

        return delta
```

### 4.6 Variante Moderna: ApproxNDCG Loss

```python
# src/v10/ranking/approx_ndcg.py

class ApproxNDCGLoss(nn.Module):
    """
    NDCG diferenciável usando softmax para aproximar ranking.

    Mais estável que LambdaRank puro.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, relevances):
        # Soft ranking via softmax
        soft_ranks = torch.softmax(-scores / self.temperature, dim=-1)

        # DCG com soft positions
        gains = (2 ** relevances) - 1
        discounts = 1.0 / torch.log2(soft_ranks.cumsum(dim=-1) + 2)

        dcg = (gains * discounts).sum(dim=-1)
        idcg = self._compute_idcg(relevances)

        ndcg = dcg / (idcg + 1e-8)

        return 1.0 - ndcg.mean()  # Minimizar = maximizar NDCG
```

---

## Fase 5: Arquitetura Híbrida com Residual Learning (Semana 5)

### 5.1 Objetivo

Integrar os três módulos (CodeBERT, Time-Decay Graph, LambdaRank) em uma arquitetura unificada que aproveita heurísticas como bias.

### 5.2 Conceito de Residual Learning

```
Ideia Central:
    O modelo neural aprende a CORRIGIR as heurísticas,
    não a substituí-las.

    score_final = h_heuristic + f_neural(h_sem, h_graph)
                  ↑               ↑
                  baseline        correção/resíduo
```

### 5.3 Arquitetura Completa V10

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FILO-PRIORI V10 - HYBRID RESIDUAL                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FEATURE EXTRACTION                          │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ HEURISTIC    │  │ CODEBERT     │  │ TIME-DECAY   │           │   │
│  │  │ FEATURES     │  │ + CO-ATTN    │  │ GRAPH + GAT  │           │   │
│  │  │              │  │              │  │              │           │   │
│  │  │ • recency    │  │ Test tokens  │  │ Co-change    │           │   │
│  │  │ • fail_rate  │  │ Code tokens  │  │ edges with   │           │   │
│  │  │ • duration   │  │ → CodeBERT   │  │ exp decay    │           │   │
│  │  │              │  │ → Co-Attn    │  │ → GAT        │           │   │
│  │  │ h_heur [3]   │  │ h_sem [768]  │  │ h_graph [256]│           │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │   │
│  │         │                 │                 │                    │   │
│  └─────────┼─────────────────┼─────────────────┼────────────────────┘   │
│            │                 │                 │                        │
│            │                 └────────┬────────┘                        │
│            │                          │                                 │
│            │                          ▼                                 │
│            │          ┌───────────────────────────────┐                 │
│            │          │      NEURAL RESIDUAL          │                 │
│            │          │                               │                 │
│            │          │  concat([h_sem, h_graph])     │                 │
│            │          │  → MLP: 1024 → 256 → 64 → 1   │                 │
│            │          │  → δ_neural (correção)        │                 │
│            │          └───────────────┬───────────────┘                 │
│            │                          │                                 │
│            │                          │ δ_neural [1]                    │
│            │                          │                                 │
│            ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      RESIDUAL FUSION                             │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │   h_baseline = MLP(h_heur) → score_heuristic [1]                 │   │
│  │                                                                   │   │
│  │   score_final = α × score_heuristic + (1-α) × δ_neural           │   │
│  │                 ↑                                                 │   │
│  │                 learnable weight (starts at 0.7)                  │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      LAMBDARANK LOSS                             │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │   loss = LambdaRank(score_final, relevances)                     │   │
│  │                                                                   │   │
│  │   Otimiza diretamente: APFD/NDCG                                 │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Implementação

```python
# src/v10/models/hybrid_model.py

class FiloPrioriV10(nn.Module):
    """
    Arquitetura híbrida V10 com Residual Learning.

    Combina:
    1. CodeBERT + Co-Attention (semântica)
    2. Time-Decay Graph + GAT (estrutura temporal)
    3. Heuristic Features (baseline forte)

    O modelo aprende a corrigir as heurísticas, não substituí-las.
    """

    def __init__(self, config):
        super().__init__()

        # Módulo 1: Semântico (CodeBERT)
        self.codebert = CodeBERTEncoder(config.encoder.model_name)
        self.co_attention = CoAttention(
            hidden_dim=768,
            num_heads=config.encoder.co_attention_heads
        )

        # Módulo 2: Grafo Temporal
        self.gat = GATConv(
            in_channels=config.graph.node_features,
            out_channels=256,
            heads=4,
            concat=False
        )

        # Módulo 3: Heurísticas
        self.heuristic_encoder = nn.Sequential(
            nn.Linear(3, 32),  # recency, fail_rate, duration
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Fusão Neural (aprende resíduo)
        self.neural_residual = nn.Sequential(
            nn.Linear(768 + 256, 256),  # h_sem + h_graph
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Peso learnable para fusão
        self.alpha = nn.Parameter(torch.tensor(0.7))  # Começa confiando na heurística

    def forward(self, test_tokens, code_tokens, graph_data, heuristic_features):
        # 1. Semântico
        h_test = self.codebert(test_tokens)
        h_code = self.codebert(code_tokens)
        h_sem, attn_weights = self.co_attention(h_test, h_code)

        # 2. Grafo Temporal
        h_graph = self.gat(
            graph_data.x,
            graph_data.edge_index,
            edge_attr=graph_data.edge_weight  # Pesos com decaimento
        )

        # 3. Heurísticas (baseline)
        score_heuristic = self.heuristic_encoder(heuristic_features)

        # 4. Resíduo Neural
        h_combined = torch.cat([h_sem, h_graph], dim=-1)
        delta_neural = self.neural_residual(h_combined)

        # 5. Fusão Residual
        alpha = torch.sigmoid(self.alpha)  # Garante [0, 1]
        score_final = alpha * score_heuristic + (1 - alpha) * delta_neural

        return score_final, attn_weights
```

### 5.5 Features Heurísticas Explícitas

```python
# src/v10/features/heuristic_features.py

class HeuristicFeatureExtractor:
    """
    Extrai features heurísticas fortes para Residual Learning.

    Estas features capturam o sinal do baseline "Recently-Failed".
    """

    def extract(self, test_history, current_build):
        features = {}

        # 1. Recência transformada
        # f = 1 / log(1 + Δbuilds)
        builds_since_fail = self._get_builds_since_failure(test_history)
        features['recency'] = 1.0 / math.log(1 + builds_since_fail + 1)

        # 2. Taxa de falha histórica
        features['fail_rate'] = test_history['failures'] / max(test_history['executions'], 1)

        # 3. Duração média (para APFD-c)
        features['duration'] = test_history['avg_duration']

        return torch.tensor([
            features['recency'],
            features['fail_rate'],
            features['duration']
        ])
```

---

## Fase 6: Pipeline de Treinamento V10 (Semana 6)

### 6.1 main_v10.py

```python
# main_v10.py

def train_v10(config):
    """
    Pipeline de treinamento Filo-Priori V10.
    """

    # 1. Carregar dados
    data_loader = RTPTorrentLoader(config.data)
    train_data, val_data, test_data = data_loader.load_temporal_split()

    # 2. Construir grafo com decaimento temporal
    graph_builder = TimeDecayGraphBuilder(
        decay_lambda=config.graph.decay_lambda,
        lookback_days=config.graph.lookback_days
    )
    train_graph = graph_builder.build(train_data)

    # 3. Inicializar modelo
    model = FiloPrioriV10(config)

    # 4. LambdaRank Loss
    criterion = LambdaRankLoss(
        sigma=config.training.loss.sigma,
        ndcg_at_k=config.training.loss.ndcg_at_k
    )

    # 5. Otimizador com diferentes LRs
    optimizer = torch.optim.AdamW([
        {'params': model.codebert.parameters(), 'lr': 1e-5},  # Fine-tune lento
        {'params': model.gat.parameters(), 'lr': 1e-4},
        {'params': model.neural_residual.parameters(), 'lr': 1e-3},
        {'params': [model.alpha], 'lr': 1e-2}  # Alpha aprende rápido
    ])

    # 6. Training loop
    for epoch in range(config.training.epochs):
        for batch in train_loader:
            scores, _ = model(
                batch['test_tokens'],
                batch['code_tokens'],
                batch['graph'],
                batch['heuristics']
            )

            loss = criterion(scores, batch['relevances'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validação
        val_apfd = evaluate_apfd(model, val_data)
        print(f"Epoch {epoch}: Val APFD = {val_apfd:.4f}")

        # Early stopping
        if val_apfd > best_apfd:
            save_checkpoint(model, 'best_v10.pt')
            best_apfd = val_apfd
```

### 6.2 Estratégia de Divisão Temporal

```python
# CRÍTICO: Divisão baseada em tempo, NÃO aleatória

def temporal_split(builds, train_ratio=0.7, val_ratio=0.15):
    """
    Divide builds cronologicamente para evitar vazamento de dados.

    Errado: Shuffle split (futuro vazando para passado)
    Certo: Time-series split (sempre treina no passado, testa no futuro)
    """
    builds_sorted = sorted(builds, key=lambda b: b.timestamp)

    n = len(builds_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        'train': builds_sorted[:train_end],
        'val': builds_sorted[train_end:val_end],
        'test': builds_sorted[val_end:]
    }
```

---

## Fase 7: Avaliação e Comparação (Semana 7)

### 7.1 Métricas

| Métrica | Descrição | Fórmula |
|---------|-----------|---------|
| **APFD** | Average Percentage of Faults Detected | $1 - \frac{\sum TF_i}{n \cdot m} + \frac{1}{2n}$ |
| **APFD-c** | APFD com custo (duração) | Ponderado por tempo |
| **NPA** | Normalized Position of First Fault | Posição da primeira falha |
| **HIT@k** | Taxa de acerto nos top-k | % de falhas nos primeiros k |

### 7.2 Baselines para Comparação

```python
baselines = {
    'random': RandomBaseline(),
    'recently_failed': RecentlyFailedBaseline(),
    'failure_rate': FailureRateBaseline(),
    'deeporder': DeepOrderBaseline(),  # Reproduzir do paper
    'filo_v9': FiloPrioriV9()  # Nossa versão anterior
}
```

### 7.3 Testes Estatísticos

```python
from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta

def statistical_comparison(v10_results, baseline_results):
    """
    Comparação estatística rigorosa.

    Para publicação, precisamos:
    1. Wilcoxon signed-rank test (p < 0.05)
    2. Cliff's Delta (effect size > 0.147 = small)
    """

    # Wilcoxon: Hipótese nula = mesma distribuição
    stat, p_value = wilcoxon(v10_results, baseline_results)

    # Cliff's Delta: Magnitude do efeito
    delta, interpretation = cliffs_delta(v10_results, baseline_results)

    return {
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cliffs_delta': delta,
        'effect_size': interpretation  # negligible/small/medium/large
    }
```

---

## Fase 8: Documentação e Publicação (Semana 8)

### 8.1 Experimentos de Ablação

Para o paper, precisamos demonstrar a contribuição de cada componente:

| Experimento | Componentes | Objetivo |
|-------------|-------------|----------|
| V10-Full | CodeBERT + TimeDecay + LambdaRank | Performance completa |
| V10-NoCodeBERT | SBERT + TimeDecay + LambdaRank | Impacto do CodeBERT |
| V10-NoTimeDecay | CodeBERT + Static + LambdaRank | Impacto do decaimento |
| V10-NoLambda | CodeBERT + TimeDecay + FocalLoss | Impacto do LambdaRank |
| V10-NoResidual | Sem heurísticas | Impacto do residual |

### 8.2 Estrutura do Paper

```
1. Introduction
   - Problema: DL falha em superar heurísticas simples
   - Tese: Falha por ignorar temporalidade e sparsity

2. Background
   - RTPTorrent dataset
   - Recently-Failed dominance
   - Limitações de DeepOrder

3. Approach: Filo-Priori V10
   - 3.1 CodeBERT + Co-Attention
   - 3.2 Time-Decay Graph
   - 3.3 LambdaRank
   - 3.4 Residual Learning

4. Experimental Setup
   - Datasets: RTPTorrent (20 projetos)
   - Baselines: Random, Recently-Failed, FailureRate, DeepOrder
   - Metrics: APFD, APFD-c, NPA

5. Results
   - RQ1: V10 vs Baselines
   - RQ2: Ablation Study
   - RQ3: Generalization across projects

6. Discussion
   - Threats to validity
   - Lessons learned

7. Conclusion
```

---

## Cronograma Consolidado

| Semana | Fase | Entregáveis |
|--------|------|-------------|
| 1 | Infraestrutura | Dependências, estrutura, configs |
| 2 | Time-Decay Graph | `time_decay_builder.py`, testes |
| 3 | CodeBERT | `codebert_encoder.py`, `co_attention.py` |
| 4 | LambdaRank | `lambda_rank.py`, `approx_ndcg.py` |
| 5 | Arquitetura | `hybrid_model.py`, `main_v10.py` |
| 6 | Treinamento | Experimentos, tuning |
| 7 | Avaliação | Comparações, estatísticas |
| 8 | Publicação | Paper draft, ablations |

---

## Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| CodeBERT lento demais | Média | Alto | Cache de embeddings em LMDB |
| LambdaRank instável | Baixa | Médio | Fallback para ApproxNDCG |
| Grafo muito denso | Média | Médio | Threshold em co-changes |
| Overfitting no RTPTorrent | Alta | Alto | Cross-project validation |

---

## Métricas de Sucesso

### Critérios de Aceite para Publicação

1. **APFD > Recently-Failed** em pelo menos 15/20 projetos
2. **p-value < 0.05** no teste Wilcoxon
3. **Cliff's Delta > 0.147** (effect size small ou maior)
4. **Ablation** mostrando contribuição de cada componente

### Target Journals

| Journal/Conf | Qualis | Prazo Típico | Foco |
|--------------|--------|--------------|------|
| IEEE TSE | A1 | 6-12 meses | ML em SE |
| ACM TOSEM | A1 | 6-9 meses | Métodos formais |
| ICSE | A1 | Novembro | Conferência top |
| FSE | A1 | Março/Setembro | Conferência top |
| ISSTA | A2 | Janeiro | Testing específico |

---

*Documento criado: Novembro 2025*
*Próxima revisão: Após Fase 2*
