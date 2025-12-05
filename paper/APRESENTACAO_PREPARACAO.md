# Preparacao para Apresentacao: Filo-Priori

## Documento de Apoio para Defesa e Perguntas

---

# PARTE 1: As 4 Contribuicoes Principais (Explicacao Detalhada)

## Contribuicao 1: Arquitetura Dual-Stream com GAT e SBERT

### O que e?
Uma arquitetura neural de duas vias (streams) que processa informacoes complementares:
- **Stream Semantico**: Processa descricoes textuais dos casos de teste e mensagens de commit usando SBERT (Sentence-BERT)
- **Stream Estrutural**: Processa padroes historicos de execucao usando Graph Attention Networks (GAT)

### Por que e inovador?
Abordagens existentes tratam casos de teste como entidades independentes. Nossa arquitetura captura:
1. **O QUE os testes fazem** (semantica via SBERT)
2. **COMO os testes se comportam** (estrutura via GAT)

### Detalhes Tecnicos
```
Stream Semantico:
- Input: Embeddings SBERT 768-dim (teste) + 768-dim (commit) = 1536-dim
- Arquitetura: FFN com conexoes residuais
- Output: Representacao semantica 256-dim

Stream Estrutural:
- Input: 10 features estruturais por teste
- Arquitetura: 2 camadas GAT (4 heads -> 1 head)
- Output: Representacao estrutural 256-dim

Fusao:
- Cross-Attention bidirecional (4 heads)
- Concatenacao: 512-dim
- MLP Classificador: [128, 64] -> 2 classes
```

### Numeros Importantes
- SBERT model: `all-mpnet-base-v2`
- Dimensao dos embeddings: 768
- Numero de heads no GAT: 4 (layer 1), 1 (layer 2)
- Dropout: 0.3-0.4

---

## Contribuicao 2: Grafo Multi-Edge de Relacionamento entre Testes

### O que e?
Um grafo onde nos representam casos de teste e arestas codificam 5 tipos de relacionamentos:

| Tipo de Aresta | Peso | Formula |
|----------------|------|---------|
| Co-Falha | 1.0 | min(P(tj|ti), P(ti|tj)) |
| Co-Sucesso | 0.5 | min(P(tj|ti), P(ti|tj)) |
| Componente | 0.4 | Jaccard(Ci, Cj) |
| Semantico | 0.3 | cos(ei, ej) > 0.65 |
| Temporal | 0.2 | count_adj / max(count) |

### Por que e inovador?
- Grafos tradicionais usam apenas co-falha (densidade ~0.02%)
- Nosso grafo multi-edge aumenta densidade para 0.5-1.0%
- **77.4% dos testes conectados** (vs 50-60% com abordagens simples)

### Impacto Comprovado
- **+10.0% APFD** (maior contribuicao individual no ablation study)
- Permite propagacao de sinais de falha entre testes relacionados
- Resolve o problema de testes "orfaos" isolados

### Parametros Chave
- Threshold semantico: 0.65 (relaxado de 0.75)
- Top-k vizinhos: 10 (aumentado de 5)
- Combinacao de pesos: soma ponderada normalizada

---

## Contribuicao 3: Engenharia de Features Discriminativas

### O que e?
Identificacao de 10 features estruturais mais discriminativas a partir de um conjunto inicial de 29, atraves de analise de importancia e filtragem de correlacao.

### Features Base (10 selecionadas)
1. **test_age**: Builds desde primeira aparicao
2. **failure_rate**: Taxa historica de falha
3. **recent_failure_rate**: Taxa de falha nos ultimos 5 builds
4. **flakiness_rate**: Frequencia de oscilacao pass/fail
5. **consecutive_failures**: Sequencia atual de falhas
6. **max_consecutive_failures**: Maior sequencia observada
7. **failure_trend**: Diferenca entre taxa recente e historica
8. **commit_count**: Numero de commits relacionados
9. **cr_count**: Numero de code reviews
10. **test_novelty**: Novidade do teste

### Features Temporais (DeepOrder-inspired, 9 adicionais)
- execution_status_last_[1,2,3,5,10]
- cycles_since_last_fail
- distance
- status_changes
- fail_rate_last_10

### Processo de Selecao
1. Feature Importance (Random Forest)
2. Correlacao de Pearson (remover redundancia)
3. Validacao cruzada temporal
4. Reducao de 29 para 19 features efetivas

### Impacto
- **+2.0% APFD** pela adicao de features DeepOrder
- Features temporais capturam padroes que metricas estaticas perdem

---

## Contribuicao 4: Validacao Empirica em 3 Dominios

### Dominio 1: Dataset Industrial (QTA)
| Metrica | Valor |
|---------|-------|
| APFD | **0.7595** |
| vs NodeRank | **+14.7%** (p < 0.001) |
| vs DeepOrder | **+9.8%** (p < 0.001) |
| vs Random | **+51.9%** |

**Caracteristicas do Dataset:**
- 52,102 execucoes de teste
- 1,339 builds (277 com falhas)
- 2,347 casos de teste unicos
- Ratio Pass:Fail = 37:1

### Dominio 2: Benchmarks GNN
| Dataset | APFD | vs NodeRank |
|---------|------|-------------|
| Cora | 0.861 | **+3.4%** |
| CiteSeer | 0.742 | **+1.3%** |
| PubMed | 0.785 | -0.6% |
| **Media** | **0.796** | **Vence 2/3** |

### Dominio 3: RTPTorrent (Open-Source)
| Metrica | Valor |
|---------|-------|
| APFD | **0.8376** |
| Projetos | 20 Java projects |
| Builds | 1,250 |
| vs recently_failed | **+2.02%** |
| Rank | **#2** (apenas oracle melhor) |

### Por que 3 Dominios?
1. **Diversidade**: Industrial, academico, open-source
2. **Reproducibilidade**: Datasets publicos (GNN, RTPTorrent)
3. **Generalizacao**: Prova que nao e overfitting a um dominio

---

# PARTE 2: Perguntas e Respostas Esperadas

## Categoria A: Perguntas sobre Arquitetura e Metodologia

### P1: Por que usar Graph Attention Networks em vez de GCN ou GraphSAGE?

**Resposta:**
GAT oferece duas vantagens criticas:

1. **Atencao Dinamica (GATv2)**: Diferente do GAT original que computa atencao "estatica", usamos GATv2 que aplica a nao-linearidade apos a transformacao linear:
   ```
   GAT original: alpha_ij = softmax(LeakyReLU(a^T [Whi || Whj]))
   GATv2:        alpha_ij = softmax(a^T LeakyReLU(W[hi || hj]))
   ```
   Isso permite que o ranking de atencao dependa de AMBOS os nos (query e key).

2. **Pesos de Aresta**: GAT permite incorporar os pesos das arestas do grafo multi-edge, permitindo que o modelo aprenda que co-falha (peso 1.0) e mais importante que similaridade semantica (peso 0.3).

GCN nao tem mecanismo de atencao e GraphSAGE usa amostragem que pode perder conexoes importantes.

---

### P2: Por que Cross-Attention em vez de simples concatenacao?

**Resposta:**
Concatenacao simples trata as duas modalidades como independentes. Cross-Attention permite:

1. **Fusao Adaptativa**: Cada modalidade pode "perguntar" para a outra o que e relevante
2. **Bidirecionalidade**: Semantica informa estrutura E estrutura informa semantica
3. **Pesos Aprendidos**: O modelo aprende QUANDO cada modalidade e mais importante

**Exemplo Pratico**:
- Teste novo (sem historico) -> Cross-attention da mais peso ao stream semantico
- Teste com historico rico -> Cross-attention pode focar no stream estrutural

O ablation study mostra que isso contribui para a performance superior.

---

### P3: Como voces lidam com o desbalanceamento de classes (37:1)?

**Resposta:**
Descobrimos que **multiplos mecanismos causam sobre-correcao**. Trabalhos anteriores usavam:
- Class weights (19x)
- Focal alpha (1.7x)
- Balanced sampling (20x)
- **Total: ~646x de compensacao** -> Mode collapse!

Nossa solucao: **Mecanismo Unico**
- Apenas Balanced Sampling (15:1)
- Focal alpha = 0.5 (neutro)
- Class weights = desabilitado

**Resultado**: +4.0% APFD e treinamento estavel.

**Threshold Optimization**: Buscamos o threshold otimo [0.05, 0.90] usando F_beta (beta=0.8). Encontramos threshold = 0.28 (vs default 0.50), melhorando recall da classe minoritaria de 5% para 30%.

---

### P4: O que sao "testes orfaos" e como voces os tratam?

**Resposta:**
**Testes Orfaos** = testes ausentes do grafo de treinamento (22.6% do total).

**Problema**: GAT retorna scores uniformes para nos fora do grafo, destruindo a capacidade de ranking.

**Solucao: Pipeline KNN de 4 Estagios**

1. **KNN Similaridade**: Distancia Euclidiana para k=20 vizinhos mais proximos
2. **Blend Estrutural**: Combina similaridade semantica com estrutural (w=0.35)
3. **Softmax com Temperatura**: T=0.7 para concentrar pesos nos vizinhos mais similares
4. **Alpha Blending**: Combina score KNN com score base (alpha=0.55)

**Impacto**:
- Antes: scores orfaos com variancia ZERO (todos 0.50)
- Depois: variancia = 0.046, range [0.29, 0.51]
- **+5.9% APFD**

---

### P5: Por que usar SBERT e nao CodeBERT ou outro modelo de codigo?

**Resposta:**
1. **Natureza dos Dados**: Nossas descricoes de teste sao texto natural (TC_Summary, TC_Steps), nao codigo-fonte. SBERT e otimizado para similaridade de sentencas.

2. **all-mpnet-base-v2**: Modelo SOTA para sentence embeddings com balance entre qualidade e velocidade.

3. **Commit Messages**: Tambem sao texto natural, nao codigo.

4. **Resultados Empiricos**: SBERT funcionou bem nos experimentos. CodeBERT seria mais apropriado se estivessemos analisando o codigo dos testes em si.

**Trabalho Futuro**: Investigar embeddings de codigo (CodeBERT, GraphCodeBERT) para capturar similaridade a nivel de implementacao.

---

## Categoria B: Perguntas sobre Resultados e Avaliacao

### P6: Por que a comparacao com NodeRank e justa se ele foi projetado para teste de GNNs?

**Resposta:**
Excelente pergunta! Fazemos duas avaliacoes:

1. **Dataset Industrial**: Adaptamos NodeRank para TCP tradicional usando as mesmas features e comparamos justamente. NodeRank usa ensemble learning com mutacoes - nos mantemos a ideia de ranking mas com nossa arquitetura.

2. **Benchmarks GNN (Cora, CiteSeer, PubMed)**: Aqui comparamos NO DOMINIO ORIGINAL do NodeRank. Usamos exatamente o mesmo protocolo experimental:
   - Treinamos GCN para classificacao de nos
   - Tratamos nos mal-classificados como "falhas"
   - Priorizamos para detectar essas falhas primeiro

**Resultado**: Vencemos NodeRank em 2/3 datasets (Cora +3.4%, CiteSeer +1.3%), validando que nossa metodologia e competitiva MESMO no dominio para o qual NodeRank foi projetado.

---

### P7: O que significa o efeito "small" no Cliff's delta?

**Resposta:**
Cliff's delta mede o tamanho do efeito (effect size):
- |d| < 0.147: negligivel
- 0.147 <= |d| < 0.33: **small**
- 0.33 <= |d| < 0.474: medium
- |d| >= 0.474: large

Nosso d = 0.179 (small) indica:
- Diferenca estatisticamente significativa (p < 0.001)
- Melhoria consistente mas nao dramatica
- Esperado em TCP onde baselines ja sao razoaveis

**Contexto Importante**: Em TCP, melhorias de 10-15% sao consideradas substanciais porque o baseline (failure_rate) ja captura muita informacao.

---

### P8: Por que o RTPTorrent mostra APFD mais alto que o dataset industrial?

**Resposta:**
Varias razoes:

1. **Caracteristicas dos Projetos**: Projetos open-source no RTPTorrent tem padroes de falha mais previsiveis (testes bem estruturados, CI maduro).

2. **Adaptacao do Modelo**: Usamos LightGBM LambdaRank no RTPTorrent (otimizado especificamente para ranking), enquanto no industrial usamos a arquitetura neural completa.

3. **Metricas Comparaveis**: O baseline "recently_failed" atinge 0.82 no RTPTorrent vs ~0.66 no industrial, indicando que o problema e intrinsecamente "mais facil".

4. **Variabilidade**: Alguns projetos RTPTorrent chegam a APFD=0.99 (apache/sling), enquanto outros ficam em 0.29. A media de 0.84 inclui essa variabilidade.

---

### P9: O ablation study mostra que o grafo denso e o mais importante (+10%). Isso nao significa que o resto e dispensavel?

**Resposta:**
Nao, porque os componentes sao **aditivos e complementares**:

| Componente | Contribuicao | Cumulativo |
|------------|--------------|------------|
| Dense Multi-Edge Graph | +10.0% | 10.0% |
| Orphan KNN Scoring | +5.9% | 15.9% |
| Single Balancing | +4.0% | 19.9% |
| DeepOrder Features | +2.0% | 21.9% |
| Threshold Optimization | +1.0% | 22.9% |

**Total**: ~16.8% sobre baseline (alguns efeitos nao sao perfeitamente aditivos)

Cada componente resolve um problema especifico:
- Grafo denso: conectividade
- Orphan KNN: testes novos
- Single Balancing: estabilidade de treinamento
- Features temporais: padroes recentes
- Threshold: calibracao de decisao

---

### P10: Como voces garantem que nao ha data leakage temporal?

**Resposta:**
Implementamos **separacao temporal estrita**:

1. **Split Cronologico**: Dados de treino SEMPRE anteriores aos de teste
2. **Temporal Cross-Validation**:
   - 5-Fold temporal
   - Sliding window
   - Concept drift test
3. **Resultados Consistentes**: APFD varia apenas 3% entre periodos (0.73-0.78)

**Processo de Construcao do Grafo**:
- Co-falha calculada apenas com historico anterior
- Features temporais (cycles_since_last_fail) usam apenas informacao passada
- Embeddings SBERT sao estaticos (nao vazam informacao temporal)

---

## Categoria C: Perguntas sobre Limitacoes e Trabalhos Futuros

### P11: Quais sao as principais limitacoes do trabalho?

**Resposta:**
1. **Cold Start**: Testes novos sem similaridade semantica nao se beneficiam do grafo. Mitigamos com KNN, mas ainda e uma limitacao.

2. **Especificidade de Dominio**: Testamos em 3 dominios, mas pode nao generalizar para todos os contextos (ex: sistemas embarcados, jogos).

3. **Custo Computacional**:
   - Treinamento: 2-3 horas (GPU)
   - Construcao do grafo: overhead de preprocessamento
   - Inferencia: <1 segundo (aceitavel para CI)

4. **Interpretabilidade**: Modelo e caixa-preta. Ablation study ajuda, mas predicoes individuais sao dificeis de explicar.

5. **Escala**: Dataset industrial tem 277 builds com falhas. Projetos Google-scale podem precisar de otimizacoes adicionais.

---

### P12: Como este trabalho pode ser aplicado na industria?

**Resposta:**
**Integracao com CI/CD Pipeline:**

```
1. Desenvolvedor faz commit
2. CI trigger dispara
3. Filo-Priori e chamado:
   a. Carrega modelo treinado
   b. Extrai features do commit (SBERT)
   c. Consulta historico de testes
   d. Gera ranking de priorizacao
4. Testes executados na ordem priorizada
5. Falhas detectadas mais cedo
6. Feedback mais rapido para o desenvolvedor
```

**Beneficios Praticos:**
- 51.9% mais rapido para detectar falhas vs random
- Inferencia <1s por build
- Retreinamento periodico (semanal/mensal)

**Requisitos Minimos:**
- 50+ builds historicos
- Descricoes de teste (opcional, melhora semantica)
- Logs de execucao (pass/fail por build)

---

### P13: Quais sao os proximos passos / trabalhos futuros?

**Resposta:**
1. **Construcao Dinamica de Grafo**: Atualmente o grafo e estatico. Queremos grafos que evoluem com o tempo, capturando mudancas nos padroes de relacionamento.

2. **Transfer Learning Cross-Project**: Pre-treinar em multiplos projetos e fazer fine-tuning em projetos novos com pouco historico.

3. **Priorizacao Real-Time**: Otimizar para decisoes online durante a execucao, permitindo re-priorizacao dinamica.

4. **Embeddings de Codigo**: Investigar CodeBERT, GraphCodeBERT para capturar similaridade a nivel de implementacao dos testes.

5. **Explicabilidade**: Desenvolver mecanismos para explicar POR QUE um teste foi priorizado (attention visualization, SHAP values).

---

## Categoria D: Perguntas Desafiadoras/Criticas

### P14: Se FailureRate simples atinge APFD similar, por que usar deep learning?

**Resposta:**
Pergunta justa! FailureRate atinge APFD ~0.75 (1.4% abaixo de nos). Vantagens do Filo-Priori:

1. **Testes Novos**: FailureRate nao tem informacao. Nos usamos similaridade semantica.

2. **Padroes Complexos**: GNN captura dependencias multi-hop que heuristicas simples nao veem.

3. **Adaptabilidade**: Modelo aprende padroes especificos do projeto automaticamente.

4. **Escalabilidade**: Uma vez treinado, lida com milhares de testes eficientemente.

**Recomendacao Pratica**:
- Projetos com padroes estaveis e simples -> FailureRate pode ser suficiente
- Projetos com evolucao rapida, muitos testes novos -> Filo-Priori

---

### P15: O dataset industrial e proprietario. Como garantir reproducibilidade?

**Resposta:**
Tomamos varias medidas:

1. **Datasets Publicos**: Avaliamos em Cora, CiteSeer, PubMed (GNN) e RTPTorrent (open-source). Resultados sao reproduziveis.

2. **Replication Package**: Fornecemos:
   - Codigo fonte completo
   - Configuracoes exatas
   - Modelos treinados
   - Dataset anonimizado (estrutura preservada)
   - Scripts de reproducao

3. **Transparencia Metodologica**: Descrevemos todos os detalhes de implementacao, hiperparametros e processo de selecao.

4. **Resultados Consistentes**: Performance similar nos 3 dominios sugere que resultados nao sao overfitting ao dataset industrial.

---

### P16: Por que nao comparar com abordagens de Reinforcement Learning como RETECS?

**Resposta:**
1. **Complexidade de Implementacao**: RETECS requer ambiente de simulacao e reward shaping cuidadoso.

2. **Foco do Trabalho**: Nossa contribuicao principal e a arquitetura dual-stream com grafos, nao a comparacao exaustiva de paradigmas de ML.

3. **Resultados Existentes**: DeepOrder (que comparamos) representa o SOTA em deep learning para TCP. NodeRank representa SOTA em abordagens baseadas em grafos.

4. **Trabalho Futuro**: Combinacao de GAT com RL para priorizacao adaptativa online e uma direcao interessante.

---

### P17: O ratio 37:1 e muito extremo. Isso e representativo de projetos reais?

**Resposta:**
Sim, e ate conservador!

**Evidencia da Literatura:**
- Google: 99%+ dos testes passam em cada build
- Microsoft: ratios de 50:1 a 100:1 sao comuns
- Estudos empiricos: ratios de 20:1 a 100:1

**Por que e comum:**
- Testes de regressao: maioria passa
- CI frequente: builds pequenos, poucas falhas
- Qualidade de codigo: desenvolvedores corrigem rapidamente

**Nossa Contribuicao**: Tecnica de Single Balancing especificamente projetada para ratios extremos, evitando mode collapse.

---

### P18: GATv2 e melhor que GAT original. Por que nao usar arquiteturas mais recentes como Graph Transformers?

**Resposta:**
1. **Trade-off Complexidade/Performance**: GATv2 ja fornece atencao dinamica suficiente para nosso problema. Graph Transformers sao mais pesados computacionalmente.

2. **Tamanho do Grafo**: Nosso grafo tem ~2,300 nos. Graph Transformers brilham em grafos muito grandes onde atencao global e necessaria.

3. **Resultados Empiricos**: Experimentos mostraram que 1-layer GATv2 supera arquiteturas mais profundas. Isso sugere que relacionamentos de 1-hop sao suficientes.

4. **Trabalho Futuro**: Investigar Graph Transformers para projetos maiores e uma direcao valida.

---

### P19: Como o modelo se comporta quando padroes de falha mudam drasticamente (concept drift)?

**Resposta:**
Testamos explicitamente com **Concept Drift Test** (RQ3):

| Validacao | APFD |
|-----------|------|
| Temporal 5-Fold CV | 0.7821 |
| Sliding Window CV | 0.7412 |
| Concept Drift Test | 0.7298 |

**Degradacao de apenas 3%** indica robustez razoavel.

**Estrategias de Mitigacao:**
1. **Retreinamento Periodico**: Atualizar modelo semanalmente/mensalmente
2. **Sliding Window**: Usar apenas historico recente para construcao do grafo
3. **Features Temporais**: execution_status_last_N captura mudancas recentes

**Limitacao Reconhecida**: Mudancas muito abruptas (refatoracao massiva) podem degradar performance temporariamente.

---

### P20: O que acontece se a descricao dos testes for de baixa qualidade ou inexistente?

**Resposta:**
O sistema e robusto a isso:

1. **Stream Estrutural Continua Funcionando**: Features historicas (failure_rate, flakiness, etc.) nao dependem de texto.

2. **Embeddings Degradados Graciosamente**: SBERT ainda gera embeddings, mesmo para texto curto/pobre. Similaridade pode ser menos precisa, mas nao quebra.

3. **Evidencia no RTPTorrent**: Dataset tem apenas nomes de teste (sem descricoes ricas). Ainda atingimos APFD=0.84.

4. **Cross-Attention Adaptativo**: Se semantica e pouco informativa, o modelo aprende a dar mais peso ao stream estrutural.

**Recomendacao**: Projetos com boas descricoes de teste se beneficiam mais. Mas a abordagem funciona mesmo sem texto rico.

---

# PARTE 3: Resumo dos Numeros Chave

## Performance
| Dataset | APFD | Melhoria |
|---------|------|----------|
| Industrial | 0.7595 | +14.7% vs NodeRank |
| GNN Benchmarks | 0.796 | Vence 2/3 datasets |
| RTPTorrent | 0.8376 | +2.02% vs recently_failed |

## Ablation Study
| Componente | Impacto |
|------------|---------|
| Dense Multi-Edge Graph | +10.0% |
| Orphan KNN Scoring | +5.9% |
| Single Balancing | +4.0% |
| DeepOrder Features | +2.0% |
| Threshold Optimization | +1.0% |

## Dataset Industrial
- 52,102 execucoes
- 1,339 builds (277 com falhas)
- 2,347 testes unicos
- Ratio 37:1 (Pass:Fail)

## Arquitetura
- SBERT: all-mpnet-base-v2 (768-dim)
- GAT: 2 layers, 4 heads
- Cross-Attention: 4 heads
- Threshold otimo: 0.28

---

# PARTE 4: Checklist Pre-Apresentacao

- [ ] Revisar as 4 contribuicoes principais
- [ ] Memorizar numeros chave (APFD, melhorias percentuais)
- [ ] Praticar explicacao do grafo multi-edge
- [ ] Entender trade-offs de cada decisao de design
- [ ] Preparar resposta para "por que nao X em vez de Y"
- [ ] Conhecer limitacoes e trabalhos futuros
- [ ] Ter exemplos praticos prontos

---

*Documento gerado para apoio a apresentacao do paper Filo-Priori*
*Grupo de Pesquisa - IComp/UFAM*
