A seguir está uma explicação completa, clara e em Português-BR, sobre o paper “DeepOrder: Deep Learning for Test Case Prioritization in Continuous Integration Testing”, com base no texto e figuras do documento 

[paper] - DeepOrder_Deep_Learni…

.

📘 Explicando o Paper “DeepOrder”

O artigo apresenta DeepOrder, um modelo de deep learning criado para priorização de casos de teste em pipelines de Continuous Integration (CI). Ele resolve limitações importantes de métodos anteriores — principalmente velocidade, escalabilidade e capacidade de usar histórico longo de execuções.

A seguir, explico o paper por seções, de forma didática.

1. Contexto e Motivação

(Seção I – Introdução)

Em pipelines CI, toda mudança no código dispara dezenas ou centenas de testes. Porém:

❗ Nem todos os testes cabem dentro do tempo limitado de cada ciclo CI.

🎯 O objetivo é rodar primeiro os testes com maior chance de falhar, para detectar bugs mais cedo.

Métodos existentes:

Método	Problema
ROCKET	Funciona bem, mas fica extremamente lento com grandes históricos.
RETECS (Reinforcement Learning)	Usa só 4 ciclos de histórico; demora muito para treinar.
Outros ML	Usam poucos dados, não escalam, ou ignoram tempo de execução.

Resultado:
➡️ A priorização fica subótima e lenta.

DeepOrder surge para resolver isso.

2. Definição do Problema

(Seção II)

O objetivo é ordenar casos de teste T de forma que:

Detectem falhas o mais cedo possível

Caibam dentro de um orçamento de tempo

Cada teste tem:

histórico de execuções (pass/fail/not executed)

duração média

momento da última execução

mudanças no comportamento (quantas vezes alternou entre pass→fail)

Os autores definem uma função de prioridade:

𝑝
(
𝑡
𝑖
)
=
∑
𝑗
=
1
𝑚
𝑤
𝑗
⋅
max
⁡
(
𝐸
𝑆
(
𝑖
,
𝑗
)
,
0
)
p(t
i
	​

)=
j=1
∑
m
	​

w
j
	​

⋅max(ES(i,j),0)

onde:

ES(i,j) ∈ {1 = falhou, 0 = passou, -1 = não rodou}

wₖ = peso maior para ciclos recentes

Essa prioridade real é usada como label para treinar o modelo.

3. Como o DeepOrder Funciona

(Seção III, Figuras 1 e 2)

🔎 3.1 Pipeline Geral (Figura 1)

A pipeline é:

Extrair histórico de execuções de CI

Extrair features (statuses, duração, mudanças, timestamp)

Balacear dataset com SMOGN (porque falhas são muito raras)

Treinar uma rede neural

Usar a rede para prever prioridades de testes futuros

🧠 3.2 Arquitetura da Rede Neural (Figura 2)

A rede é simples e eficiente:

Entrada: 14 features

3 camadas escondidas: 10 → 20 → 15 neurônios

Ativação: Mish (melhor que ReLU)

Saída: 1 número real = prioridade

Loss: MSE

Optimizer: Adam

Treino até MSE < 0.0001

O modelo prioriza casos como um regressor, e não como classificador.

4. Datasets e Preparação

(Seção IV)

O DeepOrder foi avaliado em:

Cisco (caso real principal)

ABB Robotics – Paint Control

ABB Robotics – IOF/ROL

Google GSDTSR (12 milhões de execuções)

Problema:
🔴 Proporção de falhas é extremamente baixa
Exemplo (Tabela II):

Cisco: 0.43% de falhas

Google: 0.0025% de falhas

Solução:
✔️ SMOGN para gerar dados sintéticos em regressão (não SMOTE “clássico”)

Isso força o modelo a aprender melhor os casos realmente críticos.

5. Métricas de Avaliação

(Tabela IV)

As métricas principais são:

🔹 APFD

Average Percentage of Faults Detected
→ mede quão cedo as falhas são detectadas

🔹 NAPFD

Versão normalizada, usada para comparações justas

🔹 Métricas de Tempo

Incluem:

FT (First fault time)

LT (Last fault time)

TT (Total runtime do algoritmo)

RT (Tempo para priorizar)

AT (Avg. time to detect all faults)

Essas métricas são críticas porque o objetivo é acelerar CI.

6. Resultados Experimentais

(Seção V, Tabelas e Figuras)

🟦 6.1 RQ1 – DeepOrder detecta mais falhas?

Sim.

Comparação direta com RETECS-N (Figura 4):

DeepOrder vence em 3/4 datasets em NAPFD

É muito mais estável (menos outliers)

RETECS só ganha em um dataset (Paint Control)

🟩 6.2 RQ2 – DeepOrder é mais rápido?

Sim — e MUITO.

Figura 5:
DeepOrder supera ROCKET nas métricas FT, LT, AT.

Figura 6:
Em datasets grandes (Google, com 12 milhões de execuções):

DeepOrder é 70× mais rápido que ROCKET para priorizar

DeepOrder leva 13 s vs. milhares de segundos do ROCKET

Em tempo total (TT), DeepOrder leva 0.5 h, RETECS leva 25 horas

🟨 6.3 RQ3 – Usar histórico maior melhora o resultado?

Sim — e esse é um dos achados mais importantes do paper.

A Figura 7 mostra:

➡️ Usar mais de 4 ciclos melhora APFD e NAPFD em TODOS os datasets.

Isso contradiz a crença do RETECS, que limitava histórico por questões de custo.

🟧 6.4 RQ4 – O modelo prevê prioridades com precisão?

Sim.

Valores de MSE da Tabela VI:

Cisco: 0.000038

Paint Control: 0.000123

IOF/ROL: 0.000001

Google: 0.000031

Muito baixos para regressão.

O gráfico de loss (Figura 9) mostra convergência estável.

7. Discussão e Trabalhos Relacionados

(Seção VI e VII)

Por que o DeepOrder funciona tão bem?

ML supervisionado aprende rápido

Não tem o custo gigantesco de RL

Pode usar histórico ilimitado

Escala bem para milhões de execuções

Tem arquitetura leve (apenas 631 parâmetros)

Limitações:

Não considera flaky tests

Não usa informação do código

Ignora testes não executados

Não lida com mudanças estruturais no projeto

8. Conclusão

(Seção VII)

DeepOrder oferece:

✔️ Melhor detecção de falhas do que RETECS e que práticas da indústria
✔️ Muito mais rápido (até 70×)
✔️ Aproveita histórico longo
✔️ Simples de treinar e implantar
✔️ Funciona com grandes datasets reais (Google, Cisco, ABB)

Ele demonstra que deep learning supervisionado, mesmo simples, é mais eficaz que técnicas mais complexas (reinforcement learning, heurísticas, etc.) para priorização de testes.