Resumo Executivo
O artigo apresenta o NodeRank, uma nova abordagem de priorização de entradas de teste especificamente projetada para Redes Neurais em Grafos (GNNs). O objetivo principal é identificar e priorizar as entradas de teste que têm maior probabilidade de serem classificadas incorretamente pelo modelo, reduzindo assim o alto custo de rotulagem manual de dados.



Problema Abordado

Custo de Rotulagem: A validação de modelos GNN exige rotulagem manual, que é cara e demorada, especialmente dada a complexidade dos dados estruturados em grafos.


Limitação das Técnicas Atuais: As técnicas existentes de priorização de testes para Redes Neurais Profundas (DNNs), como DeepGini e PRIMA, não são adequadas para GNNs. Elas tratam as entradas como independentes e ignoram a interdependência (conexões entre nós e arestas) que é crucial para a inferência em GNNs.


A Solução: NodeRank
O NodeRank é uma abordagem baseada em análise de mutação guiada por ensemble learning (aprendizado de conjunto). A premissa central é que, se uma entrada de teste (nó) for capaz de "matar" muitos modelos mutantes (ou seja, mudar o resultado da predição sob mutação), essa entrada tem maior probabilidade de ser uma falha (bug) e deve ter prioridade alta.


1. Principais Componentes (Regras de Mutação)
O NodeRank introduz três tipos de operadores de mutação adaptados para grafos:



Mutação da Estrutura do Grafo (GSM): Altera a interdependência das entradas introduzindo arestas adicionais aleatórias, modificando as propriedades estruturais.



Mutação de Atributos do Nó (NFM): Perturba os vetores de características (features) dos nós selecionados, influenciando o fluxo de informação no grafo.



Mutação do Modelo GNN (GMM): Altera os parâmetros de treinamento e a arquitetura do modelo GNN (ex: pesos, funções de ativação) para modificar a passagem de mensagens.

2. Processo de Priorização
O NodeRank gera vetores de características baseados nos resultados dessas mutações (se o mutante foi "morto" ou não). Esses vetores alimentam um modelo de ranqueamento baseado em ensemble learning (combinando Regressão Logística, Random Forest, XGBoost, etc.) para prever a probabilidade de erro de classificação de cada entrada.


Diferenciais e Inovações (vs. Estado da Arte)
O artigo compara o NodeRank com o GraphPrior (outra técnica para GNNs) e o PRIMA (estado da arte para DNNs). Os principais diferenciais são:


Consideração da Interdependência: Ao contrário de métodos para DNNs que assumem dados independentes, o NodeRank projeta regras de mutação (especialmente GSM) que manipulam diretamente as conexões entre os nós, refletindo a natureza dos dados em grafo.



Mutações de Entrada e Modelo: Enquanto o GraphPrior foca apenas em mutações do modelo, o NodeRank incorpora mutações na estrutura e nas características da entrada (dados), capturando mais nuances de falhas.


Ensemble Learning: O NodeRank utiliza técnicas de aprendizado de conjunto para combinar múltiplos modelos de ranqueamento, o que provou ser mais eficaz do que o uso de um único modelo de ranqueamento (como feito no GraphPrior e PRIMA).



Mitigação da Aleatoriedade: O estudo adota o método de "matar" mutantes do DeepCrime, que considera a aleatoriedade estocástica do treinamento de redes neurais, garantindo que as mudanças na predição sejam devido à mutação e não ao acaso.


Resultados e Contribuições
Os autores realizaram uma avaliação em larga escala com 124 cenários (pares de modelos e datasets), incluindo dados naturais e dados sob ataques adversariais.


Eficácia Superior: O NodeRank superou todas as abordagens comparadas (DeepGini, GraphPrior, Random, etc.) nas métricas APFD (Average Percentage of Fault-Detection) e PFD.

Ganho de Performance:

Em datasets originais (naturais), obteve uma melhoria média entre 4,41% e 58,11%.

Em datasets adversariais, a melhoria variou entre 4,96% e 62,15%.


Robustez: O método demonstrou ser eficaz mesmo sob ataques adversariais, onde métodos baseados em confiança (confidence-based) geralmente falham.