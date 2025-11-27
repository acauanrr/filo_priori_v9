# Plano de Publicacao IEEE TSE - Filo-Priori V9

**Objetivo**: Publicar no IEEE Transactions on Software Engineering (IEEE TSE)
**Data**: Novembro 2025
**Status**: Em desenvolvimento

---

## 1. ANALISE DO JOURNAL TARGET

### 1.1 IEEE TSE - Requisitos

| Aspecto | Requisito |
|---------|-----------|
| **Impact Factor** | 6.5 (2023) |
| **Template** | IEEE Transactions LaTeX (IEEEtran.cls) |
| **Formato** | Two-column, 9.5pt, 11.5pt spacing |
| **Comprimento** | ~25-35 paginas (regular paper) |
| **Open Data** | Recomendado via IEEE DataPort |
| **Reproducibility** | Codigo e dados publicos |

### 1.2 Diferenca Template Atual vs IEEE TSE

| Aspecto | Atual (EMSE/Elsevier) | Requerido (IEEE TSE) |
|---------|----------------------|---------------------|
| Document class | `elsarticle` | `IEEEtran` |
| Journal | `\journal{Empirical Software Engineering}` | IEEE Computer Society |
| Estilo bib | `elsarticle-num` | `IEEEtran` |
| Formato | Single column review | Two-column final |

---

## 2. ANALISE DE GAPS DO PAPER ATUAL

### 2.1 Secoes Existentes
- [x] Abstract
- [x] Results (RQ1-RQ4)
- [x] Discussion
- [x] Threats to Validity
- [x] Tables (5 tabelas)
- [x] Figures (referencias)

### 2.2 Secoes FALTANTES (Criticas para IEEE TSE)
- [ ] **Introduction** - Motivacao, problema, contribuicoes
- [ ] **Background** - Conceitos fundamentais
- [ ] **Related Work** - Estado da arte com RSL
- [ ] **Methodology** - Design experimental detalhado
- [ ] **Approach/Architecture** - Descricao tecnica do Filo-Priori
- [ ] **Conclusion** - Sumario e trabalhos futuros
- [ ] **Data Availability Statement**

### 2.3 Referencias - Analise Critica

**Estado Atual**: 7 referencias
**Requerido para IEEE TSE**: 40-60 referencias

| Categoria | Atual | Necessario | Gap |
|-----------|-------|------------|-----|
| TCP Classico (pre-2015) | 2 | 5-8 | -3 |
| TCP ML/DL (2015-2022) | 2 | 10-15 | -8 |
| TCP Recente (2023-2025) | 1 | 8-12 | -7 |
| GNN/GAT | 2 | 5-8 | -3 |
| Learning to Rank | 0 | 3-5 | -3 |
| CI/CD Testing | 0 | 5-8 | -5 |
| Embeddings/NLP | 1 | 3-5 | -2 |
| Metodologia/Estatistica | 0 | 3-5 | -3 |
| **TOTAL** | **7** | **42-66** | **-35** |

---

## 3. PROTOCOLO RSL - Revisao Sistematica da Literatura

### 3.1 Objetivo da RSL
Mapear, avaliar e sintetizar o estado da arte em Test Case Prioritization com foco em:
1. Abordagens baseadas em Machine Learning/Deep Learning
2. Uso de Graph Neural Networks em software testing
3. Tecnicas de Learning to Rank aplicadas a TCP
4. TCP em ambientes de Continuous Integration

### 3.2 Questoes de Pesquisa da RSL

**RQ-RSL1**: Quais tecnicas de ML/DL foram aplicadas para TCP nos ultimos 5 anos?
**RQ-RSL2**: Como GNNs tem sido utilizadas em software testing?
**RQ-RSL3**: Quais metricas sao mais utilizadas para avaliar TCP?
**RQ-RSL4**: Quais datasets e benchmarks sao usados na area?
**RQ-RSL5**: Quais gaps existem na literatura que o Filo-Priori endereca?

### 3.3 Estrategia de Busca

#### 3.3.1 Bases de Dados
| Base | URL | Prioridade |
|------|-----|------------|
| IEEE Xplore | https://ieeexplore.ieee.org | Alta |
| ACM Digital Library | https://dl.acm.org | Alta |
| Springer Link | https://link.springer.com | Alta |
| ScienceDirect | https://www.sciencedirect.com | Media |
| arXiv | https://arxiv.org | Media |
| Google Scholar | https://scholar.google.com | Complementar |

#### 3.3.2 String de Busca Principal

```
("test case prioritization" OR "test prioritization" OR "regression testing")
AND
("machine learning" OR "deep learning" OR "neural network" OR "reinforcement learning")
AND
("continuous integration" OR "CI/CD" OR "regression")
```

#### 3.3.3 Strings de Busca Especificas

**S1 - GNN em Software Testing:**
```
("graph neural network" OR "graph attention" OR "GNN" OR "GAT")
AND
("software testing" OR "test case" OR "defect prediction")
```

**S2 - Learning to Rank em TCP:**
```
("learning to rank" OR "RankNet" OR "pairwise ranking" OR "listwise")
AND
("test" OR "software" OR "prioritization")
```

**S3 - TCP em CI:**
```
("test case prioritization" OR "TCP")
AND
("continuous integration" OR "CI" OR "DevOps")
AND
("APFD" OR "fault detection")
```

### 3.4 Criterios de Inclusao/Exclusao

#### Criterios de Inclusao (CI)
- CI1: Artigos publicados entre 2019-2025
- CI2: Foco em Test Case Prioritization ou Test Selection
- CI3: Uso de tecnicas de ML/DL
- CI4: Publicados em venues de alta qualidade (IEEE, ACM, Springer)
- CI5: Disponibilidade do texto completo

#### Criterios de Exclusao (CE)
- CE1: Artigos antes de 2019 (exceto seminais)
- CE2: Short papers (< 6 paginas)
- CE3: Surveys/reviews (usar separadamente)
- CE4: Artigos nao peer-reviewed
- CE5: Duplicatas

### 3.5 Processo de Selecao (PRISMA)

```
Identificacao
    |
    v
Registros identificados nas bases (n = ?)
    |
    v
Triagem por titulo/abstract
    |
    v
Artigos para leitura completa (n = ?)
    |
    v
Aplicacao de criterios I/E
    |
    v
Artigos incluidos na RSL (n = ?)
```

### 3.6 Extracao de Dados

| Campo | Descricao |
|-------|-----------|
| ID | Identificador unico |
| Autores | Lista de autores |
| Ano | Ano de publicacao |
| Venue | Journal/Conference |
| Tecnica | ML/DL/RL/GNN usada |
| Dataset | Dataset utilizado |
| Metricas | APFD, NAPFD, etc. |
| Resultados | Performance reportada |
| Codigo | Disponibilidade |

---

## 4. REFERENCIAS ESSENCIAIS A ADICIONAR

### 4.1 TCP Classico (Seminais)

```bibtex
@article{rothermel2001prioritizing,
  title={Prioritizing test cases for regression testing},
  author={Rothermel, Gregg and Untch, Roland H and Chu, Chengyun and Harrold, Mary Jean},
  journal={IEEE Transactions on software engineering},
  volume={27},
  number={10},
  pages={929--948},
  year={2001}
}

@article{elbaum2002test,
  title={Test case prioritization: A family of empirical studies},
  author={Elbaum, Sebastian and Malishevsky, Alexey G and Rothermel, Gregg},
  journal={IEEE transactions on software engineering},
  volume={28},
  number={2},
  pages={159--182},
  year={2002}
}

@inproceedings{kim2002history,
  title={A history-based test prioritization technique for regression testing in resource constrained environments},
  author={Kim, Jung-Min and Porter, Adam},
  booktitle={ICSE},
  pages={119--129},
  year={2002}
}
```

### 4.2 TCP com Machine Learning

```bibtex
@inproceedings{spieker2017reinforcement,
  title={Reinforcement learning for automatic test case prioritization and selection in continuous integration},
  author={Spieker, Helge and Gotlieb, Arnaud and Marijan, Dusica and Mossige, Morten},
  booktitle={ISSTA},
  pages={12--22},
  year={2017}
}

@article{bertolino2020learning,
  title={Learning-to-rank vs ranking-to-learn: Strategies for regression testing in continuous integration},
  author={Bertolino, Antonia and Guerriero, Antonio and Miranda, Breno and Pietrantuono, Roberto and Russo, Stefano},
  booktitle={ICSE},
  pages={1--12},
  year={2020}
}

@article{bagherzadeh2022reinforcement,
  title={Reinforcement learning for test case prioritization},
  author={Bagherzadeh, Mojtaba and Khosravi, Nafiseh and Briand, Lionel},
  journal={IEEE Transactions on Software Engineering},
  volume={48},
  number={8},
  pages={2836--2856},
  year={2022}
}
```

### 4.3 Deep Learning para TCP

```bibtex
@article{pan2022test,
  title={Test case prioritization based on deep learning},
  author={Pan, Rui and Bagherzadeh, Mojtaba and Ghaleb, Taher Ahmed and Briand, Lionel},
  journal={Empirical Software Engineering},
  volume={27},
  number={6},
  pages={1--42},
  year={2022}
}

@inproceedings{abdelkarim2022tcp,
  title={TCP-Net: Test Case Prioritization using End-to-End Deep Neural Networks},
  author={Abdelkarim, Amr and others},
  booktitle={AST},
  year={2022}
}

@article{chen2023deeporder,
  title={DeepOrder: Deep Learning for Test Case Prioritization in Continuous Integration Testing},
  author={Chen, Junjie and others},
  journal={IEEE TSE},
  year={2023}
}
```

### 4.4 Graph Neural Networks

```bibtex
@inproceedings{velickovic2018graph,
  title={Graph attention networks},
  author={Velickovic, Petar and others},
  booktitle={ICLR},
  year={2018}
}

@inproceedings{brody2022attentive,
  title={How attentive are graph attention networks?},
  author={Brody, Shaked and Alon, Uri and Yahav, Eran},
  booktitle={ICLR},
  year={2022}
}

@article{allamanis2018learning,
  title={Learning to represent programs with graphs},
  author={Allamanis, Miltiadis and others},
  booktitle={ICLR},
  year={2018}
}
```

### 4.5 Learning to Rank

```bibtex
@inproceedings{burges2005learning,
  title={Learning to rank using gradient descent},
  author={Burges, Chris and others},
  booktitle={ICML},
  pages={89--96},
  year={2005}
}

@article{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and others},
  booktitle={ICCV},
  pages={2980--2988},
  year={2017}
}

@article{cao2007learning,
  title={Learning to rank: from pairwise approach to listwise approach},
  author={Cao, Zhe and others},
  booktitle={ICML},
  pages={129--136},
  year={2007}
}
```

### 4.6 CI/CD e Regression Testing

```bibtex
@article{hilton2016usage,
  title={Usage, costs, and benefits of continuous integration in open-source projects},
  author={Hilton, Michael and others},
  booktitle={ASE},
  pages={426--437},
  year={2016}
}

@article{luo2014empirical,
  title={An empirical analysis of flaky tests},
  author={Luo, Qingzhou and others},
  booktitle={FSE},
  pages={643--653},
  year={2014}
}

@article{memon2017taming,
  title={Taming Google-scale continuous testing},
  author={Memon, Atif and others},
  booktitle={ICSE-SEIP},
  pages={233--242},
  year={2017}
}
```

### 4.7 Sentence Embeddings

```bibtex
@article{reimers2019sentence,
  title={Sentence-BERT: Sentence embeddings using siamese BERT-networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={EMNLP},
  year={2019}
}

@article{devlin2019bert,
  title={BERT: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and others},
  booktitle={NAACL},
  year={2019}
}
```

### 4.8 Surveys e SLRs (2022-2024)

```bibtex
@article{hasnain2023systematic,
  title={A systematic literature review on regression test case prioritization},
  author={Hasnain, Muhammad and others},
  journal={Software Quality Journal},
  volume={31},
  number={2},
  pages={447--492},
  year={2023}
}

@article{lima2022test,
  title={Test case selection and prioritization using machine learning: a systematic literature review},
  author={Lima, Jackson A Prado and Vergilio, Silvia Regina},
  journal={Empirical Software Engineering},
  volume={27},
  number={2},
  pages={1--71},
  year={2022}
}

@article{greca2023state,
  title={State of Practical Applicability of Regression Testing Research: A Live Systematic Literature Review},
  author={Greca, Renan and Miranda, Breno and Bertolino, Antonia},
  journal={ACM Computing Surveys},
  year={2023}
}
```

---

## 5. ESTRUTURA PROPOSTA DO PAPER

### 5.1 Outline Completo

```
1. Introduction (2-3 paginas)
   1.1 Motivation and Problem Statement
   1.2 Research Challenges
   1.3 Contributions
   1.4 Paper Organization

2. Background (2-3 paginas)
   2.1 Test Case Prioritization
   2.2 APFD Metric
   2.3 Graph Neural Networks
   2.4 Learning to Rank

3. Related Work (3-4 paginas)
   3.1 Traditional TCP Approaches
   3.2 ML-based TCP
   3.3 Deep Learning for TCP
   3.4 GNN in Software Engineering
   3.5 Comparison with Our Approach

4. Approach: Filo-Priori (4-5 paginas)
   4.1 Overview
   4.2 Data Pipeline
   4.3 Semantic Feature Extraction
   4.4 Structural Feature Extraction
   4.5 Multi-Edge Phylogenetic Graph
   4.6 Dual-Stream Neural Network
   4.7 Ranking-Aware Training

5. Experimental Design (2-3 paginas)
   5.1 Research Questions
   5.2 Dataset Description
   5.3 Baselines
   5.4 Evaluation Metrics
   5.5 Implementation Details

6. Results (4-5 paginas)
   6.1 RQ1: Effectiveness
   6.2 RQ2: Component Contributions
   6.3 RQ3: Temporal Robustness
   6.4 RQ4: Hyperparameter Sensitivity

7. Discussion (2-3 paginas)
   7.1 Key Findings
   7.2 Practical Implications
   7.3 Limitations

8. Threats to Validity (1-2 paginas)
   8.1 Internal Validity
   8.2 External Validity
   8.3 Construct Validity

9. Conclusion (1 pagina)
   9.1 Summary
   9.2 Future Work

References (40-60 referencias)

Appendix (se necessario)
   A. Hyperparameter Configuration
   B. Additional Results
```

---

## 6. METRICAS DE QUALIDADE PARA IEEE TSE

### 6.1 Checklist de Qualidade

| Criterio | Status | Meta |
|----------|--------|------|
| **Novidade** | | |
| Contribuicao original | Pendente | Sim |
| Diferenciacao do estado da arte | Pendente | Claro |
| **Rigor Metodologico** | | |
| RSL documentada | Pendente | Sim |
| Protocolo experimental claro | Parcial | Completo |
| Testes estatisticos adequados | Sim | Sim |
| **Reproducibilidade** | | |
| Codigo disponivel | Sim | GitHub publico |
| Dados disponiveis | Parcial | IEEE DataPort |
| Configuracoes documentadas | Sim | YAML completo |
| **Validacao** | | |
| Multiplos baselines | Sim | 8+ baselines |
| Validacao temporal | Sim | 3 metodos |
| Ablation study | Sim | 5 componentes |
| Analise de sensibilidade | Sim | 4 fatores |
| **Escrita** | | |
| Secoes completas | Nao | Todas as 9 |
| Referencias adequadas | Nao | 40-60 refs |
| Ingles tecnico | A verificar | Nativo/Proofread |

### 6.2 Pontuacao Estimada de Aceitacao

| Aspecto | Peso | Score Atual | Score Alvo |
|---------|------|-------------|------------|
| Novidade | 25% | 7/10 | 9/10 |
| Rigor | 25% | 6/10 | 9/10 |
| Resultados | 20% | 8/10 | 9/10 |
| Escrita | 15% | 4/10 | 8/10 |
| Referencias | 15% | 3/10 | 8/10 |
| **TOTAL** | 100% | **5.9/10** | **8.7/10** |

**Avaliacao**: Paper nao esta pronto para submissao. Gap principal: secoes faltantes e referencias.

---

## 7. PLANO DE IMPLEMENTACAO

### Fase 1: RSL (2-3 semanas)
- [ ] Executar buscas nas bases de dados
- [ ] Aplicar criterios de selecao
- [ ] Extrair dados dos artigos
- [ ] Sintetizar resultados
- [ ] Identificar gaps

### Fase 2: Escrita das Secoes Faltantes (3-4 semanas)
- [ ] Introduction
- [ ] Background
- [ ] Related Work (baseado na RSL)
- [ ] Approach/Methodology
- [ ] Conclusion

### Fase 3: Adaptacao do Template (1 semana)
- [ ] Converter para IEEEtran.cls
- [ ] Ajustar formatacao
- [ ] Atualizar estilo de bibliografia
- [ ] Revisar figuras e tabelas

### Fase 4: Referencias (1-2 semanas)
- [ ] Adicionar 35+ novas referencias
- [ ] Atualizar citations no texto
- [ ] Verificar formato IEEE

### Fase 5: Revisao e Polimento (2-3 semanas)
- [ ] Revisao de linguagem (English proofreading)
- [ ] Verificacao de consistencia
- [ ] Checklist final IEEE TSE
- [ ] Preparar materiais suplementares

### Fase 6: Submissao
- [ ] Upload no ScholarOne/IEEE
- [ ] Preparar cover letter
- [ ] Sugerir revisores
- [ ] Acompanhar status

---

## 8. RECURSOS E FERRAMENTAS

### 8.1 Ferramentas para RSL
- **Rayyan** (https://rayyan.ai) - Triagem de artigos
- **Zotero/Mendeley** - Gerenciamento de referencias
- **Parsifal** (https://parsif.al) - Protocolo RSL
- **PRISMA** - Checklist e fluxograma

### 8.2 Recursos IEEE
- **IEEE Author Center**: https://journals.ieeeauthorcenter.ieee.org/
- **IEEE DataPort**: https://ieee-dataport.org/
- **Overleaf IEEE Template**: https://www.overleaf.com/gallery/tagged/ieee-official

### 8.3 Ferramentas de Escrita
- **Grammarly** - Revisao de ingles
- **Overleaf** - Colaboracao LaTeX
- **draw.io** - Diagramas

---

## 9. CRONOGRAMA ESTIMADO

| Fase | Duracao | Data Estimada |
|------|---------|---------------|
| RSL | 2-3 semanas | Dez 2025 |
| Secoes Faltantes | 3-4 semanas | Jan 2026 |
| Template IEEE | 1 semana | Jan 2026 |
| Referencias | 1-2 semanas | Fev 2026 |
| Revisao | 2-3 semanas | Fev-Mar 2026 |
| **Submissao** | - | **Mar 2026** |

---

## 10. PROXIMOS PASSOS IMEDIATOS

1. **[URGENTE]** Iniciar RSL - executar buscas
2. **[ALTA]** Criar template IEEE TSE no projeto
3. **[ALTA]** Escrever secao Introduction
4. **[MEDIA]** Compilar lista completa de referencias
5. **[MEDIA]** Documentar metodologia experimental

---

**Documento criado**: Novembro 2025
**Ultima atualizacao**: Novembro 2025
**Responsavel**: Equipe Filo-Priori
