# Especificação 00 — Visão Geral e Requisitos

## 1. Propósito do Projeto

O **cpp-vibes-ai** é um repositório **educativo** focado em construir uma **IA conversacional do zero em C++**, permitindo compreender passo a passo a matemática, as estruturas de dados e os algoritmos por trás do Machine Learning moderno.

## 2. Objetivos Principais

1. **Clareza didática acima de abstrações opacas**: cada conceito (tokenização, embeddings, backpropagation, atenção, otimização) deve ser implementado explicitamente e acompanhado de documentação em `docs/`.
2. **Evolução progressiva**: construir modelos em etapas independentes e executáveis (`Bigram -> MLP -> Mini-Transformer`), de modo que cada etapa resolva uma limitação concreta da anterior.
3. **Interatividade real**: cada etapa deve permitir testar o modelo via terminal (geração de texto e chat interativo).
4. **Desempenho honesto em CPU**: código C++ moderno com layout contíguo em memória e paralelismo simples via OpenMP, sem depender de GPU ou frameworks externos.

## 3. Requisitos Funcionais (RF)

- **RF01 — Tokenização UTF-8**: converter textos em português (incluindo acentos) para sequências de IDs inteiros e vice-versa sem quebrar caracteres multibyte.
- **RF02 — Treinamento do Zero**: treinar modelos a partir de arquivos texto simples (`dados/*.txt`) com exibição clara da evolução da perda (*loss*) por época/passo.
- **RF03 — Geração Autoregressiva**: gerar texto token a token com controle de temperatura (e futuramente `top-k`).
- **RF04 — Interface de Chat em Terminal**: permitir conversas interativas com o modelo treinado, suportando comandos de controle (`/temp`, `/sair` e futuramente `/limpar`, `/salvar`, `/carregar`).
- **RF05 — Persistência de Modelo (a partir da Etapa 2/3)**: salvar e carregar vocabulário e pesos treinados em arquivo binário simples para conversar sem retreinar toda vez.
- **RF06 — Modo Não-Interativo**: todo executável deve aceitar `--sem-chat` para permitir execução automatizada e validação rápida em linha de comando.

## 4. Requisitos Não-Funcionais (RNF)

- **RNF01 — Zero Dependências Externas de ML**: apenas biblioteca padrão C++20 (`STL`), `CMake` (>= 3.20) e `OpenMP` (opcional com fallback automático para single-thread).
- **RNF02 — Idioma**: 100% em português (identificadores de código, comentários, documentação em `docs/`, specs em `.spec/` e mensagens de terminal).
- **RNF03 — Qualidade de Compilação**: compilar sem avisos com `-Wall -Wextra -Wpedantic` (GCC/Clang).
- **RNF04 — Reprodutibilidade**: inicialização de pesos e amostragem controladas por semente determinística (`seed = 42` por padrão).
- **RNF05 — Validação Numérica**: toda nova camada com gradiente manual (MLP, Attention, LayerNorm) deve ter verificação de gradiente (*gradient checking* numérico) ou referência analítica antes de ser considerada pronta.
