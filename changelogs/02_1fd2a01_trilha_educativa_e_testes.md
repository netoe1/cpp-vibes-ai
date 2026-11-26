# Changelog 02 — Trilha Educativa Completa, Specs e Testes (`1fd2a01`)

- **Commit**: `1fd2a01`
- **Data**: 2026-11-26
- **Mensagem**: `Adiciona testes para componentes do projeto`
- **Escopo**: 23 arquivos adicionados (`+4.076` linhas)

## O que foi construído

### 1. Governança, Especificações e Didática
- **`.spec/`**: especificações vivas (`00_visao_e_requisitos.md`, `01_arquitetura_e_convencoes.md`, `02_plano_de_etapas.md`, `03_status_e_decisoes.md`) definindo requisitos, convenções C++20/OpenMP 100% em português e ADRs (`ADR-001` a `ADR-006`).
- **`AGENTS.md`**: diretrizes obrigatórias de simulação pré-código (*shapes*, invariantes, estabilidade numérica) e testes automatizados/regressão via `CTest`.
- **`docs/`**: guias teóricos passo a passo (`00_roteiro.md`, `01_bigram.md`, `02_mlp.md`, `03_transformer.md`).

### 2. Biblioteca Comum (`src/comum/`)
- `tokenizador.hpp`: tokenizador UTF-8 por caractere (preservando acentos multibyte do português como 1 token).
- `matriz.hpp`: matriz 2D contígua em memória (*row-major*), `softmax` numericamente estável (subtração do máximo + temperatura) e multiplicação de matrizes paralelizada com OpenMP.
- `aleatorio.hpp`: gerador determinístico (`std::mt19937`), amostragem categórica e `amostrar_top_k`.

### 3. Evolução Progressiva dos 3 Modelos (Zero Dependências Externas)
1. **Etapa 1 — Bigram (`src/etapa1_bigram/main.cpp`)**:
   - Comparou a tabela estatística ótima (`loss = 2.0475`) com o Bigram neural por gradiente descendente em OpenMP (`loss = 2.0644`).
   - Provou que olhar apenas 1 caractere anterior gera sílabas do português, mas sem memória de frase.
2. **Etapa 2 — MLP (`src/etapa2_mlp/mlp.hpp` e `main.cpp`)**:
   - Introduziu Embeddings ($C \in \mathbb{R}^{V \times E}$), janela de contexto de 16 caracteres, camada oculta `tanh`, backpropagation manual, Mini-Batch SGD + Momentum e checkpoints binários (`VBMLP01`).
   - Perda de treino caiu de `3.60` para `0.02` em `dados/dialogos.txt`, passando a gerar respostas coerentes do domínio.
3. **Etapa 3 — Mini-Transformer / GPT (`src/etapa3_transformer/transformer.hpp` e `main.cpp`)**:
   - Implementou Embeddings de token + posição ($T = 64, D = 64, 4$ cabeças, $2$ camadas), Pre-LayerNorm, Causal Multi-Head Self-Attention ($Q, K, V, O$), FFN com GELU, conexões residuais, AdamW + Gradient Clipping + Cosine LR, checkpoints (`VBGPT01`) e chat com histórico.
   - Generalização superior à MLP: perda de validação caiu para `2.13` (contra `2.83` da MLP no mesmo corpus).

### 4. Suíte Automatizada de Testes (`tests/` + `CTest`)
- `teste_comum.cpp`: UTF-8, `softmax`, multiplicação de matrizes OpenMP, determinismo e Top-K.
- `teste_mlp.cpp`: *gradient checking* numérico em `double` (`9.11e-7`), convergência e persistência.
- `teste_transformer.cpp`: teste de causalidade da máscara de atenção, *gradient checking* completo (`1.53e-4` unitário / `5.16e-5` no modelo real), convergência AdamW e integridade de checkpoints.
