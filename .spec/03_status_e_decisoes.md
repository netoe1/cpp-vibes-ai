# Especificação 03 — Status Atual e Decisões Arquiteturais (ADR)

> **Regra de manutenção**: este arquivo deve ser consultado antes de iniciar novas tarefas e atualizado ao final de cada marco concluído ou decisão técnica relevante.

## 1. Estado Atual do Projeto

- **Última atualização**: 2026-11-26
- **Etapa atual**: **Todas as 3 etapas da trilha base concluídas e validadas (`6/6` testes CTest passando, 0 warnings)!** Pronto para evoluções (ex.: expandir dataset, tokenizador BPE, KV-cache na geração).
- **Ambiente verificado**: Linux, `g++ 15.2.0`, `CMake 4.2.3`, OpenMP 4.5 (12 cores).

### Checklist de Progresso

- [x] Estrutura inicial do repositório, `CMakeLists.txt` e `README.md`
- [x] Dataset inicial em português (`dados/dialogos.txt` — 30 diálogos, 2976 tokens, 37 caracteres únicos)
- [x] Biblioteca comum (`src/comum/tokenizador.hpp`, `matriz.hpp`, `aleatorio.hpp` com `amostrar_top_k`)
- [x] **Etapa 1 — Bigram** implementada, documentada (`docs/01_bigram.md`) e validada (loss `2.0644` vs ótimo estatístico `2.0475`)
- [x] Especificações do projeto criadas e mantidas em `.spec/`
- [x] Diretrizes de agentes criadas em `AGENTS.md` (simulação pré-código + testes automatizados e de regressão)
- [x] Suíte de testes automatizados (`tests/` + `CTest`) cobrindo `src/comum/`, Etapa 1, Etapa 2 e Etapa 3 (`6/6` passando)
- [x] **Etapa 2 — MLP** (Embeddings, janela de 16 tokens, `tanh`, backpropagation manual, Momentum, checkpoints `.vibes`, gradient check `9.1e-7`, `docs/02_mlp.md`)
- [x] **Etapa 3 — Mini-Transformer (GPT)** (Causal Multi-Head Self-Attention, Pre-LayerNorm, GELU FFN, residuais, AdamW + Cosine LR + Clipping, checkpoints `.vibes`, Top-K, chat com histórico, gradient check `5.16e-5`, `docs/03_transformer.md`)
- [x] `Makefile` na raiz com atalhos para build, testes (`CTest`), treino e chat de todas as etapas (documentado no `README.md` e `.spec/`)
- [x] Expansão de datasets em `dados/` (`conversa_cotidiana.txt`, `programacao_cpp.txt`, `machine_learning.txt` e geração automática de `completo.txt` via `make dados-completo`)
- [x] Script `scripts/baixar_dados.py` + `make baixar-dados` para baixar e formatar grandes datasets abertos em português (`OpenAssistant/oasst1`, `FreedomIntelligence/alpaca-gpt4-portuguese`, *Dom Casmurro* — Gutenberg #55752) em `dados/externos/`
- [x] Suíte de treinos no `Makefile` (`suite-treino-rapida`, `suite-treino`, `suite-treino-grande`, `suite-benchmark`)
- [x] Evolução do Mini-Transformer: Tokenizador **Byte-Level BPE** (`src/comum/bpe.hpp`), **SFT** com máscara de loss (`--sft`), **Weight Tying**, checkpoint `VBGPT02` com estado AdamW (`--continuar`), CLI configurável (`--dim`, `--camadas`, `--contexto`, `--bpe`) e vetorização `-march=native`
- [x] Arquivamento das etapas anteriores (`etapa1_bigram`, `etapa2_mlp`, `teste_mlp.cpp`) em `backup/`, deixando apenas o Mini-Transformer atual ativo no `CMakeLists.txt` e `Makefile`
- [x] Regularização e experiência de chat: **Inverted Dropout** (`--dropout`), salvamento automático do melhor checkpoint de validação (`--melhor-val`) + Early Stopping (`--paciencia`), medidor de confiança + fallback honesto (`/conf`) e corpus calibrado para CPU (`dados/externos/conversas_cpu.txt`, `make treinar-cpu` / `make chat-cpu`)
- [x] Pasta `changelogs/` criada documentando a evolução completa commit a commit (`f761445` → `9986c34`)
- [x] **KV-Cache** na geração autoregressiva (`forward_token` validado com equivalência `< 1e-5` vs `forward_sequencia`), escalamento para `D=128, BPE=640, T=80` (~680k params) e 3 novos datasets ricos (`filosofia.txt`, `computacao.txt`, `musica_rock_metal.txt`)

## 2. Registro de Decisões Arquiteturais (ADRs)

### ADR-001 — Trilha Progressiva em Etapas Independentes
- **Decisão**: Construir 3 executáveis separados (`etapa1_bigram`, `etapa2_mlp`, `etapa3_transformer`) em vez de um único modelo monolítico.
- **Motivo**: Maximizar o valor educativo; o usuário pode comparar a evolução da perda, da arquitetura e da qualidade das respostas entre gerações de modelos.

### ADR-002 — C++20 Puro + OpenMP (Sem Frameworks Externos)
- **Decisão**: Não usar PyTorch/LibTorch, Eigen ou BLAS externo; implementar operações matriciais e gradientes diretamente em C++20 com paralelismo via OpenMP.
- **Motivo**: Garantir transparência total do que acontece na CPU e na matemática, mantendo o treino ágil em máquinas multicore.

### ADR-003 — Código, Comentários, Docs e Specs 100% em Português
- **Decisão**: Adotar português em todo o repositório (nomes de classes, funções, variáveis, comentários, `docs/` e `.spec/`).
- **Motivo**: Preferência explícita do autor para tornar o estudo mais acessível e fluido.

### ADR-004 — Tokenização UTF-8 por Caractere nas Etapas Iniciais
- **Decisão**: Começar com `TokenizadorUTF8` por caractere (agrupando bytes multibyte UTF-8 em 1 token).
- **Motivo**: Mantém o vocabulário pequeno (~40–80 tokens), elimina tokens desconhecidos complexos e facilita visualizar probabilidades. BPE (*Byte Pair Encoding*) pode ser adicionado como extensão opcional na Etapa 3.

### ADR-005 — Fluxo de Simulação Pré-Código e Testes de Regressão via CTest
- **Decisão**: Formalizar em `AGENTS.md` a simulação prévia de comportamento/dimensões antes de codificar e a execução obrigatória de testes automatizados e de regressão (`CTest`) ao concluir qualquer feature.
- **Motivo**: Bugs em Machine Learning frequentemente são silenciosos (o código compila, mas o gradiente ou shape está errado). Simular antes e rodar testes de regressão continuamente garante confiabilidade didática e técnica.

### ADR-006 — Precisão Dupla (`double`) nas Diferenças Finitas do Gradient Checking
- **Decisão**: Avaliar a função de perda em `double` (`eps = 1e-5`) durante `verificar_gradiente_numerico`, mantendo o treino principal em `float32`.
- **Motivo**: Em `float32`, a subtração `(L(θ+ε) - L(θ-ε)) / (2ε)` sofre erro de cancelamento numérico (~`1e-2`), mascarando a precisão real do backpropagation. Com avaliação em `double`, o erro relativo cai para `~1e-6`, garantindo detecção rigorosa de qualquer erro de gradiente.

### ADR-007 — Arquivamento das Etapas 1 e 2 em `backup/`
- **Decisão**: Mover `etapa1_bigram`, `etapa2_mlp` e `teste_mlp.cpp` para `backup/` e manter apenas a versão mais atual (`etapa3_transformer` com BPE + SFT + Weight Tying) ativa no `CMakeLists.txt` e `Makefile`.
- **Motivo**: Simplificar o fluxo diário de treino, testes e chat na versão mais capaz, preservando intacto o histórico educativo no diretório `backup/` e em `docs/`.
