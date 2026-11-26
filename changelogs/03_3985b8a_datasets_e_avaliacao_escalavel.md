# Changelog 03 — Datasets Temáticos, Downloader e Avaliação Escalável (`3985b8a`)

- **Commit**: `3985b8a`
- **Data**: 2026-11-26
- **Mensagem**: `Adiciona datasets tematicos, downloader de grandes corpora e otimiza avaliacao`
- **Escopo**: 6 arquivos alterados (`+612` / `-4` linhas)

## Motivação
Com apenas 30 diálogos em `dados/dialogos.txt` (~3 mil tokens), os modelos memorizavam rapidamente o treino (*overfitting*). Era necessário ampliar a variedade temática em português (com acentuação UTF-8 real) e viabilizar o treinamento em corpora de vários megabytes na CPU.

## O que mudou

### 1. Novos Datasets Curados em Português UTF-8 (`dados/`)
- `dados/conversa_cotidiana.txt`: diálogos naturais do dia a dia, cumprimentos, curiosidades e exemplos multi-turno.
- `dados/programacao_cpp.txt`: perguntas e respostas didáticas sobre C++20, `std::vector`, `std::span`, OpenMP, *race conditions*, CMake, `CTest` e complexidade Big-O.
- `dados/machine_learning.txt`: explicações sobre tokenização, embeddings, backpropagation, overfitting, Self-Attention, LayerNorm, AdamW, temperatura e Top-K.
- `dados/dialogos.txt` foi mantido intacto como baseline determinístico da suíte de regressão.

### 2. Downloader de Grandes Corpora (`scripts/baixar_dados.py`)
- Script em Python 3 puro (zero dependências externas) para baixar e converter grandes datasets abertos em português para o formato `Usuario: / IA:` em `dados/externos/` (ignorado no `.gitignore`):
  - `OpenAssistant/oasst1` filtrado para português (conversas multi-turno reais).
  - `FreedomIntelligence/alpaca-gpt4-portuguese` (milhares de pares de instrução/resposta).
  - *Dom Casmurro* (Machado de Assis, Project Gutenberg #55752, domínio público).
- Geração automática de `dados/externos/conversas_grande.txt` (~4,5 MB / 4,6M tokens caractere).

### 3. Otimização Crítica de Desempenho (`src/etapa3_transformer/main.cpp`)
- `construir_janelas_avaliacao` passou a amostrar no máximo `max_janelas = 32` janelas determinísticas espaçadas ao longo do split.
- **Impacto**: manteve comportamento idêntico em arquivos pequenos e eliminou um gargalo que levaria minutos por avaliação em arquivos de vários MB (passando para `< 0.2s`).
