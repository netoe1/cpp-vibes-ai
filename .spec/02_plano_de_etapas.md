# Especificação 02 — Plano Técnico das Etapas

Cada etapa gera seu próprio executável independente no `CMakeLists.txt` e seu próprio guia teórico em `docs/`.

---

## Etapa 1 — Modelo Bigram (Estatístico e Neural) `[CONCLUÍDA]`

- **Executável**: `etapa1_bigram` (`src/etapa1_bigram/main.cpp`)
- **Doc**: `docs/01_bigram.md`
- **Arquitetura**:
  - Entrada: 1 token $x_t$ (contexto de tamanho 1).
  - Parâmetros: matriz $W \in \mathbb{R}^{V \times V}$.
  - Forward: $\text{probs} = \text{softmax}(W[x_t])$.
  - Loss: Entropia Cruzada $-\ln(p_{y_t})$.
  - Otimização: Batch Gradient Descent com gradiente analítico paralelizado em OpenMP.
- **Critérios de Aceitação**:
  - [x] Compilar sem avisos em C++20 com OpenMP.
  - [x] Perda neural convergir próxima à perda do Bigram estatístico por contagem (`2.0644` vs `2.0475`).
  - [x] Suportar geração de amostra, `--sem-chat` e chat interativo com `/temp`.

---

## Etapa 2 — MLP (Multi-Layer Perceptron, estilo Bengio 2003) `[CONCLUÍDA]`

- **Executável**: `etapa2_mlp` (`src/etapa2_mlp/main.cpp`)
- **Doc**: `docs/02_mlp.md`
- **Objetivo**: Introduzir **Embeddings**, **janela de contexto** ($N$ tokens anteriores), **camada oculta não-linear** e **backpropagation manual** com mini-batches.
- **Arquitetura**:
  - Hiperparâmetros iniciais sugeridos:
    - Janela de contexto: $B = 16$ tokens (com padding inicial).
    - Dimensão de embedding: $E = 24$ (ou $32$).
    - Camada oculta: $H = 128$ neurônios com ativação `tanh`.
  - Parâmetros:
    - Tabela de embeddings $C \in \mathbb{R}^{V \times E}$.
    - Camada 1: $W_1 \in \mathbb{R}^{(B \cdot E) \times H}$, $b_1 \in \mathbb{R}^{H}$.
    - Camada 2 (saída): $W_2 \in \mathbb{R}^{H \times V}$, $b_2 \in \mathbb{R}^{V}$.
  - Forward:
    1. Concatenar embeddings dos $B$ tokens de contexto $\rightarrow x \in \mathbb{R}^{B \cdot E}$.
    2. $h = \tanh(x W_1 + b_1)$.
    3. $\text{logits} = h W_2 + b_2$, seguido de `softmax` e entropia cruzada.
  - Otimização: Mini-Batch SGD (com decaimento da taxa de aprendizado ou Momentum), paralelizado sobre os exemplos do batch com OpenMP.
  - Divisão treino/validação (ex.: 90% / 10%) para demonstrar *generalização* vs *overfitting*.
- **Melhorias no Dataset e Infraestrutura**:
  - Expandir `dados/dialogos.txt` se necessário para enriquecer padrões.
  - Adicionar teste de *gradient checking* numérico (`--verificar-gradiente`) para provar que o backpropagation manual está matematicamente correto.
  - Suporte opcional a salvar/carregar pesos em formato binário simples.
- **Critérios de Aceitação**:
  - [x] Gradient check numérico com erro relativo $< 10^{-3}$ (obtido: `9.11e-7` em teste unitário e `2.44e-6` no modelo completo, usando diferenças finitas em `double`).
  - [x] Perda de treino significativamente menor que a do Bigram (`0.0220` em 80 épocas vs `2.0644` do Bigram).
  - [x] Geração formando palavras inteiras e respostas coerentes no chat.
  - [x] Documentação completa em `docs/02_mlp.md`.

---

## Etapa 3 — Mini-Transformer Conversacional (Estilo GPT) `[CONCLUÍDA]`

- **Executável**: `etapa3_transformer` (`src/etapa3_transformer/main.cpp`)
- **Doc**: `docs/03_transformer.md`
- **Objetivo**: Implementar a arquitetura Transformer decoder-only completa do zero em C++ para conversa com contexto real.
- **Arquitetura**:
  - Embeddings de token + Embeddings posicionais aprendíveis (contexto $T = 64$ ou $128$ tokens).
  - $L$ blocos Transformer ($L = 2$ a $4$), cada um com:
    - **Pre-LayerNorm**
    - **Causal Multi-Head Self-Attention** ($Q, K, V$ com máscara triangular causal)
    - **Conexão Residual**
    - **Feed-Forward (MLP com GELU)** + **Conexão Residual**
  - LayerNorm final + projeção linear para o vocabulário $V$.
  - Otimizador **AdamW** + *gradient clipping* + *cosine learning rate schedule*.
  - Serialização completa (salvar/carregar checkpoint `.vibes` em disco).
  - Amostragem com **temperatura** e **top-k**.
  - Chat com **histórico de conversa** na janela de contexto.
- **Critérios de Aceitação**:
  - [x] Gradient check numérico validando Attention, LayerNorm e bloco Transformer (obtido: `1.53e-4` no teste unitário e `5.16e-5` no modelo completo, abaixo de `< 1e-3`).
  - [x] Treino estável com perda convergindo em CPU (OpenMP) em tempo razoável (validação caiu de `3.6051` para `2.1300` em 100 passos, melhor generalização que a MLP).
  - [x] Salvar e recarregar pesos preservando exatamente as mesmas previsões (`< 1e-6`).
  - [x] Chat com histórico na janela de contexto (`/temp`, `/topk`, `/limpar`, `/salvar`, `/sair`).
  - [x] Documentação completa em `docs/03_transformer.md`.
