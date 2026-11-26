# cpp-vibes-ai

Uma IA conversacional construída **do zero em C++20**, sem bibliotecas externas de Machine Learning, com aceleração simples em CPU via **OpenMP**.

O objetivo principal deste repositório é **educativo**: entender, linha por linha, a matemática e a engenharia por trás de modelos de linguagem — desde a previsão estatística mais simples até um Mini-Transformer estilo GPT.

## Trilha Educativa

| Etapa | Arquitetura | Conceitos aprendidos | Status |
| :--- | :--- | :--- | :--- |
| **Etapa 1** | **Bigram (Estatístico e Neural)** | Tokenização UTF-8, logits, `softmax`, entropia cruzada (*cross-entropy*), gradiente descendente, amostragem com temperatura | Pronto |
| **Etapa 2** | **MLP (Multi-Layer Perceptron)** | Embeddings, janela de contexto ($N$ tokens), camada oculta (`tanh`), *backpropagation* manual, mini-batches, Momentum, *gradient checking* e checkpoints | Pronto |
| **Etapa 3** | **Mini-Transformer (GPT)** | Self-Attention causal, Multi-Head Attention, LayerNorm, GELU, conexões residuais, AdamW, Top-K, chat com histórico e persistência de pesos | Pronto |

Consulte o roteiro completo em [docs/00_roteiro.md](docs/00_roteiro.md).

## Estrutura do Projeto

```text
cpp-vibes-ai/
├── CMakeLists.txt           # Configuração de build (C++20 + OpenMP)
├── dados/
│   └── dialogos.txt         # Corpus inicial de diálogos em português
├── docs/
│   ├── 00_roteiro.md        # Visão geral da trilha educativa
│   └── 01_bigram.md         # Teoria e matemática da Etapa 1
└── src/
    ├── comum/               # Código compartilhado entre as etapas
    │   ├── aleatorio.hpp    # RNG determinístico e amostragem
    │   ├── matriz.hpp       # Matriz 2D + operações com OpenMP
    │   └── tokenizador.hpp  # Tokenizador UTF-8 a nível de caractere
    └── etapa1_bigram/
        └── main.cpp         # Treino, comparação estatística vs neural e chat
```

## Como Compilar e Executar

### Requisitos
- Compilador C++20 (`g++` ou `clang++`) com suporte a OpenMP
- `cmake` (>= 3.20)

### Build (modo Release recomendado para treino rápido)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Executar as Etapas

```bash
# Etapa 1: Bigram
./build/etapa1_bigram dados/dialogos.txt

# Etapa 2: MLP (com gradient check opcional e salvamento de pesos)
./build/etapa2_mlp dados/dialogos.txt --verificar-gradiente --salvar modelo_mlp.vibes

# Etapa 3: Mini-Transformer (GPT) com chat com histórico e checkpoints
./build/etapa3_transformer dados/dialogos.txt --verificar-gradiente --salvar modelo_gpt.vibes

# Rodar toda a suíte de testes automatizados e de regressão
ctest --test-dir build --output-on-failure
```

Todos os executáveis aceitam `--sem-chat` para rodar apenas treino e geração de exemplo sem entrar no chat interativo.
