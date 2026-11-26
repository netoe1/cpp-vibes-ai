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
│   ├── dialogos.txt            # Corpus inicial curto (baseline/regressão)
│   ├── conversa_cotidiana.txt  # Conversas naturais do dia a dia e multi-turno (UTF-8)
│   ├── programacao_cpp.txt     # Diálogos educativos sobre C++20, OpenMP e CMake
│   ├── machine_learning.txt    # Diálogos sobre ML, Backprop, Attention e LLMs
│   └── completo.txt            # Todos os datasets combinados (via make dados-completo)
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
- `cmake` (>= 3.20) e `make`

### Atalhos com `Makefile` (Recomendado)

O repositório inclui um [Makefile](Makefile) com atalhos prontos para compilar, testar e conversar com cada etapa:

```bash
make help         # Lista todos os comandos disponíveis
make build        # Compila todas as etapas e testes (Release + OpenMP)
make test         # Roda toda a suíte de testes automatizados e regressão (CTest)

# Treinar e abrir chat direto em cada etapa
make etapa1       # Etapa 1: Bigram
make etapa2       # Etapa 2: MLP
make etapa3       # Etapa 3: Mini-Transformer (GPT)

# Conversar rapidamente usando checkpoint salvo (treina automaticamente se ainda não existir)
make chat-mlp          # Chat com a MLP usando build/modelo_mlp.vibes
make chat-gpt          # Chat com o GPT usando build/modelo_gpt.vibes

# Juntar todos os datasets em dados/completo.txt e treinar/conversar nele
make dados-completo
make treinar-gpt-completo PASSOS=1000
make chat-gpt-completo

# Ou escolher qualquer dataset específico pela variável DADOS
make etapa3 DADOS=dados/conversa_cotidiana.txt PASSOS=600
make etapa3 DADOS=dados/programacao_cpp.txt PASSOS=600
make etapa3 DADOS=dados/machine_learning.txt PASSOS=600

# Baixar GRANDES datasets abertos em português (OASST1, Alpaca GPT-4 PT)
make baixar-dados
make treinar-gpt-grande PASSOS=1500
make chat-gpt-grande

# Suíte de treinos (treinar e comparar todas as etapas em sequência)
make suite-treino-rapida   # Smoke de treino nas 3 etapas em dados/completo.txt
make suite-treino          # Treina as 3 etapas em dados/completo.txt e salva checkpoints
make suite-treino-grande   # Treina no corpus grande (~4.5 MB em dados/externos/)
make suite-benchmark       # Compara tempo e perda das 3 arquiteturas lado a lado

make clean        # Remove a pasta build/
```

### Build e execução manual via CMake (Opcional)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Todos os executáveis aceitam `--sem-chat` para rodar apenas treino e geração de exemplo sem entrar no chat interativo.
