# cpp-vibes-ai

Uma IA conversacional construída **do zero em C++20**, sem bibliotecas externas de Machine Learning, com aceleração em CPU via **OpenMP** e vetorização nativa (`-O3 -march=native`).

O objetivo principal deste repositório é **educativo**: entender, linha por linha, a matemática e a engenharia por trás de modelos de linguagem modernos (**Mini-Transformer estilo GPT** com **Byte-Level BPE**, **SFT**, **Weight Tying** e **AdamW**).

> As etapas iniciais da trilha educativa (**Etapa 1: Bigram** e **Etapa 2: MLP**) estão preservadas em [backup/](backup/) com sua documentação completa em [docs/](docs/).

## Estrutura do Projeto

```text
cpp-vibes-ai/
├── CMakeLists.txt           # Build C++20 + OpenMP + CTest (versão ativa)
├── Makefile                 # Atalhos práticos para build, testes, treino e chat
├── backup/                  # Versões educativas anteriores (Bigram e MLP)
├── changelogs/              # Histórico detalhado da evolução do projeto por commit
├── dados/
│   ├── dialogos.txt            # Corpus inicial curto (baseline/testes)
│   ├── conversa_cotidiana.txt  # Conversas naturais do dia a dia e multi-turno
│   ├── programacao_cpp.txt     # Diálogos sobre C++20, OpenMP e CMake
│   ├── machine_learning.txt    # Diálogos sobre ML, Backprop, Attention e LLMs
│   ├── filosofia.txt           # Sócrates, Estoicismo, Nietzsche, Kant, Camus e ética da IA
│   ├── computacao.txt          # SO, Linux, arquitetura, redes, compiladores e algoritmos
│   ├── musica_rock_metal.txt   # Heavy Metal, Rock clássico, subgêneros e teoria musical
│   └── completo.txt            # Todos os datasets locais combinados
├── docs/
│   ├── 00_roteiro.md        # Visão geral da trilha educativa
│   ├── 01_bigram.md         # Teoria da Etapa 1 (arquivada em backup/)
│   ├── 02_mlp.md            # Teoria da Etapa 2 (arquivada em backup/)
│   └── 03_transformer.md    # Teoria do Mini-Transformer atual (GPT + BPE + SFT)
├── scripts/
│   └── baixar_dados.py      # Baixa grandes datasets abertos em PT (OASST1, Alpaca-PT)
├── src/
│   ├── comum/               # Código compartilhado (BPE, UTF-8, Matriz, Aleatório)
│   └── etapa3_transformer/  # Versão atual: Mini-Transformer Conversacional (GPT)
└── tests/                   # Testes unitários, gradient checking e integração (CTest)
```

## Como Compilar, Treinar e Conversar

### Requisitos
- Compilador C++20 (`g++` ou `clang++`) com suporte a OpenMP
- `cmake` (>= 3.20) e `make`

### Comandos principais (`Makefile`)

```bash
make help              # Lista todos os comandos e variáveis
make build             # Compila o Mini-Transformer e os testes
make test              # Roda toda a suíte de testes e gradient checking (CTest)

# Treinar e conversar no corpus curado (rápido e coerente)
make treinar           # Treina BPE 320 + SFT em dados/completo.txt -> build/modelo.vibes
make continuar         # Continua treino preservando estado do AdamW
make chat              # Abre o chat carregando build/modelo.vibes

# Treinar e conversar no corpus otimizado para CPU (~250 KB, Dropout + Melhor Validação)
make treinar-cpu       # Treina em dados/externos/conversas_cpu.txt -> build/modelo_cpu.vibes
make continuar-cpu     # Continua treino preservando AdamW
make chat-cpu          # Abre chat com indicador de confiança e fallback (/conf)

# Treinar e conversar no corpus grande (4,5 MB: OASST1 + Alpaca-PT)
make baixar-dados      # Baixa os datasets externos em dados/externos/
make treinar-grande    # Treina no corpus grande -> build/modelo_gpt_bpe.vibes
make continuar-grande  # Retoma treino no corpus grande
make chat-grande       # Abre o chat carregando build/modelo_gpt_bpe.vibes

make clean             # Remove build/
```

Você pode ajustar hiperparâmetros direto na linha de comando:
```bash
make treinar PASSOS=800 DIM=128 CAMADAS=4 CONTEXTO=96 LR=0.004
```
