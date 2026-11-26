# Especificação 01 — Arquitetura e Convenções Técnicas

## 1. Estrutura de Diretórios

```text
cpp-vibes-ai/
├── .spec/                     # Especificações vivas do projeto (consultar e atualizar sempre)
│   ├── 00_visao_e_requisitos.md
│   ├── 01_arquitetura_e_convencoes.md
│   ├── 02_plano_de_etapas.md
│   └── 03_status_e_decisoes.md
├── CMakeLists.txt             # Build C++20 + OpenMP; 1 executável por etapa
├── README.md                  # Porta de entrada e instruções rápidas
├── dados/                     # Datasets de treinamento em texto puro (UTF-8)
│   └── dialogos.txt
├── docs/                      # Material educativo (teoria + matemática de cada etapa)
│   ├── 00_roteiro.md
│   ├── 01_bigram.md
│   ├── 02_mlp.md              # (a criar na Etapa 2)
│   └── 03_transformer.md      # (a criar na Etapa 3)
└── src/
    ├── comum/                 # Headers reutilizáveis entre todas as etapas
    │   ├── aleatorio.hpp      # PRNG determinístico e amostragem categórica
    │   ├── matriz.hpp         # Matriz 2D contígua (row-major) + softmax + matmul OpenMP
    │   └── tokenizador.hpp    # Tokenizador UTF-8 por caractere
    ├── etapa1_bigram/         # Executável: etapa1_bigram
    ├── etapa2_mlp/            # Executável: etapa2_mlp
    └── etapa3_transformer/    # Executável: etapa3_transformer
└── tests/                     # Testes unitários, gradient checking e regressão (CTest)
    ├── teste_comum.cpp
    ├── teste_mlp.cpp
    └── teste_transformer.cpp
```

## 2. Convenções de Código C++

- **Padrão**: C++20 (`CMAKE_CXX_STANDARD 20`), sem extensões específicas de compilador.
- **Namespace principal**: `vibes` para código compartilhado em `src/comum/`.
- **Nomenclatura em português**:
  - Classes e structs: `PascalCase` (ex.: `TokenizadorUTF8`, `ModeloBigramNeural`, `GeradorAleatorio`).
  - Funções, métodos e variáveis: `snake_case` (ex.: `treinar_epoca`, `taxa_aprendizado`, `tamanho_vocabulario`).
  - Atributos privados: `snake_case_` com underscore no final (ex.: `pesos_`, `vocab_`).
  - Constantes: `constexpr` em `snake_case` ou `kPascalCase`.
- **Memória e Cache**:
  - Matrizes e tensores devem armazenar `float` em um único `std::vector<float>` contíguo (*row-major*).
  - Evitar alocações dentro de laços quentes (*hot loops*) de treino; reutilizar buffers de ativação e gradiente quando possível.
- **Paralelismo com OpenMP**:
  - Sempre proteger diretivas com `#ifdef USAR_OPENMP` para garantir compilação mesmo sem OpenMP.
  - Em acumulação de gradientes, usar acumuladores locais por thread e somar ao final com `#pragma omp critical` (ou `reduction` para escalares), evitando *race conditions* e *false sharing*.

## 3. Formato do Dataset Conversacional

Arquivos em `dados/` seguem o formato textual simples em UTF-8, separando turnos por linha e conversas por linha em branco:

```text
Usuario: <mensagem do usuário>
IA: <resposta esperada da IA>

Usuario: <próxima mensagem>
IA: <próxima resposta>
```

Esse formato é simples o suficiente para modelos a nível de caractere aprenderem a estrutura de turno (`Usuario:` -> `IA:`) e pararem ao encontrar `\n`.

## 4. Comandos Padrão de Build e Execução

```bash
# Configurar e compilar em Release (recomendado para treino rápido)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Rodar uma etapa específica (modo chat)
./build/<nome_da_etapa> dados/dialogos.txt

# Rodar em modo validação rápida (sem bloquear no stdin)
./build/<nome_da_etapa> dados/dialogos.txt --sem-chat
```
