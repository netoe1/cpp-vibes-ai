# cpp-vibes-ai

Uma IA conversacional construída **do zero em C++20**, sem bibliotecas externas de Machine Learning, com aceleração em CPU via **OpenMP**, **KV-Cache** e vetorização nativa (`-O3 -march=native`).

O objetivo principal deste repositório é **educativo**: entender, linha por linha, a matemática e a engenharia por trás de modelos de linguagem modernos (**Mini-Transformer estilo GPT** com **Byte-Level BPE**, **SFT**, **Weight Tying**, **Dropout** e **AdamW**).

> As etapas educativas iniciais (**Etapa 1: Bigram** e **Etapa 2: MLP**) estão preservadas em [backup/](backup/), a teoria completa está em [docs/](docs/) e o histórico detalhado da evolução está em [changelogs/](changelogs/).

---

## Início Rápido (2 comandos)

### Requisitos
- Compilador C++20 (`g++` ou `clang++`) com suporte a OpenMP
- `cmake` (>= 3.20) e `make`
- *(Opcional)* `python3` para baixar grandes datasets externos

### 1. Treinar a IA e abrir o chat

```bash
# Compila, junta todos os datasets locais, treina o Mini-Transformer (BPE 640 + SFT) e salva o checkpoint
make treinar

# Abre o chat interativo carregando o modelo treinado (instantâneo)
make chat
```

---

## Como Conversar com a IA

Ao rodar `make chat`, a IA já mostra **5 sugestões de perguntas** que ela conhece. Basta digitar sua mensagem e pressionar `Enter`.

### Exemplo de conversa no terminal

```text
Voce: Qual banda criou o heavy metal?
IA (GPT | conf 94%): O Black Sabbath, formado em Birmingham em 1968 com Tony Iommi, Ozzy Osbourne, Geezer Butler e Bill Ward, é considerado o criador do heavy metal.

Voce: O que significa Amor Fati?
IA (GPT | conf 96%): Amor Fati significa "amor ao destino": não apenas suportar o que acontece de necessário na vida, mas abraçá-lo e usá-lo como combustível para crescer.

Voce: /temas filosofia
[Temas] Mostrando 4 de 4 perguntas sobre 'filosofia':
  - O que é filosofia?
  - Como a filosofia ajuda quem trabalha com computação?
  - Qual a relação entre filosofia e heavy metal?
  - Você gosta de filosofia?
```

### Comandos disponíveis dentro do chat

| Comando | O que faz | Exemplo |
| :--- | :--- | :--- |
| `/temas` ou `/ajuda` | Sorteia 12 perguntas conhecidas do corpus para puxar assunto | `/temas` |
| `/temas <filtro>` | Lista perguntas conhecidas filtradas por palavra-chave | `/temas metal`, `/temas C++`, `/temas linux` |
| `/temp <valor>` | Ajusta a temperatura da geração (`0.2` mais fiel, `0.8` mais criativo) | `/temp 0.25` |
| `/topk <k>` | Restringe o sorteio aos `K` tokens mais prováveis (`0` desativa) | `/topk 4` |
| `/conf <0..1>` | Ajusta o limiar mínimo de confiança para resposta (padrão `0.72`; `0` mostra a geração crua) | `/conf 0` |
| `/limpar` | Limpa o histórico acumulado da conversa | `/limpar` |
| `/salvar <arquivo>` | Salva os pesos atuais (e `.bpe`) em disco | `/salvar meu_modelo.vibes` |
| `/sair` | Encerra o chat | `/sair` |

> **Como funciona o indicador `conf XX%`**: ele mede a confiança real do modelo (`temperatura = 1.0`) nos tokens gerados. Quando você muda de assunto ou pergunta algo fora do que ela aprendeu, a IA primeiro tenta responder sem o histórico antigo e, se a confiança continuar baixa, avisa honestamente em vez de inventar palavras.

---

## Como Ensinar Novos Assuntos para a IA

Todos os datasets em `dados/` usam o formato simples em texto UTF-8:

```text
Usuario: Sua pergunta aqui?
IA: A resposta que você quer ensinar para a IA.

Usuario: Outra pergunta?
IA: Outra resposta.
```

Para ensinar novos diálogos:
1. Adicione seus pares `Usuario:` / `IA:` em um dos arquivos de `dados/` (como [dados/conversa_cotidiana.txt](dados/conversa_cotidiana.txt)).
2. Rode `make treinar` (ou `make continuar PASSOS=400` para continuar treinando sem começar do zero).
3. Rode `make chat` e teste!

---

## Modos de Treino e Grandes Datasets

```bash
# Ver todos os alvos e variáveis disponíveis
make help

# Rodar a suíte completa de testes unitários, gradient checking e KV-Cache
make test

# Continuar treinando o modelo atual preservando o estado do AdamW
make continuar PASSOS=400

# Baixar grandes datasets abertos em português (OpenAssistant OASST1 + Alpaca GPT-4 PT)
make baixar-dados

# Treinar e conversar no corpus calibrado para CPU (~232 KB, Dropout + Melhor Validação)
make treinar-cpu
make chat-cpu

# Treinar e conversar no corpus grande completo (~4,5 MB)
make treinar-grande PASSOS_GRANDE=2000
make chat-grande

# Remover arquivos compilados e checkpoints em build/
make clean
```

### Personalizando hiperparâmetros na linha de comando

Você pode ajustar arquitetura e treino direto pelo `make`:

```bash
make treinar DIM=128 CAMADAS=4 CONTEXTO=96 VOCAB_BPE=640 LR=0.004 PASSOS=1000
```

Ou chamar o executável diretamente para experimentar todas as flags:

```bash
./build/etapa3_transformer dados/completo.txt \
  --bpe 640 --sft --dropout 0.05 \
  --dim 128 --camadas 3 --cabecas 4 --contexto 80 \
  --batch 16 --lr 0.004 --passos 900 \
  --salvar build/modelo.vibes
```

---

## Estrutura do Projeto

```text
cpp-vibes-ai/
├── CMakeLists.txt           # Build C++20 + OpenMP + CTest (versão ativa)
├── Makefile                 # Atalhos práticos para build, testes, treino e chat
├── AGENTS.md                # Diretrizes de simulação pré-código, testes e changelogs
├── backup/                  # Versões educativas anteriores (Bigram e MLP)
├── changelogs/              # Histórico detalhado da evolução do projeto por commit
├── dados/
│   ├── dialogos.txt            # Corpus inicial curto (baseline/testes)
│   ├── conversa_cotidiana.txt  # Conversas naturais do dia a dia e variações
│   ├── programacao_cpp.txt     # Diálogos sobre C++20, OpenMP e CMake
│   ├── machine_learning.txt    # Diálogos sobre ML, Backprop, Attention e LLMs
│   ├── filosofia.txt           # Sócrates, Estoicismo, Nietzsche, Kant, Camus e ética da IA
│   ├── computacao.txt          # SO, Linux, arquitetura, redes, compiladores e algoritmos
│   ├── musica_rock_metal.txt   # Heavy Metal, Rock clássico, subgêneros e teoria musical
│   └── completo.txt            # Todos os datasets locais combinados (gerado pelo Makefile)
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
└── tests/                   # Testes unitários, gradient checking, KV-Cache e integração (CTest)
```
