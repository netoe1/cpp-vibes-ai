# Changelog 06 — KV-Cache, Novos Datasets Temáticos e `/temas` no Chat

- **Data**: 2026-11-26
- **Resumo**: Geração acelerada com KV-Cache, 3 novos datasets ricos (Filosofia, Computação e Rock/Heavy Metal), escalamento do modelo e comando `/temas` no chat.

## Motivação
1. Ampliar o conhecimento conversacional da IA para temas ricos e aprofundados: **Filosofia**, **Ciência da Computação** e **Música (Heavy Metal e Rock)**.
2. Acelerar a geração autoregressiva em CPU para suportar modelos maiores sem lentidão no chat.
3. Permitir que o usuário descubra facilmente o que a IA sabe conversar diretamente pelo terminal.

## O que mudou

### 1. Novos Conjuntos de Dados de Alta Qualidade (`dados/`)
- `dados/filosofia.txt` (42 diálogos): Sócrates, Platão, Aristóteles, Estoicismo (Sêneca, Epicteto, Marco Aurélio, *Amor Fati*, *Memento Mori*), Descartes, Kant, Nietzsche, Sartre, Camus, Schopenhauer, Spinoza, lógica e filosofia da IA.
- `dados/computacao.txt` (36 diálogos): Alan Turing, Von Neumann, hierarquia de cache, Linux, processos/threads, compiladores, estruturas de dados, grafos, TCP/UDP, DNS, HTTPS, criptografia, ACID, SIMD e Docker.
- `dados/musica_rock_metal.txt` (36 diálogos): Black Sabbath, Iron Maiden, Metallica, Megadeth, Slayer, Dio, Judas Priest, Sepultura, Angra, Andre Matos, Pantera, Tool, Death, Led Zeppelin, Pink Floyd, Queen, Deep Purple, AC/DC, Grunge e teoria musical (power chords, palm mute, Drop D, compassos ímpares).
- `dados/completo.txt` cresceu para **894 linhas (~21 mil tokens BPE, 299 perguntas)**.

### 2. KV-Cache na Geração (`src/etapa3_transformer/transformer.hpp`)
- Implementadas as estruturas `KVCache` e `forward_token` para armazenar $K$ e $V$ já computados em cada camada, evitando recalcular toda a sequência anterior a cada novo token gerado.
- Adicionado teste automatizado em `tests/teste_transformer.cpp` comprovando equivalência numérica exata (`max_diff < 1e-5`) entre `forward_token` (com KV-Cache) e `forward_sequencia`.

### 3. Escalamento do Modelo Padrão (`Makefile`)
- Hiperparâmetros padrão atualizados para absorver o corpus expandido: `VOCAB_BPE = 640`, `DIM = 128`, `CAMADAS = 3`, `CONTEXTO = 80`, `PASSOS = 900` (**687.872 parâmetros** com Weight Tying).
- Perda de treino em `dados/completo.txt` caiu de `6.48` para `0.31`, com respostas coerentes e confiança de `94%–97%` sobre Filosofia, Computação e Heavy Metal.

### 4. Descoberta de Assuntos no Chat (`src/etapa3_transformer/main.cpp`)
- Adicionado o comando `/temas [filtro]` (e `/ajuda`), além de 5 sugestões automáticas ao iniciar `make chat`.
- Permite filtrar perguntas conhecidas por palavra-chave (ex.: `/temas metal`, `/temas filosofia`, `/temas C++`).

### 5. Governança (`AGENTS.md`)
- Formalizada a seção **Changelogs e Commits Organizados** no `AGENTS.md`, tornando obrigatória a atualização de `changelogs/` e `.spec/` junto com a validação de testes a cada marco.
