# Backup das Etapas Educativas Anteriores

Esta pasta preserva as versões iniciais construídas durante a trilha educativa do **cpp-vibes-ai**, mantidas como referência histórica de aprendizado:

- `src/etapa1_bigram/` — **Etapa 1**: Modelo Bigram (estatístico vs. neural com gradiente descendente). Documentação em [docs/01_bigram.md](../docs/01_bigram.md).
- `src/etapa2_mlp/` — **Etapa 2**: MLP estilo Bengio (2003) com Embeddings, janela fixa de 16 caracteres, `tanh` e Momentum. Documentação em [docs/02_mlp.md](../docs/02_mlp.md).
- `tests/teste_mlp.cpp` — Testes unitários e *gradient checking* da Etapa 2.

A versão ativa e atual do projeto é o **Mini-Transformer Conversacional (GPT + BPE + SFT + Weight Tying)** em [src/etapa3_transformer/](../src/etapa3_transformer/).
