# Changelog 05 — BPE, SFT, Weight Tying, Dropout, Confiança e `backup/` (`9986c34`)

- **Commit**: `9986c34`
- **Data**: 2026-11-26
- **Mensagem**: `Evolui Mini-Transformer (BPE, SFT, Weight Tying, Dropout, confianca), arquiva etapas antigas em backup/ e melhora datasets`
- **Escopo**: 18 arquivos alterados (`+1.246` / `-397` linhas)

## Motivação
Na versão caractere-a-caractere do Mini-Transformer:
1. A janela de 64 caracteres comportava apenas ~10 palavras, fazendo o modelo gastar capacidade soletrando e inventando palavras quando a perda era alta.
2. A perda era calculada também sobre o prompt do usuário (`Usuario:`), desperdiçando gradientes.
3. Perguntas fora do domínio ou mudanças de assunto com histórico acumulado faziam o modelo "bugar" e gerar texto quebrado.

## Como o projeto evoluiu

### 1. Tokenizador Byte-Level BPE do Zero (`src/comum/bpe.hpp`)
- Implementado BPE baseado nos 256 bytes (estilo GPT-2), sem tokens desconhecidos e 100% reversível para UTF-8 com acentos.
- Comprimiu o corpus grande de **4,61M** para **2,22M tokens** (**>2x** mais contexto efetivo na mesma janela).
- Persistência binária automática (`.bpe`, cabeçalho `VBBPE01`) salva e carregada junto com o checkpoint do modelo, coberta por testes unitários em `tests/teste_comum.cpp`.

### 2. Evolução do Mini-Transformer (`src/etapa3_transformer/transformer.hpp` e `main.cpp`)
- **SFT (*Supervised Fine-Tuning*, `--sft`)**: máscara de loss que aplica peso `0.0` nos tokens do `Usuario:` e `1.0` apenas nas respostas após `IA:`.
- **Weight Tying**: compartilha `w_tok` com a projeção de saída (*LM Head*), economizando $V \times D$ parâmetros (~49 mil pesos em `V=512, D=96`) e validado no *gradient check* em `tests/teste_transformer.cpp`.
- **Inverted Dropout (`--dropout`)**: aplicado nas saídas da Atenção e da FFN durante o treino (desligado em avaliação/chat), com RNG independente por thread OpenMP.
- **Checkpoint `VBGPT02` + `--continuar`**: salva e restaura o estado completo do AdamW (`m`, `v`, `passo`), permitindo retomar treinos em sessões incrementais sem perder a inércia da otimização.
- **Melhor Validação (`--melhor-val`) e Early Stopping (`--paciencia`)**: salva automaticamente o checkpoint de menor perda de validação.
- **Vetorização nativa**: adicionado `-march=native` ao build Release no `CMakeLists.txt`.

### 3. Chat com Indicador de Confiança e Fallback Inteligente
- `gerar_com_confianca` calcula a média geométrica das probabilidades reais (`temperatura=1.0`) e exibe `IA (GPT | conf XX%)`.
- Temperatura padrão ajustada para `0.25` e `top_k = 4`, eliminando quase totalmente palavras inventadas.
- Se a confiança com o histórico acumulado cair abaixo do limiar (`72%`, ajustável via `/conf`), o chat tenta automaticamente responder usando apenas a pergunta atual (mudança de assunto); se ainda assim ficar abaixo do limiar, informa honestamente que está fora do domínio em vez de gerar texto quebrado.

### 4. Dados Tratados e Foco na Versão Atual (`backup/`)
- Expandido `dados/conversa_cotidiana.txt` com variações naturais de conversa (com/sem acento, piadas, saudações e despedidas).
- Adicionado filtro de corpus calibrado para CPU em `scripts/baixar_dados.py` (`dados/externos/conversas_cpu.txt`, ~232 KB de diálogos curtos e limpos).
- Etapas educativas anteriores (`etapa1_bigram`, `etapa2_mlp`, `teste_mlp.cpp`) foram movidas para `backup/` com `backup/README.md`, deixando `CMakeLists.txt` e `Makefile` limpos e focados no Mini-Transformer atual (`4/4` testes `CTest` passando em ~3.8s).
