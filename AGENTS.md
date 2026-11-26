# AGENTS.md — Diretrizes de Trabalho para Agentes

Este arquivo define as regras obrigatórias para qualquer agente atuando no repositório **cpp-vibes-ai**. **Consulte este arquivo e a pasta `.spec/` antes de iniciar qualquer tarefa.**

---

## 1. Fonte de Verdade e Especificações (`.spec/`)

Antes de propor alterações ou escrever código, consulte os arquivos em `.spec/`:
- `.spec/00_visao_e_requisitos.md` — objetivos educativos, requisitos funcionais e não-funcionais.
- `.spec/01_arquitetura_e_convencoes.md` — estrutura do projeto, convenções C++20/OpenMP e comandos de build.
- `.spec/02_plano_de_etapas.md` — especificações técnicas e critérios de aceitação de cada etapa.
- `.spec/03_status_e_decisoes.md` — status atual, checklist e decisões arquiteturais (ADRs).

**Manutenção obrigatória**: sempre que uma feature, etapa ou decisão técnica for concluída ou alterada, atualize `.spec/03_status_e_decisoes.md` (e as demais specs impactadas).

---

## 2. Simulação Antes de Codificar (Pre-Code Simulation)

**Nunca saia escrevendo código diretamente para uma nova feature ou correção complexa.** Antes de implementar, simule o comportamento esperado passo a passo:

1. **Rastreio de dimensões (*shapes*) e dados**: trace um exemplo mínimo na mão (ex.: batch = 2, contexto = 4, vocabulário = 5) verificando as dimensões de entrada, camadas intermediárias, saída e gradientes (`forward` e `backward`).
2. **Fluxo de execução e invariantes**: simule o comportamento de ponta a ponta (leitura de arquivo -> tokenização -> treino -> cálculo da perda -> atualização de pesos -> geração/chat).
3. **Casos de borda e estabilidade numérica**: antecipe situações críticas antes de codificar:
   - Caracteres UTF-8 multibyte, texto vazio ou menor que a janela de contexto.
   - *Overflow/underflow* em `exp()` e `log(0)` (`softmax`, entropia cruzada).
   - Condições de corrida (*race conditions*) e compartilhamento de buffers em blocos OpenMP.
   - Inicialização de pesos e explosão/desaparecimento de gradientes.
4. **Critério de sucesso mensurável**: defina qual resultado numérico ou comportamento indica que o código está certo (ex.: perda menor que referência, erro relativo de *gradient check* `< 1e-3`).

Essa simulação prévia reduz drasticamente bugs silenciosos de ML (onde o programa compila e roda, mas aprende errado) e facilita o debug.

---

## 3. Testes Automatizados e de Regressão

Ao finalizar **qualquer feature, correção ou etapa**, é obrigatório criar e executar testes automatizados antes de considerar a entrega concluída:

### 3.1. Testes da Nova Feature
- **Testes unitários** (`tests/` integrados via `CTest` no `CMakeLists.txt`): cobrir novas funções em `src/comum/` e componentes da etapa (tokenização UTF-8, `softmax`, multiplicação de matrizes, embeddings, camadas, serialização).
- **Verificação numérica de gradiente (*Gradient Checking*)**: obrigatória para toda camada com *backpropagation* manual (MLP, Attention, LayerNorm), comparando o gradiente analítico com diferenças finitas.
- **Teste de integração / fumaça (*smoke test*)**: executar o binário da etapa com `--sem-chat` e validar que a perda diminui e converge conforme o esperado.

### 3.2. Testes de Regressão
- Sempre que alterar código compartilhado (`src/comum/`, `CMakeLists.txt`, `dados/`) ou finalizar uma nova etapa:
  1. Recompile **todos** os alvos em modo Release com avisos estritos (`-Wall -Wextra -Wpedantic`) e garanta **zero warnings**.
  2. Execute **toda a suíte de testes** (`ctest --test-dir build --output-on-failure`), incluindo os testes das etapas anteriores.
  3. Confirme que nenhuma etapa anterior quebrou ou regrediu em convergência/determinismo.

---

## 4. Changelogs e Commits Organizados

Sempre que finalizar um marco, nova feature ou evolução relevante do projeto:

1. **Atualize `.spec/`**: reflita o novo status e eventuais decisões arquiteturais (ADRs) em `.spec/03_status_e_decisoes.md` (e demais specs impactadas).
2. **Crie um novo changelog em `changelogs/`**:
   - Use numeração sequencial: `changelogs/NN_<resumo_curto>.md` (ex.: `06_kv_cache_datasets_e_temas.md`).
   - Documente **motivação**, **o que mudou**, **como o modelo evoluiu** (arquitetura, dados, testes, métricas observadas) e **impacto prático**.
   - Adicione a nova entrada na tabela e no diagrama de `changelogs/README.md`.
3. **Valide antes do commit**: execute `make test` (ou `ctest --test-dir build --output-on-failure`) e garanta compilação limpa sem avisos (`-Wall -Wextra -Wpedantic`).
4. **Commit e Push**: faça commits com mensagens claras e descritivas em português, sem incluir artefatos temporários (`build/`, `dados/externos/`, `dados/completo.txt`).

---

## 5. Regras Gerais do Repositório

- **100% em português**: identificadores, comentários, mensagens de terminal, `docs/` e `.spec/`.
- **C++20 puro + OpenMP**: zero dependências externas de Machine Learning; manter fallback funcional caso OpenMP não esteja disponível.
- **Didática primeiro**: toda nova etapa deve vir acompanhada de sua explicação teórica em `docs/` e comentários claros nos trechos matemáticos críticos.
