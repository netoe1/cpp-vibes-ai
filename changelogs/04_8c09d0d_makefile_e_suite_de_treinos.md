# Changelog 04 — Makefile e Suíte de Treinos (`8c09d0d`)

- **Commit**: `8c09d0d`
- **Data**: 2026-11-26
- **Mensagem**: `Adiciona Makefile com suite de treinos e atualiza documentacao e specs`
- **Escopo**: 4 arquivos alterados (`+252` / `-24` linhas)

## Motivação
Com 3 etapas, vários datasets e múltiplas flags de linha de comando (`--sem-chat`, `--salvar`, `--carregar`, `--passos`, `--epocas`), testar e comparar modelos manualmente ficava repetitivo.

## O que mudou
- **Criado `Makefile`** na raiz (e removido `Makefile` do `.gitignore` para versionamento):
  - Atalhos de build e validação: `make help`, `make build`, `make test`, `make clean`.
  - Atalhos de chat e treino por etapa: `make etapa1`, `make etapa2`, `make etapa3`, `make treinar-mlp`, `make chat-mlp`, `make treinar-gpt`, `make chat-gpt`.
  - Integração de dados: `make dados-completo` (combina datasets locais em `dados/completo.txt`) e `make baixar-dados` / `make treinar-gpt-grande` / `make chat-gpt-grande`.
  - **Suíte de Treinos**: `make suite-treino-rapida`, `make suite-treino`, `make suite-treino-grande` e `make suite-benchmark` (medição de tempo e perda das 3 arquiteturas lado a lado).
- **Documentação sincronizada**: atualizados `README.md`, `.spec/01_arquitetura_e_convencoes.md` e `.spec/03_status_e_decisoes.md` com todos os comandos e variáveis sobrescrevíveis (`DADOS`, `PASSOS`, `EPOCAS`, `TIPO`).
