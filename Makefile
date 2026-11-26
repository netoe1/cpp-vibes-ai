# =============================================================================
# cpp-vibes-ai — Makefile (Mini-Transformer GPT + BPE + SFT)
# =============================================================================
# Uso basico:
#   make help              -> Mostra todos os comandos disponiveis
#   make build             -> Compila o Mini-Transformer e os testes (Release)
#   make test              -> Roda toda a suite de testes e regressao (CTest)
#   make treinar           -> Treina no corpus curado (dados/completo.txt)
#   make continuar         -> Retoma treino do checkpoint preservando AdamW
#   make chat              -> Abre o chat com o modelo treinado
#   make baixar-dados      -> Baixa grandes datasets PT (OASST1 + Alpaca-PT)
#   make treinar-grande    -> Treina no corpus grande (4,5 MB)
#   make continuar-grande  -> Retoma treino no corpus grande
#   make chat-grande       -> Abre o chat com o modelo do corpus grande
#   make clean             -> Remove build/
# =============================================================================

BUILD_DIR    ?= build
TIPO         ?= Release
BIN          := $(BUILD_DIR)/etapa3_transformer

# Hiperparametros padrao para treino rapido e coerente em dados/completo.txt
DADOS        ?= dados/completo.txt
VOCAB_BPE    ?= 640
DIM          ?= 128
CAMADAS      ?= 3
CABECAS      ?= 4
CONTEXTO     ?= 80
BATCH        ?= 16
LR           ?= 0.004
PASSOS       ?= 900
CKPT         ?= $(BUILD_DIR)/modelo.vibes

# Hiperparametros padrao para o corpus otimizado para CPU (~250 KB, ~1300 dialogos limpos)
DADOS_CPU       ?= dados/externos/conversas_cpu.txt
VOCAB_CPU       ?= 512
DIM_CPU         ?= 96
CAMADAS_CPU     ?= 3
CONTEXTO_CPU    ?= 64
PASSOS_CPU      ?= 1200
LR_CPU          ?= 0.004
CKPT_CPU        ?= $(BUILD_DIR)/modelo_cpu.vibes

# Hiperparametros padrao para o corpus grande (dados/externos/conversas_grande.txt)
VOCAB_GRANDE    ?= 512
DIM_GRANDE      ?= 96
CAMADAS_GRANDE  ?= 3
CONTEXTO_GRANDE ?= 96
PASSOS_GRANDE   ?= 800
LR_GRANDE       ?= 0.003
CKPT_GRANDE     ?= $(BUILD_DIR)/modelo_gpt_bpe.vibes

DADOS_FONTES := dados/dialogos.txt dados/conversa_cotidiana.txt dados/programacao_cpp.txt dados/machine_learning.txt dados/filosofia.txt dados/computacao.txt dados/musica_rock_metal.txt

.DEFAULT_GOAL := help

.PHONY: help configurar build test clean \
        dados-completo treinar continuar chat \
        treinar-cpu continuar-cpu chat-cpu \
        baixar-dados treinar-grande continuar-grande chat-grande \
        suite-treino

help:
	@echo "=============================================================="
	@echo "  cpp-vibes-ai — Mini-Transformer Conversacional (Makefile)"
	@echo "=============================================================="
	@echo "  [Build e Testes]"
	@echo "  make build             Compila o modelo e os testes ($(TIPO) + OpenMP)"
	@echo "  make test              Executa testes unitarios, gradcheck e integracao"
	@echo "  make clean             Remove o diretorio $(BUILD_DIR)/"
	@echo ""
	@echo "  [Treino e Chat — Corpus Curado (dados/completo.txt)]"
	@echo "  make dados-completo    Gera dados/completo.txt juntando os datasets locais"
	@echo "  make treinar           Treina GPT (BPE $(VOCAB_BPE) + SFT, $(PASSOS) passos)"
	@echo "  make continuar         Continua treino de $(CKPT) preservando AdamW"
	@echo "  make chat              Abre chat carregando $(CKPT)"
	@echo ""
	@echo "  [Treino e Chat — Corpus Otimizado para CPU (~250 KB, recomendado!)]"
	@echo "  make treinar-cpu       Treina com Dropout + Melhor Validacao em $(DADOS_CPU)"
	@echo "  make continuar-cpu     Continua treino de $(CKPT_CPU)"
	@echo "  make chat-cpu          Abre chat carregando $(CKPT_CPU) (com indicador de confianca)"
	@echo ""
	@echo "  [Treino e Chat — Corpus Grande (OASST1 + Alpaca-PT, 4.5 MB)]"
	@echo "  make baixar-dados      Baixa e formata os grandes datasets em PT"
	@echo "  make treinar-grande    Treina GPT (BPE $(VOCAB_GRANDE) + SFT, $(PASSOS_GRANDE) passos)"
	@echo "  make continuar-grande  Continua treino de $(CKPT_GRANDE)"
	@echo "  make chat-grande       Abre chat carregando $(CKPT_GRANDE)"
	@echo ""
	@echo "  [Suite Completa]"
	@echo "  make suite-treino      Roda testes + treino no corpus curado"
	@echo ""
	@echo "Variaveis opcionais: DADOS=... PASSOS=... DIM=... CAMADAS=... CONTEXTO=... LR=... VOCAB_BPE=..."

configurar:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(TIPO)

build: configurar
	@cmake --build $(BUILD_DIR) -j

test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

dados/completo.txt: $(DADOS_FONTES)
	@awk 'FNR==1 && NR!=1 {print ""} {print}' $(DADOS_FONTES) > dados/completo.txt
	@echo "Dataset combinado gerado em dados/completo.txt ($$(wc -l < dados/completo.txt) linhas)."

dados-completo: dados/completo.txt

treinar: build dados/completo.txt
	@./$(BIN) $(DADOS) --sem-chat \
		--bpe $(VOCAB_BPE) --sft --dropout 0.05 \
		--dim $(DIM) --camadas $(CAMADAS) --cabecas $(CABECAS) --contexto $(CONTEXTO) \
		--batch $(BATCH) --lr $(LR) --passos $(PASSOS) \
		--salvar $(CKPT)

continuar: build dados/completo.txt $(CKPT)
	@./$(BIN) $(DADOS) --sem-chat \
		--bpe $(VOCAB_BPE) --sft --dropout 0.05 \
		--dim $(DIM) --camadas $(CAMADAS) --cabecas $(CABECAS) --contexto $(CONTEXTO) \
		--batch $(BATCH) --lr $(LR) --passos $(PASSOS) \
		--continuar $(CKPT) --salvar $(CKPT)

$(CKPT):
	@$(MAKE) treinar

chat: build dados/completo.txt $(CKPT)
	@./$(BIN) $(DADOS) \
		--bpe $(VOCAB_BPE) \
		--dim $(DIM) --camadas $(CAMADAS) --cabecas $(CABECAS) --contexto $(CONTEXTO) \
		--carregar $(CKPT)

treinar-cpu: build $(DADOS_CPU)
	@./$(BIN) $(DADOS_CPU) --sem-chat \
		--bpe $(VOCAB_CPU) --sft --dropout 0.1 --melhor-val \
		--dim $(DIM_CPU) --camadas $(CAMADAS_CPU) --cabecas $(CABECAS) --contexto $(CONTEXTO_CPU) \
		--batch $(BATCH) --lr $(LR_CPU) --passos $(PASSOS_CPU) \
		--salvar $(CKPT_CPU)

continuar-cpu: build $(DADOS_CPU) $(CKPT_CPU)
	@./$(BIN) $(DADOS_CPU) --sem-chat \
		--bpe $(VOCAB_CPU) --sft --dropout 0.1 --melhor-val \
		--dim $(DIM_CPU) --camadas $(CAMADAS_CPU) --cabecas $(CABECAS) --contexto $(CONTEXTO_CPU) \
		--batch $(BATCH) --lr $(LR_CPU) --passos $(PASSOS_CPU) \
		--continuar $(CKPT_CPU) --salvar $(CKPT_CPU)

$(CKPT_CPU):
	@$(MAKE) treinar-cpu

chat-cpu: build $(DADOS_CPU) $(CKPT_CPU)
	@./$(BIN) $(DADOS_CPU) \
		--bpe $(VOCAB_CPU) \
		--dim $(DIM_CPU) --camadas $(CAMADAS_CPU) --cabecas $(CABECAS) --contexto $(CONTEXTO_CPU) \
		--carregar $(CKPT_CPU)

baixar-dados:
	@python3 scripts/baixar_dados.py --dataset todos

dados/externos/conversas_grande.txt:
	@$(MAKE) baixar-dados

treinar-grande: build dados/externos/conversas_grande.txt
	@./$(BIN) dados/externos/conversas_grande.txt --sem-chat \
		--bpe $(VOCAB_GRANDE) --sft \
		--dim $(DIM_GRANDE) --camadas $(CAMADAS_GRANDE) --cabecas $(CABECAS) --contexto $(CONTEXTO_GRANDE) \
		--batch $(BATCH) --lr $(LR_GRANDE) --passos $(PASSOS_GRANDE) \
		--salvar $(CKPT_GRANDE)

continuar-grande: build dados/externos/conversas_grande.txt $(CKPT_GRANDE)
	@./$(BIN) dados/externos/conversas_grande.txt --sem-chat \
		--bpe $(VOCAB_GRANDE) --sft \
		--dim $(DIM_GRANDE) --camadas $(CAMADAS_GRANDE) --cabecas $(CABECAS) --contexto $(CONTEXTO_GRANDE) \
		--batch $(BATCH) --lr $(LR_GRANDE) --passos $(PASSOS_GRANDE) \
		--continuar $(CKPT_GRANDE) --salvar $(CKPT_GRANDE)

$(CKPT_GRANDE):
	@$(MAKE) treinar-grande

chat-grande: build dados/externos/conversas_grande.txt $(CKPT_GRANDE)
	@./$(BIN) dados/externos/conversas_grande.txt \
		--bpe $(VOCAB_GRANDE) \
		--dim $(DIM_GRANDE) --camadas $(CAMADAS_GRANDE) --cabecas $(CABECAS) --contexto $(CONTEXTO_GRANDE) \
		--carregar $(CKPT_GRANDE)

suite-treino: test treinar
	@echo "=== Suite de testes e treino concluida! Rode 'make chat' para conversar. ==="

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Diretorio $(BUILD_DIR)/ removido."
