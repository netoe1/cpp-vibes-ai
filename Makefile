# =============================================================================
# cpp-vibes-ai — Makefile para compilar, testar e conversar com os modelos
# =============================================================================
# Uso basico:
#   make help          -> Mostra todos os comandos disponiveis
#   make build         -> Compila todos os modelos e testes (Release + OpenMP)
#   make test          -> Executa toda a suite de testes e regressao (CTest)
#   make etapa1        -> Roda o chat da Etapa 1 (Bigram)
#   make etapa2        -> Roda o chat da Etapa 2 (MLP)
#   make etapa3        -> Roda o chat da Etapa 3 (Mini-Transformer / GPT)
#   make chat-gpt      -> Treina (se necessario) e abre chat rapido com checkpoint
#   make clean         -> Remove arquivos compilados (build/)
#
# Voce pode sobrescrever variaveis na linha de comando, por exemplo:
#   make treinar-gpt PASSOS=1000
#   make etapa3 DADOS=dados/dialogos.txt
# =============================================================================

BUILD_DIR   ?= build
TIPO        ?= Release
DADOS       ?= dados/dialogos.txt
EPOCAS      ?= 80
PASSOS      ?= 600
CKPT_MLP          ?= $(BUILD_DIR)/modelo_mlp.vibes
CKPT_MLP_COMPLETO ?= $(BUILD_DIR)/modelo_mlp_completo.vibes
CKPT_GPT          ?= $(BUILD_DIR)/modelo_gpt.vibes
CKPT_GPT_COMPLETO ?= $(BUILD_DIR)/modelo_gpt_completo.vibes
CKPT_GPT_GRANDE   ?= $(BUILD_DIR)/modelo_gpt_grande.vibes
DADOS_FONTES := dados/dialogos.txt dados/conversa_cotidiana.txt dados/programacao_cpp.txt dados/machine_learning.txt

.DEFAULT_GOAL := help

.PHONY: help configurar build test clean \
        etapa1 etapa2 etapa3 \
        treinar-mlp chat-mlp \
        treinar-gpt chat-gpt \
        dados-completo treinar-gpt-completo chat-gpt-completo \
        baixar-dados treinar-gpt-grande chat-gpt-grande \
        suite-treino suite-treino-rapida suite-treino-grande suite-benchmark

help:
	@echo "=============================================================="
	@echo "  cpp-vibes-ai — Comandos disponiveis (Makefile)"
	@echo "=============================================================="
	@echo "  [Build e Testes]"
	@echo "  make build                Compila todas as etapas e testes ($(TIPO))"
	@echo "  make test                 Executa todos os testes e regressao (CTest)"
	@echo "  make clean                Remove o diretorio $(BUILD_DIR)/"
	@echo ""
	@echo "  [Suite de Treinos]"
	@echo "  make suite-treino         Treina Etapas 1, 2 e 3 em dados/completo.txt"
	@echo "  make suite-treino-rapida  Treino rapido (smoke) das 3 etapas em completo.txt"
	@echo "  make suite-treino-grande  Treina GPT no corpus grande (OASST1 + Alpaca-PT)"
	@echo "  make suite-benchmark      Compara tempo e perda das 3 etapas lado a lado"
	@echo ""
	@echo "  [Chat Interativo e Treinos Individuais]"
	@echo "  make etapa1               Treina e abre chat da Etapa 1 (Bigram)"
	@echo "  make etapa2               Treina e abre chat da Etapa 2 (MLP)"
	@echo "  make etapa3               Treina e abre chat da Etapa 3 (Transformer)"
	@echo "  make treinar-mlp          Treina a MLP e salva em $(CKPT_MLP)"
	@echo "  make chat-mlp             Abre chat da MLP usando checkpoint salvo"
	@echo "  make treinar-gpt          Treina o GPT ($(PASSOS) passos) e salva checkpoint"
	@echo "  make chat-gpt             Abre chat do GPT usando checkpoint salvo"
	@echo "  make dados-completo       Junta datasets locais em dados/completo.txt"
	@echo "  make treinar-gpt-completo Treina o GPT em dados/completo.txt"
	@echo "  make chat-gpt-completo    Abre chat do GPT treinado em dados/completo.txt"
	@echo "  make baixar-dados         Baixa grandes datasets PT (OASST1, Alpaca-PT)"
	@echo "  make treinar-gpt-grande   Treina o GPT em dados/externos/conversas_grande.txt"
	@echo "  make chat-gpt-grande      Abre chat do GPT treinado no corpus grande"
	@echo ""
	@echo "Variaveis opcionais: DADOS=... EPOCAS=... PASSOS=... TIPO=Release|Debug"

configurar:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(TIPO)

build: configurar
	@cmake --build $(BUILD_DIR) -j

test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

etapa1: build
	@./$(BUILD_DIR)/etapa1_bigram $(DADOS)

etapa2: build
	@./$(BUILD_DIR)/etapa2_mlp $(DADOS) --epocas $(EPOCAS)

etapa3: build
	@./$(BUILD_DIR)/etapa3_transformer $(DADOS) --passos $(PASSOS)

treinar-mlp: build
	@./$(BUILD_DIR)/etapa2_mlp $(DADOS) --sem-chat --epocas $(EPOCAS) --salvar $(CKPT_MLP)

$(CKPT_MLP):
	@$(MAKE) treinar-mlp

chat-mlp: build $(CKPT_MLP)
	@./$(BUILD_DIR)/etapa2_mlp $(DADOS) --carregar $(CKPT_MLP)

treinar-gpt: build
	@./$(BUILD_DIR)/etapa3_transformer $(DADOS) --sem-chat --passos $(PASSOS) --salvar $(CKPT_GPT)

$(CKPT_GPT):
	@$(MAKE) treinar-gpt

chat-gpt: build $(CKPT_GPT)
	@./$(BUILD_DIR)/etapa3_transformer $(DADOS) --carregar $(CKPT_GPT)

dados/completo.txt: $(DADOS_FONTES)
	@awk 'FNR==1 && NR!=1 {print ""} {print}' $(DADOS_FONTES) > dados/completo.txt
	@echo "Dataset combinado gerado em dados/completo.txt ($$(wc -l < dados/completo.txt) linhas, $$(wc -c < dados/completo.txt) bytes)."

dados-completo: dados/completo.txt

treinar-gpt-completo: build dados/completo.txt
	@./$(BUILD_DIR)/etapa3_transformer dados/completo.txt --sem-chat --passos $(PASSOS) --salvar $(CKPT_GPT_COMPLETO)

$(CKPT_GPT_COMPLETO):
	@$(MAKE) treinar-gpt-completo

chat-gpt-completo: build dados/completo.txt $(CKPT_GPT_COMPLETO)
	@./$(BUILD_DIR)/etapa3_transformer dados/completo.txt --carregar $(CKPT_GPT_COMPLETO)

baixar-dados:
	@python3 scripts/baixar_dados.py --dataset todos

dados/externos/conversas_grande.txt:
	@$(MAKE) baixar-dados

treinar-gpt-grande: build dados/externos/conversas_grande.txt
	@./$(BUILD_DIR)/etapa3_transformer dados/externos/conversas_grande.txt --sem-chat --passos $(PASSOS) --salvar $(CKPT_GPT_GRANDE)

$(CKPT_GPT_GRANDE):
	@$(MAKE) treinar-gpt-grande

chat-gpt-grande: build dados/externos/conversas_grande.txt $(CKPT_GPT_GRANDE)
	@./$(BUILD_DIR)/etapa3_transformer dados/externos/conversas_grande.txt --carregar $(CKPT_GPT_GRANDE)

# -----------------------------------------------------------------------------
# Suite de Treinos (comparacao e treinamento em lote)
# -----------------------------------------------------------------------------
suite-treino-rapida: build dados/completo.txt
	@echo "=== [Suite Rapida 1/3] Etapa 1: Bigram (dados/completo.txt) ==="
	@./$(BUILD_DIR)/etapa1_bigram dados/completo.txt --sem-chat
	@echo "=== [Suite Rapida 2/3] Etapa 2: MLP (15 epocas, dados/completo.txt) ==="
	@./$(BUILD_DIR)/etapa2_mlp dados/completo.txt --sem-chat --epocas 15
	@echo "=== [Suite Rapida 3/3] Etapa 3: Mini-Transformer (60 passos, dados/completo.txt) ==="
	@./$(BUILD_DIR)/etapa3_transformer dados/completo.txt --sem-chat --passos 60
	@echo "=== Suite rapida concluida com sucesso! ==="

suite-treino: build dados/completo.txt
	@echo "=== [Suite Completa 1/3] Etapa 1: Bigram (dados/completo.txt) ==="
	@./$(BUILD_DIR)/etapa1_bigram dados/completo.txt --sem-chat
	@echo "=== [Suite Completa 2/3] Etapa 2: MLP ($(EPOCAS) epocas -> $(CKPT_MLP_COMPLETO)) ==="
	@./$(BUILD_DIR)/etapa2_mlp dados/completo.txt --sem-chat --epocas $(EPOCAS) --salvar $(CKPT_MLP_COMPLETO)
	@echo "=== [Suite Completa 3/3] Etapa 3: Mini-Transformer ($(PASSOS) passos -> $(CKPT_GPT_COMPLETO)) ==="
	@./$(BUILD_DIR)/etapa3_transformer dados/completo.txt --sem-chat --passos $(PASSOS) --salvar $(CKPT_GPT_COMPLETO)
	@echo "=== Suite completa concluida! Use 'make chat-gpt-completo' para conversar. ==="

suite-treino-grande: build dados/externos/conversas_grande.txt
	@echo "=== [Suite Grande 1/2] Etapa 1: Bigram (conversas_grande.txt) ==="
	@./$(BUILD_DIR)/etapa1_bigram dados/externos/conversas_grande.txt --sem-chat
	@echo "=== [Suite Grande 2/2] Etapa 3: Mini-Transformer ($(PASSOS) passos -> $(CKPT_GPT_GRANDE)) ==="
	@./$(BUILD_DIR)/etapa3_transformer dados/externos/conversas_grande.txt --sem-chat --passos $(PASSOS) --salvar $(CKPT_GPT_GRANDE)
	@echo "=== Suite grande concluida! Use 'make chat-gpt-grande' para conversar. ==="

suite-benchmark: build
	@echo "=== Benchmark comparativo em $(DADOS) ==="
	@echo "--- Etapa 1 (Bigram) ---"
	@/usr/bin/time -f "Tempo Etapa 1: %es" ./$(BUILD_DIR)/etapa1_bigram $(DADOS) --sem-chat | grep -E "Parametros|Resultado|Perda otima"
	@echo "--- Etapa 2 (MLP, 40 epocas) ---"
	@/usr/bin/time -f "Tempo Etapa 2: %es" ./$(BUILD_DIR)/etapa2_mlp $(DADOS) --sem-chat --epocas 40 | grep -E "Parametros|Epoca  40|Perda inicial"
	@echo "--- Etapa 3 (Mini-Transformer, 100 passos) ---"
	@/usr/bin/time -f "Tempo Etapa 3: %es" ./$(BUILD_DIR)/etapa3_transformer $(DADOS) --sem-chat --passos 100 | grep -E "Parametros|Passo  100|Perda inicial"

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Diretorio $(BUILD_DIR)/ removido."
