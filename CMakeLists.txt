cmake_minimum_required(VERSION 3.20)
project(cpp_vibes_ai
    VERSION 0.3.0
    DESCRIPTION "IA conversacional educativa construida do zero em C++"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Exporta compile_commands.json para facilitar autocomplete em IDEs/editores
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenMP para paralelismo simples em CPU
find_package(OpenMP)

# Opcoes de compilacao: avisos estritos e otimizacao em Release
add_library(config_projeto INTERFACE)
target_include_directories(config_projeto INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(config_projeto INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CONFIG:Release>:-O3>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(config_projeto INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(config_projeto INTERFACE USAR_OPENMP=1)
    message(STATUS "OpenMP encontrado: paralelismo em CPU ativado.")
else()
    message(WARNING "OpenMP nao encontrado: compilando em modo single-thread.")
endif()

# -----------------------------------------------------------------------------
# Etapa 1: Modelo Bigram (Estatistico vs Neural com Gradiente Descendente)
# -----------------------------------------------------------------------------
add_executable(etapa1_bigram
    src/etapa1_bigram/main.cpp
)
target_link_libraries(etapa1_bigram PRIVATE config_projeto)

# -----------------------------------------------------------------------------
# Etapa 2: Modelo MLP (Embeddings + Janela de Contexto + Backprop Manual)
# -----------------------------------------------------------------------------
add_executable(etapa2_mlp
    src/etapa2_mlp/main.cpp
)
target_link_libraries(etapa2_mlp PRIVATE config_projeto)

# -----------------------------------------------------------------------------
# Etapa 3: Mini-Transformer Conversacional (Causal Self-Attention + AdamW)
# -----------------------------------------------------------------------------
add_executable(etapa3_transformer
    src/etapa3_transformer/main.cpp
)
target_link_libraries(etapa3_transformer PRIVATE config_projeto)

# -----------------------------------------------------------------------------
# Testes Automatizados e de Regressao (CTest)
# -----------------------------------------------------------------------------
enable_testing()

add_executable(teste_comum tests/teste_comum.cpp)
target_link_libraries(teste_comum PRIVATE config_projeto)
add_test(NAME unitario_comum COMMAND teste_comum)

add_executable(teste_mlp tests/teste_mlp.cpp)
target_link_libraries(teste_mlp PRIVATE config_projeto)
add_test(NAME unitario_e_gradcheck_mlp COMMAND teste_mlp)

add_executable(teste_transformer tests/teste_transformer.cpp)
target_link_libraries(teste_transformer PRIVATE config_projeto)
add_test(NAME unitario_e_gradcheck_transformer COMMAND teste_transformer)

add_test(
    NAME regressao_etapa1_bigram
    COMMAND etapa1_bigram ${CMAKE_CURRENT_SOURCE_DIR}/dados/dialogos.txt --sem-chat
)

add_test(
    NAME regressao_etapa2_mlp
    COMMAND etapa2_mlp ${CMAKE_CURRENT_SOURCE_DIR}/dados/dialogos.txt --sem-chat --verificar-gradiente --epocas 40
)

add_test(
    NAME integracao_etapa3_transformer
    COMMAND etapa3_transformer ${CMAKE_CURRENT_SOURCE_DIR}/dados/dialogos.txt --sem-chat --verificar-gradiente --passos 60
)
