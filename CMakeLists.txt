cmake_minimum_required(VERSION 3.20)
project(cpp_vibes_ai
    VERSION 0.4.0
    DESCRIPTION "IA conversacional educativa (Mini-Transformer GPT) construida do zero em C++"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Exporta compile_commands.json para facilitar autocomplete em IDEs/editores
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenMP para paralelismo em CPU
find_package(OpenMP)

# Opcoes de compilacao: avisos estritos + otimizacao nativa em Release
add_library(config_projeto INTERFACE)
target_include_directories(config_projeto INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(config_projeto INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CONFIG:Release>:-O3 -march=native>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(config_projeto INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(config_projeto INTERFACE USAR_OPENMP=1)
    message(STATUS "OpenMP encontrado: paralelismo em CPU ativado.")
else()
    message(WARNING "OpenMP nao encontrado: compilando em modo single-thread.")
endif()

# -----------------------------------------------------------------------------
# Modelo Atual: Mini-Transformer Conversacional (GPT + BPE + SFT + Weight Tying)
# (As etapas educativas anteriores estao preservadas em backup/)
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

add_executable(teste_transformer tests/teste_transformer.cpp)
target_link_libraries(teste_transformer PRIVATE config_projeto)
add_test(NAME unitario_e_gradcheck_transformer COMMAND teste_transformer)

add_test(
    NAME integracao_transformer_utf8
    COMMAND etapa3_transformer ${CMAKE_CURRENT_SOURCE_DIR}/dados/dialogos.txt --sem-chat --verificar-gradiente --passos 40
)

add_test(
    NAME integracao_transformer_bpe_sft
    COMMAND etapa3_transformer ${CMAKE_CURRENT_SOURCE_DIR}/dados/dialogos.txt --sem-chat --bpe 300 --sft --passos 30
)
