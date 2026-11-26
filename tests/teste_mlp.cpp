#include "etapa2_mlp/mlp.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

int falhas = 0;

void verificar(bool condicao, const std::string& nome) {
    if (!condicao) {
        std::cerr << "[FALHA] " << nome << "\n";
        ++falhas;
    } else {
        std::cout << "[OK] " << nome << "\n";
    }
}

void testar_gradient_check_mlp() {
    vibes::ConfigMLP cfg;
    cfg.vocab = 6;
    cfg.contexto = 4;
    cfg.dim_emb = 3;
    cfg.dim_oculta = 8;

    vibes::GeradorAleatorio rng(7);
    vibes::ModeloMLP modelo(cfg, rng);

    // Inclui tokens repetidos no mesmo contexto para testar scatter-add em dC
    const std::vector<vibes::ExemploContexto> exemplos = {
        {{1, 1, 2, 3}, 4},
        {{0, 2, 2, 5}, 1},
        {{3, 4, 5, 0}, 2},
    };

    const float erro = vibes::verificar_gradiente_numerico(modelo, exemplos, 64, 1e-5);
    std::cout << "  -> Maior erro relativo no Gradient Check: " << erro << "\n";
    verificar(erro < 1e-4f, "Gradient checking numerico da MLP abaixo de 1e-4");
}

void testar_convergencia_e_persistencia() {
    const std::string mini_corpus =
        "Usuario: ola\nIA: ola tudo bem\n"
        "Usuario: oi\nIA: oi tudo otimo\n"
        "Usuario: ola\nIA: ola tudo bem\n";

    vibes::TokenizadorUTF8 tok;
    tok.construir_vocabulario(mini_corpus);
    const auto tokens = tok.codificar(mini_corpus);
    const int id_pad = tok.obter_id("\n");

    vibes::ConfigMLP cfg;
    cfg.vocab = tok.tamanho_vocabulario();
    cfg.contexto = 8;
    cfg.dim_emb = 12;
    cfg.dim_oculta = 32;

    const auto exemplos = vibes::construir_exemplos(tokens, cfg.contexto, id_pad);
    std::vector<std::size_t> indices(exemplos.size());
    std::iota(indices.begin(), indices.end(), 0);

    vibes::GeradorAleatorio rng(42);
    vibes::ModeloMLP modelo(cfg, rng);
    vibes::GradientesMLP grad(cfg);

    const float perda_inicial = modelo.avaliar(exemplos);
    for (int epoca = 0; epoca < 120; ++epoca) {
        modelo.calcular_perda_e_gradiente(exemplos, indices, grad);
        modelo.aplicar_gradientes(grad, 0.1f, 0.9f, 0.0f);
    }
    const float perda_final = modelo.avaliar(exemplos);

    std::cout << "  -> Mini-corpus perda inicial: " << perda_inicial
              << " | final: " << perda_final << "\n";
    verificar(perda_final < 0.5f && perda_final < perda_inicial * 0.25f,
              "MLP reduz fortemente a perda em mini-corpus");

    // Testar salvar e carregar checkpoint
    const std::string caminho_tmp = "/tmp/teste_cpp_vibes_mlp.vibes";
    modelo.salvar(caminho_tmp);

    vibes::GeradorAleatorio rng_outro(999);
    vibes::ModeloMLP carregado(cfg, rng_outro);
    carregado.carregar(caminho_tmp);
    std::remove(caminho_tmp.c_str());

    const float perda_carregado = carregado.avaliar(exemplos);
    verificar(std::abs(perda_carregado - perda_final) < 1e-6f,
              "Checkpoint salvo e recarregado preserva perda identica");
}

}  // namespace

int main() {
    testar_gradient_check_mlp();
    testar_convergencia_e_persistencia();

    if (falhas > 0) {
        std::cerr << "\nTotal de falhas: " << falhas << "\n";
        return 1;
    }
    std::cout << "\nTodos os testes da Etapa 2 (MLP) passaram!\n";
    return 0;
}
