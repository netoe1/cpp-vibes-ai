#include "etapa3_transformer/transformer.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
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

void testar_mascara_causal() {
    vibes::ConfigTransformer cfg;
    cfg.vocab = 6;
    cfg.contexto_max = 8;
    cfg.dim_modelo = 8;
    cfg.num_cabecas = 2;
    cfg.num_camadas = 2;
    cfg.dim_ff = 16;

    vibes::GeradorAleatorio rng(11);
    vibes::ModeloTransformer modelo(cfg, rng);

    vibes::CacheTransformer c1, c2;
    const std::vector<int> seq_a = {1, 2, 3, 4};
    const std::vector<int> seq_b = {1, 2, 3, 5};  // muda apenas o ultimo token (t=3)

    modelo.forward_sequencia(seq_a, c1);
    modelo.forward_sequencia(seq_b, c2);

    float max_diff_passado = 0.0f;
    for (std::size_t t = 0; t < 3; ++t) {
        for (std::size_t v = 0; v < cfg.vocab; ++v) {
            max_diff_passado =
                std::max(max_diff_passado, std::abs(c1.logits(t, v) - c2.logits(t, v)));
        }
    }
    verificar(max_diff_passado < 1e-6f,
              "Mascara causal impede que tokens futuros influenciem posicoes anteriores");
}

void testar_gradient_check_transformer() {
    vibes::ConfigTransformer cfg;
    cfg.vocab = 5;
    cfg.contexto_max = 6;
    cfg.dim_modelo = 8;
    cfg.num_cabecas = 2;
    cfg.num_camadas = 2;
    cfg.dim_ff = 16;

    vibes::GeradorAleatorio rng(7);
    vibes::ModeloTransformer modelo(cfg, rng);

    const std::vector<vibes::SequenciaTreino> batch = {
        {{0, 1, 2, 1, 3}, {1, 2, 1, 3, 4}, {}},
        {{2, 3, 4, 0, 2}, {3, 4, 0, 2, 1}, {0.0f, 0.0f, 1.0f, 1.0f, 1.0f}},  // mascara SFT!
    };

    const float erro = vibes::verificar_gradiente_numerico_gpt(modelo, batch, 8, 1e-5);
    std::cout << "  -> Maior erro relativo no Gradient Check (GPT + Weight Tying + SFT): " << erro
              << "\n";
    verificar(erro < 5e-4f,
              "Gradient checking numerico do Mini-Transformer (Attention + LN + Weight Tying + SFT) < 5e-4");
}

void testar_convergencia_adamw_e_checkpoint() {
    const std::string mini_corpus =
        "Usuario: ola\nIA: ola! tudo bem?\n"
        "Usuario: oi\nIA: oi! como vai?\n";

    vibes::TokenizadorUTF8 tok;
    tok.construir_vocabulario(mini_corpus);
    const auto tokens = tok.codificar(mini_corpus);

    vibes::ConfigTransformer cfg;
    cfg.vocab = tok.tamanho_vocabulario();
    cfg.contexto_max = 32;
    cfg.dim_modelo = 24;
    cfg.num_cabecas = 4;
    cfg.num_camadas = 1;
    cfg.dim_ff = 64;

    vibes::SequenciaTreino seq;
    const std::size_t T = std::min<std::size_t>(cfg.contexto_max, tokens.size() - 1);
    seq.entrada.assign(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(T));
    seq.alvo.assign(tokens.begin() + 1, tokens.begin() + static_cast<std::ptrdiff_t>(T + 1));
    const std::vector<vibes::SequenciaTreino> batch = {seq};

    vibes::GeradorAleatorio rng(42);
    vibes::ModeloTransformer modelo(cfg, rng);
    vibes::ConjuntoTensoresGPT grad(cfg, 0.0f);

    const float perda_inicial = modelo.avaliar(batch);
    for (int passo = 0; passo < 80; ++passo) {
        modelo.calcular_perda_e_gradiente(batch, grad);
        modelo.passo_adamw(grad, 1e-2f, 0.01f, 1.0f);
    }
    const float perda_final = modelo.avaliar(batch);

    std::cout << "  -> Mini-Transformer perda inicial: " << perda_inicial
              << " | final: " << perda_final << "\n";
    verificar(perda_final < 0.3f && perda_final < perda_inicial * 0.2f,
              "Mini-Transformer converge fortemente com AdamW");

    // Persistir e recarregar checkpoint
    const std::string caminho_tmp = "/tmp/teste_cpp_vibes_gpt.vibes";
    modelo.salvar(caminho_tmp);

    vibes::GeradorAleatorio rng_outro(999);
    vibes::ModeloTransformer carregado(cfg, rng_outro);
    carregado.carregar(caminho_tmp);
    std::remove(caminho_tmp.c_str());

    const float perda_carregado = carregado.avaliar(batch);
    verificar(std::abs(perda_carregado - perda_final) < 1e-6f,
              "Checkpoint GPT salvo e recarregado preserva perda identica");
    verificar(carregado.passo_otimizador() == 80,
              "Checkpoint GPT preserva passo e estado do otimizador AdamW");

    // Validar Dropout no modo treino e metrica de confianca na geracao
    const float perda_drop = carregado.calcular_perda_e_gradiente(batch, grad, true, 1234u);
    verificar(std::isfinite(perda_drop) && perda_drop >= 0.0f,
              "Dropout no treino produz perda e gradientes finitos");

    const auto gen = carregado.gerar_com_confianca({0, 1, 2}, 10, 0.3f, 4, rng, -1);
    verificar(gen.ids.size() == 10 && gen.confianca_media > 0.0f && gen.confianca_media <= 1.0f &&
                  gen.perplexidade >= 1.0f,
              "gerar_com_confianca retorna confianca media em (0, 1] e perplexidade >= 1");
}

}  // namespace

int main() {
    testar_mascara_causal();
    testar_gradient_check_transformer();
    testar_convergencia_adamw_e_checkpoint();

    if (falhas > 0) {
        std::cerr << "\nTotal de falhas: " << falhas << "\n";
        return 1;
    }
    std::cout << "\nTodos os testes da Etapa 3 (Mini-Transformer) passaram!\n";
    return 0;
}
