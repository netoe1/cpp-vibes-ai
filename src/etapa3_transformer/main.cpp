#include "etapa3_transformer/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USAR_OPENMP
#include <omp.h>
#endif

namespace {

std::string ler_arquivo(const std::string& caminho) {
    std::ifstream arquivo(caminho);
    if (!arquivo.is_open()) {
        throw std::runtime_error("Nao foi possivel abrir o arquivo: " + caminho);
    }
    std::ostringstream buffer;
    buffer << arquivo.rdbuf();
    return buffer.str();
}

// Amostra um mini-batch aleatorio de janelas de comprimento T dentro de um vetor de tokens.
std::vector<vibes::SequenciaTreino> amostrar_batch(const std::vector<int>& tokens,
                                                   std::size_t tamanho_batch,
                                                   std::size_t comprimento_seq,
                                                   vibes::GeradorAleatorio& rng) {
    std::vector<vibes::SequenciaTreino> batch;
    if (tokens.size() < 2) return batch;
    const std::size_t T = std::min(comprimento_seq, tokens.size() - 1);
    const std::size_t max_inicio = tokens.size() - T - 1;

    std::uniform_int_distribution<std::size_t> dist(0, max_inicio);
    batch.reserve(tamanho_batch);
    for (std::size_t b = 0; b < tamanho_batch; ++b) {
        const std::size_t ini = dist(rng.motor());
        vibes::SequenciaTreino seq;
        seq.entrada.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini),
                           tokens.begin() + static_cast<std::ptrdiff_t>(ini + T));
        seq.alvo.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini + 1),
                        tokens.begin() + static_cast<std::ptrdiff_t>(ini + T + 1));
        batch.push_back(std::move(seq));
    }
    return batch;
}

// Constroi ate `max_janelas` janelas deterministas espacadas ao longo do split
// para manter a avaliacao rapida mesmo em datasets de varios megabytes.
std::vector<vibes::SequenciaTreino> construir_janelas_avaliacao(const std::vector<int>& tokens,
                                                                std::size_t comprimento_seq,
                                                                std::size_t max_janelas = 32) {
    std::vector<vibes::SequenciaTreino> seqs;
    if (tokens.size() < 2) return seqs;
    const std::size_t T = std::min(comprimento_seq, tokens.size() - 1);
    const std::size_t total_possiveis = (tokens.size() - 1) / T;
    if (total_possiveis == 0) return seqs;

    const std::size_t passo_janelas =
        std::max<std::size_t>(1, total_possiveis / std::max<std::size_t>(1, max_janelas));

    for (std::size_t j = 0; j < total_possiveis && seqs.size() < max_janelas; j += passo_janelas) {
        const std::size_t ini = j * T;
        if (ini + T >= tokens.size()) break;
        vibes::SequenciaTreino seq;
        seq.entrada.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini),
                           tokens.begin() + static_cast<std::ptrdiff_t>(ini + T));
        seq.alvo.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini + 1),
                        tokens.begin() + static_cast<std::ptrdiff_t>(ini + T + 1));
        seqs.push_back(std::move(seq));
    }
    return seqs;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string caminho_dados = "dados/dialogos.txt";
    std::string caminho_salvar;
    std::string caminho_carregar;
    bool modo_chat = true;
    bool rodar_grad_check = false;
    int total_passos = 250;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sem-chat") {
            modo_chat = false;
        } else if (arg == "--verificar-gradiente") {
            rodar_grad_check = true;
        } else if (arg == "--passos" && i + 1 < argc) {
            total_passos = std::stoi(argv[++i]);
        } else if (arg == "--salvar" && i + 1 < argc) {
            caminho_salvar = argv[++i];
        } else if (arg == "--carregar" && i + 1 < argc) {
            caminho_carregar = argv[++i];
        } else {
            caminho_dados = arg;
        }
    }

    std::cout << "==============================================================\n";
    std::cout << "  cpp-vibes-ai | Etapa 3: Mini-Transformer Conversacional (GPT)\n";
    std::cout << "==============================================================\n";

#ifdef USAR_OPENMP
    std::cout << "[Info] OpenMP ativo com " << omp_get_max_threads() << " threads.\n";
#else
    std::cout << "[Info] Executando em modo single-thread.\n";
#endif

    // 1. Ler corpus e tokenizar
    std::string texto;
    try {
        texto = ler_arquivo(caminho_dados);
    } catch (const std::exception& e) {
        std::cerr << "[Erro] " << e.what() << "\n";
        return 1;
    }

    vibes::TokenizadorUTF8 tokenizador;
    tokenizador.construir_vocabulario(texto);
    const std::vector<int> tokens = tokenizador.codificar(texto);

    if (tokens.size() < 16) {
        std::cerr << "[Erro] Corpus pequeno demais para treinar o Transformer.\n";
        return 1;
    }

    // Separacao 90% treino / 10% validacao
    const std::size_t corte = (tokens.size() * 9) / 10;
    const std::vector<int> tokens_treino(tokens.begin(),
                                         tokens.begin() + static_cast<std::ptrdiff_t>(corte));
    const std::vector<int> tokens_val(tokens.begin() + static_cast<std::ptrdiff_t>(corte),
                                      tokens.end());

    vibes::ConfigTransformer cfg;
    cfg.vocab = tokenizador.tamanho_vocabulario();
    cfg.contexto_max = 64;
    cfg.dim_modelo = 64;
    cfg.num_cabecas = 4;
    cfg.num_camadas = 2;
    cfg.dim_ff = 256;

    vibes::GeradorAleatorio rng(42);
    vibes::ModeloTransformer modelo(cfg, rng);

    std::cout << "[Dados] Arquivo: " << caminho_dados << " | Tokens: " << tokens.size()
              << " (Treino: " << tokens_treino.size() << ", Val: " << tokens_val.size() << ")\n";
    std::cout << "[Modelo] Contexto=" << cfg.contexto_max << " | D=" << cfg.dim_modelo
              << " | Cabecas=" << cfg.num_cabecas << " | Camadas=" << cfg.num_camadas
              << " | Parametros: " << modelo.total_parametros() << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    // 2. Gradient Check opcional
    if (rodar_grad_check) {
        const auto mini_batch = amostrar_batch(tokens_treino, 2, 12, rng);
        const float erro = vibes::verificar_gradiente_numerico_gpt(modelo, mini_batch, 3, 1e-5);
        std::cout << "[GradCheck] Maior erro relativo (analitico vs numerico): "
                  << std::scientific << erro << std::fixed << "\n";
        if (erro >= 1e-3f) {
            std::cerr << "[GradCheck] FALHA: erro relativo acima do limite esperado!\n";
            return 2;
        }
        std::cout << "[GradCheck] OK! Attention, LayerNorm e FFN verificados numericamente.\n\n";
    }

    const auto janelas_treino_eval = construir_janelas_avaliacao(tokens_treino, cfg.contexto_max);
    const auto janelas_val_eval = construir_janelas_avaliacao(tokens_val, cfg.contexto_max);

    // 3. Carregar checkpoint ou treinar com AdamW + Cosine LR Schedule
    if (!caminho_carregar.empty()) {
        try {
            modelo.carregar(caminho_carregar);
            std::cout << "[Checkpoint] Pesos carregados de: " << caminho_carregar << "\n";
            std::cout << "[Avaliacao] Treino: " << modelo.avaliar(janelas_treino_eval)
                      << " | Validacao: " << modelo.avaliar(janelas_val_eval) << "\n\n";
        } catch (const std::exception& e) {
            std::cerr << "[Erro] " << e.what() << "\n";
            return 1;
        }
    } else {
        constexpr std::size_t tamanho_batch = 16;
        constexpr float lr_max = 3e-3f;
        constexpr float lr_min = 3e-4f;
        const int warmup = std::max(1, total_passos / 10);

        vibes::ConjuntoTensoresGPT grad(cfg, 0.0f);

        std::cout << "[Treino] Perda inicial -> Treino: " << modelo.avaliar(janelas_treino_eval)
                  << " | Validacao: " << modelo.avaliar(janelas_val_eval) << "\n";
        std::cout << "[Treino] Iniciando " << total_passos
                  << " passos (batch=" << tamanho_batch << ", AdamW + Cosine LR)...\n";

        for (int passo = 1; passo <= total_passos; ++passo) {
            // Warmup linear seguido de decaimento cosseno
            float lr = lr_max;
            if (passo <= warmup) {
                lr = lr_max * (static_cast<float>(passo) / static_cast<float>(warmup));
            } else {
                const float prog = static_cast<float>(passo - warmup) /
                                   static_cast<float>(std::max(1, total_passos - warmup));
                constexpr float pi = 3.14159265358979323846f;
                lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(pi * prog));
            }

            const auto batch = amostrar_batch(tokens_treino, tamanho_batch, cfg.contexto_max, rng);
            const float perda_batch = modelo.calcular_perda_e_gradiente(batch, grad);
            modelo.passo_adamw(grad, lr, 0.01f, 1.0f);

            if (passo == 1 || passo == 25 || passo % 50 == 0 || passo == total_passos) {
                const float perda_val = modelo.avaliar(janelas_val_eval);
                std::cout << "  Passo " << std::setw(4) << passo << "/" << total_passos
                          << " | lr=" << std::setprecision(5) << lr << std::setprecision(4)
                          << " | Batch: " << perda_batch
                          << " | Validacao: " << perda_val << "\n";
            }
        }

        if (!caminho_salvar.empty()) {
            modelo.salvar(caminho_salvar);
            std::cout << "\n[Checkpoint] Modelo salvo em: " << caminho_salvar << "\n";
        }
        std::cout << "\n";
    }

    // 4. Demonstracao de geracao com Temperatura + Top-K
    const std::string prompt_demo = "Usuario: como funciona um transformer?\nIA: ";
    const auto ids_demo = tokenizador.codificar(prompt_demo);
    std::cout << "[Amostra gerada (temperatura=0.6, top_k=10)]:\n";
    std::cout << "--------------------------------------------------------------\n";
    std::cout << prompt_demo
              << modelo.gerar(tokenizador, ids_demo, 140, 0.6f, 10, rng, true) << "\n";
    std::cout << "--------------------------------------------------------------\n\n";

    if (!modo_chat) {
        return 0;
    }

    // 5. Chat interativo com historico na janela de contexto
    float temperatura = 0.6f;
    std::size_t top_k = 10;
    std::string historico;

    std::cout << "=== Modo Chat Conversacional (Etapa 3: Mini-Transformer) ===\n";
    std::cout << "Comandos: /temp <val> | /topk <k> | /limpar | /salvar <arq> | /sair\n\n";

    std::string entrada;
    while (true) {
        std::cout << "Voce: " << std::flush;
        if (!std::getline(std::cin, entrada)) {
            break;
        }
        if (entrada == "/sair" || entrada == "sair") {
            std::cout << "Encerrando chat. Parabens por concluir as 3 etapas!\n";
            break;
        }
        if (entrada == "/limpar") {
            historico.clear();
            std::cout << "[Chat] Historico limpo.\n";
            continue;
        }
        if (entrada.rfind("/temp ", 0) == 0) {
            try {
                const float nova = std::stof(entrada.substr(6));
                if (nova > 0.05f && nova <= 5.0f) {
                    temperatura = nova;
                    std::cout << "[Config] Temperatura=" << temperatura << "\n";
                } else {
                    std::cout << "[Config] Use um valor entre 0.05 e 5.0.\n";
                }
            } catch (...) {
                std::cout << "[Config] Valor invalido. Exemplo: /temp 0.6\n";
            }
            continue;
        }
        if (entrada.rfind("/topk ", 0) == 0) {
            try {
                top_k = static_cast<std::size_t>(std::stoul(entrada.substr(6)));
                std::cout << "[Config] Top-K=" << top_k << (top_k == 0 ? " (desativado)" : "")
                          << "\n";
            } catch (...) {
                std::cout << "[Config] Valor invalido. Exemplo: /topk 10\n";
            }
            continue;
        }
        if (entrada.rfind("/salvar ", 0) == 0) {
            const std::string destino = entrada.substr(8);
            try {
                modelo.salvar(destino);
                std::cout << "[Checkpoint] Salvo com sucesso em: " << destino << "\n";
            } catch (const std::exception& e) {
                std::cout << "[Erro] " << e.what() << "\n";
            }
            continue;
        }

        historico += "Usuario: " + entrada + "\nIA: ";
        const auto ids_ctx = tokenizador.codificar(historico);
        const std::string resposta =
            modelo.gerar(tokenizador, ids_ctx, 140, temperatura, top_k, rng, true);
        std::cout << "IA (GPT): " << resposta << "\n";
        historico += resposta + "\n\n";
    }

    return 0;
}
