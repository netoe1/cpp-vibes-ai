#include "etapa2_mlp/mlp.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

}  // namespace

int main(int argc, char* argv[]) {
    std::string caminho_dados = "dados/dialogos.txt";
    std::string caminho_carregar;
    std::string caminho_salvar;
    bool modo_chat = true;
    bool rodar_grad_check = false;
    int total_epocas = 80;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sem-chat") {
            modo_chat = false;
        } else if (arg == "--verificar-gradiente") {
            rodar_grad_check = true;
        } else if (arg == "--epocas" && i + 1 < argc) {
            total_epocas = std::stoi(argv[++i]);
        } else if (arg == "--salvar" && i + 1 < argc) {
            caminho_salvar = argv[++i];
        } else if (arg == "--carregar" && i + 1 < argc) {
            caminho_carregar = argv[++i];
        } else {
            caminho_dados = arg;
        }
    }

    std::cout << "=========================================================\n";
    std::cout << "  cpp-vibes-ai | Etapa 2: MLP de Linguagem (Bengio 2003)\n";
    std::cout << "=========================================================\n";

#ifdef USAR_OPENMP
    std::cout << "[Info] OpenMP ativo com " << omp_get_max_threads() << " threads.\n";
#else
    std::cout << "[Info] Executando em modo single-thread.\n";
#endif

    // 1. Ler corpus e construir vocabulario UTF-8
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

    if (tokens.size() < 4) {
        std::cerr << "[Erro] Corpus pequeno demais para treinar a MLP.\n";
        return 1;
    }

    const int id_pad = tokenizador.obter_id("\n") >= 0 ? tokenizador.obter_id("\n") : 0;

    vibes::ConfigMLP cfg;
    cfg.vocab = tokenizador.tamanho_vocabulario();
    cfg.contexto = 16;
    cfg.dim_emb = 24;
    cfg.dim_oculta = 128;

    // 2. Montar exemplos (janela de 16 tokens -> proximo token) e separar 90%/10%
    const auto todos_exemplos = vibes::construir_exemplos(tokens, cfg.contexto, id_pad);
    const std::size_t corte = std::max<std::size_t>(1, (todos_exemplos.size() * 9) / 10);

    std::vector<vibes::ExemploContexto> treino(todos_exemplos.begin(),
                                               todos_exemplos.begin() + static_cast<std::ptrdiff_t>(corte));
    std::vector<vibes::ExemploContexto> validacao(todos_exemplos.begin() + static_cast<std::ptrdiff_t>(corte),
                                                  todos_exemplos.end());

    vibes::GeradorAleatorio rng(42);
    vibes::ModeloMLP modelo(cfg, rng);

    std::cout << "[Dados] Arquivo: " << caminho_dados << " | Tokens: " << tokens.size()
              << " | Vocabulario: " << cfg.vocab << "\n";
    std::cout << "[Dados] Exemplos -> Treino: " << treino.size()
              << " | Validacao: " << validacao.size() << "\n";
    std::cout << "[Modelo] Contexto=" << cfg.contexto << " | Emb=" << cfg.dim_emb
              << " | Oculta=" << cfg.dim_oculta
              << " | Parametros: " << modelo.total_parametros() << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    // 3. Verificacao numerica opcional de gradiente
    if (rodar_grad_check) {
        const std::size_t n_check = std::min<std::size_t>(8, treino.size());
        std::vector<vibes::ExemploContexto> amostra(treino.begin(),
                                                    treino.begin() + static_cast<std::ptrdiff_t>(n_check));
        const float erro = vibes::verificar_gradiente_numerico(modelo, amostra);
        std::cout << "[GradCheck] Maior erro relativo (analitico vs numerico): "
                  << std::scientific << erro << std::fixed << "\n";
        if (erro >= 1e-2f) {
            std::cerr << "[GradCheck] FALHA: erro relativo acima do limite esperado!\n";
            return 2;
        }
        std::cout << "[GradCheck] OK! Backpropagation verificado numericamente.\n\n";
    }

    // 4. Carregar checkpoint ou treinar do zero com Mini-Batch SGD + Momentum
    if (!caminho_carregar.empty()) {
        try {
            modelo.carregar(caminho_carregar);
            std::cout << "[Checkpoint] Pesos carregados de: " << caminho_carregar << "\n";
            std::cout << "[Avaliacao] Perda Treino: " << modelo.avaliar(treino)
                      << " | Perda Validacao: " << modelo.avaliar(validacao) << "\n\n";
        } catch (const std::exception& e) {
            std::cerr << "[Erro] " << e.what() << "\n";
            return 1;
        }
    } else {
        constexpr std::size_t tamanho_batch = 64;
        const float lr_inicial = 0.08f;
        const float lr_final = 0.01f;

        std::vector<std::size_t> ordem(treino.size());
        std::iota(ordem.begin(), ordem.end(), 0);
        vibes::GradientesMLP grad(cfg);

        std::cout << "[Treino] Perda inicial -> Treino: " << modelo.avaliar(treino)
                  << " | Validacao: " << modelo.avaliar(validacao) << "\n";
        std::cout << "[Treino] Iniciando " << total_epocas
                  << " epocas (batch=" << tamanho_batch << ", momentum=0.9)...\n";

        for (int epoca = 1; epoca <= total_epocas; ++epoca) {
            std::shuffle(ordem.begin(), ordem.end(), rng.motor());

            // Decaimento linear da taxa de aprendizado
            const float progresso = static_cast<float>(epoca - 1) /
                                    static_cast<float>(std::max(1, total_epocas - 1));
            const float lr = lr_inicial + progresso * (lr_final - lr_inicial);

            float soma_perda = 0.0f;
            std::size_t num_batches = 0;

            for (std::size_t inicio = 0; inicio < ordem.size(); inicio += tamanho_batch) {
                const std::size_t fim = std::min(inicio + tamanho_batch, ordem.size());
                std::vector<std::size_t> batch_idx(ordem.begin() + static_cast<std::ptrdiff_t>(inicio),
                                                   ordem.begin() + static_cast<std::ptrdiff_t>(fim));
                soma_perda += modelo.calcular_perda_e_gradiente(treino, batch_idx, grad);
                modelo.aplicar_gradientes(grad, lr, 0.9f, 1e-4f);
                ++num_batches;
            }

            if (epoca == 1 || epoca == 10 || epoca % 20 == 0 || epoca == total_epocas) {
                const float perda_treino = soma_perda / static_cast<float>(num_batches);
                const float perda_val = modelo.avaliar(validacao);
                std::cout << "  Epoca " << std::setw(3) << epoca << "/" << total_epocas
                          << " | lr=" << std::setprecision(3) << lr << std::setprecision(4)
                          << " | Treino: " << perda_treino
                          << " | Validacao: " << perda_val << "\n";
            }
        }

        if (!caminho_salvar.empty()) {
            modelo.salvar(caminho_salvar);
            std::cout << "\n[Checkpoint] Modelo salvo em: " << caminho_salvar << "\n";
        }
        std::cout << "\n";
    }

    // 5. Geracao de demonstracao
    const std::string prompt_demo = "Usuario: o que e uma rede neural?\nIA: ";
    const auto ids_demo = tokenizador.codificar(prompt_demo);
    std::cout << "[Amostra gerada (temperatura=0.6)]:\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << prompt_demo
              << modelo.gerar(tokenizador, ids_demo, id_pad, 120, 0.6f, rng, true) << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "Nota educativa: agora o modelo enxerga " << cfg.contexto
              << " caracteres de contexto!\n"
              << "Compare com o Bigram (loss ~2.06): a MLP forma palavras inteiras\n"
              << "e frases do dominio, embora ainda tenha janela curta e fixa.\n"
              << "Na Etapa 3 (Mini-Transformer) vamos adicionar Self-Attention!\n\n";

    if (!modo_chat) {
        return 0;
    }

    // 6. Loop interativo de chat
    float temperatura = 0.6f;
    std::cout << "=== Modo Chat (Etapa 2: MLP) ===\n";
    std::cout << "Comandos: /temp <valor> | /salvar <arquivo> | /sair\n\n";

    std::string entrada;
    while (true) {
        std::cout << "Voce: " << std::flush;
        if (!std::getline(std::cin, entrada)) {
            break;
        }
        if (entrada == "/sair" || entrada == "sair") {
            std::cout << "Encerrando chat. Ate a Etapa 3 (Transformer)!\n";
            break;
        }
        if (entrada.rfind("/temp ", 0) == 0) {
            try {
                const float nova_temp = std::stof(entrada.substr(6));
                if (nova_temp > 0.05f && nova_temp <= 5.0f) {
                    temperatura = nova_temp;
                    std::cout << "[Config] Temperatura ajustada para " << temperatura << "\n";
                } else {
                    std::cout << "[Config] Use um valor entre 0.05 e 5.0.\n";
                }
            } catch (...) {
                std::cout << "[Config] Valor invalido. Exemplo: /temp 0.5\n";
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

        const std::string prompt = "Usuario: " + entrada + "\nIA: ";
        const auto ids_prompt = tokenizador.codificar(prompt);
        const std::string resposta =
            modelo.gerar(tokenizador, ids_prompt, id_pad, 140, temperatura, rng, true);
        std::cout << "IA (MLP): " << resposta << "\n";
    }

    return 0;
}
