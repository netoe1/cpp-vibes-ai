#include "comum/aleatorio.hpp"
#include "comum/matriz.hpp"
#include "comum/tokenizador.hpp"

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

// Le um arquivo texto inteiro para uma std::string.
std::string ler_arquivo(const std::string& caminho) {
    std::ifstream arquivo(caminho);
    if (!arquivo.is_open()) {
        throw std::runtime_error("Nao foi possivel abrir o arquivo: " + caminho);
    }
    std::ostringstream buffer;
    buffer << arquivo.rdbuf();
    return buffer.str();
}

// -----------------------------------------------------------------------------
// Parte 1: Modelo Bigram Estatistico (Tabela de Contagem)
// -----------------------------------------------------------------------------
// Conta quantas vezes cada par (token_atual -> proximo_token) aparece no texto
// e calcula a entropia cruzada (Negative Log-Likelihood) media.
//
// Serve como "gabarito matematico": nossa rede neural da Parte 2 devera
// convergir para uma perda (loss) muito proxima deste valor!
float avaliar_bigram_estatistico(const std::vector<int>& tokens, std::size_t vocab) {
    vibes::Matriz contagens(vocab, vocab, 0.0f);

    const std::size_t num_pares = tokens.size() - 1;
    for (std::size_t i = 0; i < num_pares; ++i) {
        const auto x = static_cast<std::size_t>(tokens[i]);
        const auto y = static_cast<std::size_t>(tokens[i + 1]);
        contagens(x, y) += 1.0f;
    }

    // Suavizacao pequena (Laplace) evita log(0) caso algum par nunca ocorra.
    constexpr float suavizacao = 1e-4f;
    float soma_nll = 0.0f;

    for (std::size_t x = 0; x < vocab; ++x) {
        float total_linha = 0.0f;
        for (std::size_t y = 0; y < vocab; ++y) {
            total_linha += contagens(x, y) + suavizacao;
        }
        for (std::size_t y = 0; y < vocab; ++y) {
            const float c = contagens(x, y);
            if (c > 0.0f) {
                const float prob = (c + suavizacao) / total_linha;
                soma_nll += -c * std::log(prob);
            }
        }
    }

    return soma_nll / static_cast<float>(num_pares);
}

// -----------------------------------------------------------------------------
// Parte 2: Modelo Bigram Neural (Matriz de Pesos + Gradiente Descendente)
// -----------------------------------------------------------------------------
class ModeloBigramNeural {
public:
    ModeloBigramNeural(std::size_t vocab, vibes::GeradorAleatorio& rng)
        : vocab_(vocab), pesos_(vocab, vocab, 0.0f) {
        // Inicializacao aleatoria pequena (distribuicao normal padrao).
        for (float& w : pesos_.dados()) {
            w = rng.normal(0.0f, 0.5f);
        }
    }

    // Executa uma epoca completa de treinamento (Batch Gradient Descent):
    // 1. Forward pass: calcula probabilidades via softmax para cada par (x, y)
    // 2. Loss: acumula -log(P(y | x)) (Entropia Cruzada)
    // 3. Backward pass: calcula o gradiente analitico dLoss/dW
    // 4. Atualizacao: W = W - taxa_aprendizado * gradiente
    //
    // A derivacao matematica esta detalhada em docs/01_bigram.md:
    //   dLoss / dLogit_j = prob_j          (se j != y)
    //   dLoss / dLogit_y = prob_y - 1.0    (se j == y)
    float treinar_epoca(const std::vector<int>& tokens, float taxa_aprendizado) {
        const std::size_t num_pares = tokens.size() - 1;
        vibes::Matriz gradiente_total(vocab_, vocab_, 0.0f);
        float perda_total = 0.0f;

#ifdef USAR_OPENMP
#pragma omp parallel reduction(+ : perda_total)
        {
            // Cada thread acumula seus gradientes localmente para evitar
            // condicoes de corrida (race conditions) na memoria compartilhada.
            vibes::Matriz grad_local(vocab_, vocab_, 0.0f);
            std::vector<float> probs(vocab_, 0.0f);

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < num_pares; ++i) {
                const auto x = static_cast<std::size_t>(tokens[i]);
                const auto y = static_cast<std::size_t>(tokens[i + 1]);

                // Forward: linha x de W contem os logits para o proximo token
                vibes::softmax(pesos_.linha(x), probs, 1.0f);

                // Loss (entropia cruzada negativa do token correto y)
                perda_total += -std::log(std::max(probs[y], 1e-12f));

                // Backward: dL/dW[x, j] = probs[j] - (j == y ? 1 : 0)
                auto linha_grad = grad_local.linha(x);
                for (std::size_t j = 0; j < vocab_; ++j) {
                    linha_grad[j] += probs[j];
                }
                linha_grad[y] -= 1.0f;
            }

            // Soma os gradientes locais no gradiente global de forma segura
#pragma omp critical
            {
                auto& dst = gradiente_total.dados();
                const auto& src = grad_local.dados();
                for (std::size_t k = 0; k < dst.size(); ++k) {
                    dst[k] += src[k];
                }
            }
        }
#else
        std::vector<float> probs(vocab_, 0.0f);
        for (std::size_t i = 0; i < num_pares; ++i) {
            const auto x = static_cast<std::size_t>(tokens[i]);
            const auto y = static_cast<std::size_t>(tokens[i + 1]);

            vibes::softmax(pesos_.linha(x), probs, 1.0f);
            perda_total += -std::log(std::max(probs[y], 1e-12f));

            auto linha_grad = gradiente_total.linha(x);
            for (std::size_t j = 0; j < vocab_; ++j) {
                linha_grad[j] += probs[j];
            }
            linha_grad[y] -= 1.0f;
        }
#endif

        // Passo do Gradiente Descendente (media dos gradientes no dataset)
        const float escala = taxa_aprendizado / static_cast<float>(num_pares);
        auto& w = pesos_.dados();
        const auto& g = gradiente_total.dados();
        for (std::size_t k = 0; k < w.size(); ++k) {
            w[k] -= escala * g[k];
        }

        return perda_total / static_cast<float>(num_pares);
    }

    // Gera texto passo a passo (amostragem autoregressiva):
    // O token gerado vira a entrada para prever o proximo token.
    std::string gerar(const vibes::TokenizadorUTF8& tok, int token_inicial,
                      std::size_t max_tokens, float temperatura,
                      vibes::GeradorAleatorio& rng, bool parar_em_quebra_linha = false) const {
        std::vector<int> gerados;
        gerados.reserve(max_tokens);

        int atual = token_inicial;
        std::vector<float> probs(vocab_, 0.0f);
        const int id_quebra = tok.obter_id("\n");

        for (std::size_t passo = 0; passo < max_tokens; ++passo) {
            vibes::softmax(pesos_.linha(static_cast<std::size_t>(atual)), probs, temperatura);
            const int proximo = static_cast<int>(rng.amostrar_categorica(probs));

            if (parar_em_quebra_linha && proximo == id_quebra && !gerados.empty()) {
                break;
            }
            gerados.push_back(proximo);
            atual = proximo;
        }

        return tok.decodificar(gerados);
    }

private:
    std::size_t vocab_;
    vibes::Matriz pesos_;
};

}  // namespace

int main(int argc, char* argv[]) {
    std::string caminho_dados = "dados/dialogos.txt";
    bool modo_chat = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sem-chat") {
            modo_chat = false;
        } else {
            caminho_dados = arg;
        }
    }

    std::cout << "=========================================================\n";
    std::cout << "  cpp-vibes-ai | Etapa 1: Modelo Bigram (Neural em C++)\n";
    std::cout << "=========================================================\n";

#ifdef USAR_OPENMP
    std::cout << "[Info] OpenMP ativo com " << omp_get_max_threads() << " threads.\n";
#else
    std::cout << "[Info] Executando em modo single-thread.\n";
#endif

    // 1. Carregar corpus e tokenizar
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

    if (tokens.size() < 2) {
        std::cerr << "[Erro] O arquivo de dados precisa de pelo menos 2 caracteres.\n";
        return 1;
    }

    const std::size_t vocab = tokenizador.tamanho_vocabulario();
    std::cout << "[Dados] Arquivo: " << caminho_dados << "\n";
    std::cout << "[Dados] Total de tokens: " << tokens.size()
              << " | Vocabulario unico: " << vocab << " caracteres\n";
    std::cout << "[Modelo] Parametros (matriz W " << vocab << "x" << vocab
              << "): " << (vocab * vocab) << " pesos\n\n";

    // 2. Calcular referencia estatistica (otimo teorico do Bigram)
    const float perda_estatistica = avaliar_bigram_estatistico(tokens, vocab);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[Referencia] Perda otima (Bigram estatistico por contagem): "
              << perda_estatistica << "\n";
    std::cout << "[Referencia] Perda esperada de chutes aleatorios (-ln(1/V)): "
              << std::log(static_cast<float>(vocab)) << "\n\n";

    // 3. Treinar o modelo neural via Gradiente Descendente
    vibes::GeradorAleatorio rng(42);
    ModeloBigramNeural modelo(vocab, rng);

    constexpr int total_epocas = 400;
    constexpr float taxa_aprendizado = 30.0f;

    std::cout << "[Treino] Iniciando " << total_epocas
              << " epocas (taxa_aprendizado=" << taxa_aprendizado << ")...\n";

    float perda_final = 0.0f;
    for (int epoca = 1; epoca <= total_epocas; ++epoca) {
        perda_final = modelo.treinar_epoca(tokens, taxa_aprendizado);
        if (epoca == 1 || epoca == 10 || epoca == 50 || epoca % 100 == 0) {
            std::cout << "  Epoca " << std::setw(3) << epoca << "/" << total_epocas
                      << " | Perda (Cross-Entropy): " << perda_final << "\n";
        }
    }

    std::cout << "\n[Resultado] A rede neural convergiu para " << perda_final
              << " (referencia estatistica: " << perda_estatistica << ").\n\n";

    // 4. Amostra de texto gerado pelo modelo
    const int id_inicio = tokenizador.obter_id("\n") >= 0 ? tokenizador.obter_id("\n") : tokens[0];
    std::cout << "[Amostra gerada (temperatura=0.8)]:\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << modelo.gerar(tokenizador, id_inicio, 180, 0.8f, rng) << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "Nota educativa: o Bigram olha APENAS 1 caractere anterior!\n"
              << "Por isso ele aprende silabas e pedacos de palavras, mas ainda\n"
              << "nao tem memoria para responder perguntas com sentido.\n"
              << "Na Etapa 2 (MLP) vamos dar uma janela de contexto a ele!\n\n";

    if (!modo_chat) {
        return 0;
    }

    // 5. Loop interativo de conversa no terminal
    float temperatura = 0.8f;
    std::cout << "=== Modo Chat Experimental (Bigram) ===\n";
    std::cout << "Digite uma mensagem. Comandos: /temp <valor> | /sair\n\n";

    std::string entrada;
    while (true) {
        std::cout << "Voce: " << std::flush;
        if (!std::getline(std::cin, entrada)) {
            break;
        }
        if (entrada == "/sair" || entrada == "sair") {
            std::cout << "Encerrando chat. Ate a proxima etapa!\n";
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
                std::cout << "[Config] Valor invalido. Exemplo: /temp 0.7\n";
            }
            continue;
        }

        // O Bigram so usa o ultimo caractere do prompt ("IA: ") para comecar a gerar.
        const std::string prompt = "Usuario: " + entrada + "\nIA: ";
        const auto ids_prompt = tokenizador.codificar(prompt);
        const int ultimo_token = ids_prompt.empty() ? id_inicio : ids_prompt.back();

        const std::string resposta =
            modelo.gerar(tokenizador, ultimo_token, 80, temperatura, rng, true);
        std::cout << "IA (Bigram): " << resposta << "\n";
    }

    return 0;
}
