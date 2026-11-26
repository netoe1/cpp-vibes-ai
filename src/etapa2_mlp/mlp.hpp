#pragma once

#include "comum/aleatorio.hpp"
#include "comum/matriz.hpp"
#include "comum/tokenizador.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USAR_OPENMP
#include <omp.h>
#endif

namespace vibes {

// Hiperparametros estruturais do modelo MLP (estilo Bengio et al., 2003).
struct ConfigMLP {
    std::size_t vocab = 0;          // Tamanho do vocabulario (V)
    std::size_t contexto = 16;      // Janela de contexto em tokens (B)
    std::size_t dim_emb = 24;       // Dimensao de cada embedding (E)
    std::size_t dim_oculta = 128;   // Neuronios na camada oculta (H)

    std::size_t dim_entrada() const { return contexto * dim_emb; }
};

// Um exemplo de treinamento: B tokens de contexto -> 1 token alvo.
struct ExemploContexto {
    std::vector<int> contexto;
    int alvo = 0;
};

// Constroi pares (contexto[B], alvo) a partir de uma sequencia de tokens.
// As posicoes anteriores ao inicio do texto sao preenchidas com `id_pad`.
inline std::vector<ExemploContexto> construir_exemplos(const std::vector<int>& tokens,
                                                       std::size_t tamanho_contexto,
                                                       int id_pad) {
    std::vector<ExemploContexto> exemplos;
    if (tokens.size() < 2) {
        return exemplos;
    }
    exemplos.reserve(tokens.size() - 1);

    std::vector<int> janela(tamanho_contexto, id_pad);
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
        // Desloca a janela 1 posicao para a esquerda e insere tokens[i] no fim
        for (std::size_t p = 0; p + 1 < tamanho_contexto; ++p) {
            janela[p] = janela[p + 1];
        }
        janela[tamanho_contexto - 1] = tokens[i];

        exemplos.push_back({janela, tokens[i + 1]});
    }
    return exemplos;
}

// Acumulador de gradientes com exatamente as mesmas dimensoes dos pesos da MLP.
struct GradientesMLP {
    Matriz dC;
    Matriz dW1;
    std::vector<float> db1;
    Matriz dW2;
    std::vector<float> db2;

    explicit GradientesMLP(const ConfigMLP& cfg)
        : dC(cfg.vocab, cfg.dim_emb, 0.0f),
          dW1(cfg.dim_entrada(), cfg.dim_oculta, 0.0f),
          db1(cfg.dim_oculta, 0.0f),
          dW2(cfg.dim_oculta, cfg.vocab, 0.0f),
          db2(cfg.vocab, 0.0f) {}

    void zerar() {
        dC.preencher(0.0f);
        dW1.preencher(0.0f);
        std::fill(db1.begin(), db1.end(), 0.0f);
        dW2.preencher(0.0f);
        std::fill(db2.begin(), db2.end(), 0.0f);
    }

    void somar_de(const GradientesMLP& outro) {
        auto somar_vec = [](std::vector<float>& dst, const std::vector<float>& src) {
            for (std::size_t i = 0; i < dst.size(); ++i) {
                dst[i] += src[i];
            }
        };
        somar_vec(dC.dados(), outro.dC.dados());
        somar_vec(dW1.dados(), outro.dW1.dados());
        somar_vec(db1, outro.db1);
        somar_vec(dW2.dados(), outro.dW2.dados());
        somar_vec(db2, outro.db2);
    }
};

// Buffers temporarios reutilizados por thread durante forward e backward,
// evitando alocacoes repetidas na memoria dentro dos lacos de treino.
struct BuffersMLP {
    std::vector<float> x;
    std::vector<float> h;
    std::vector<float> logits;
    std::vector<float> probs;
    std::vector<float> dlogits;
    std::vector<float> dh;
    std::vector<float> dz1;
    std::vector<float> dx;

    explicit BuffersMLP(const ConfigMLP& cfg)
        : x(cfg.dim_entrada(), 0.0f),
          h(cfg.dim_oculta, 0.0f),
          logits(cfg.vocab, 0.0f),
          probs(cfg.vocab, 0.0f),
          dlogits(cfg.vocab, 0.0f),
          dh(cfg.dim_oculta, 0.0f),
          dz1(cfg.dim_oculta, 0.0f),
          dx(cfg.dim_entrada(), 0.0f) {}
};

// -----------------------------------------------------------------------------
// Modelo MLP (Multi-Layer Perceptron) de Linguagem
// -----------------------------------------------------------------------------
// Arquitetura:
//   contexto [B] -> Embeddings C [V x E] -> concatenacao x [B*E]
//   -> Camada oculta h = tanh(x * W1 + b1) [H]
//   -> Logits = h * W2 + b2 [V]
//   -> Softmax -> Probabilidades [V]
class ModeloMLP {
public:
    ModeloMLP(const ConfigMLP& cfg, GeradorAleatorio& rng)
        : cfg_(cfg),
          C_(cfg.vocab, cfg.dim_emb, 0.0f),
          W1_(cfg.dim_entrada(), cfg.dim_oculta, 0.0f),
          b1_(cfg.dim_oculta, 0.0f),
          W2_(cfg.dim_oculta, cfg.vocab, 0.0f),
          b2_(cfg.vocab, 0.0f),
          vel_C_(cfg.vocab, cfg.dim_emb, 0.0f),
          vel_W1_(cfg.dim_entrada(), cfg.dim_oculta, 0.0f),
          vel_b1_(cfg.dim_oculta, 0.0f),
          vel_W2_(cfg.dim_oculta, cfg.vocab, 0.0f),
          vel_b2_(cfg.vocab, 0.0f) {
        // Inicializacao calibrada para evitar saturacao de tanh e perda inicial explosiva:
        // - C: pequena distribuicao normal
        // - W1: escala Kaiming/Xavier para tanh (~ (5/3) / sqrt(D))
        // - W2: pequena (0.1 / sqrt(H)) para logits iniciais proximos de zero
        const float escala_w1 = (5.0f / 3.0f) / std::sqrt(static_cast<float>(cfg.dim_entrada()));
        const float escala_w2 = 0.1f / std::sqrt(static_cast<float>(cfg.dim_oculta));

        for (float& v : C_.dados()) v = rng.normal(0.0f, 0.2f);
        for (float& v : W1_.dados()) v = rng.normal(0.0f, escala_w1);
        for (float& v : W2_.dados()) v = rng.normal(0.0f, escala_w2);
    }

    const ConfigMLP& config() const { return cfg_; }

    std::size_t total_parametros() const {
        return C_.total_elementos() + W1_.total_elementos() + b1_.size() +
               W2_.total_elementos() + b2_.size();
    }

    // Forward de um unico exemplo: preenche x, h, logits e probs nos buffers.
    void forward_exemplo(const std::vector<int>& contexto, BuffersMLP& buf) const {
        assert(contexto.size() == cfg_.contexto);
        const std::size_t E = cfg_.dim_emb;
        const std::size_t D = cfg_.dim_entrada();
        const std::size_t H = cfg_.dim_oculta;
        const std::size_t V = cfg_.vocab;

        // 1. Concatenacao dos embeddings do contexto -> vetor x [D]
        for (std::size_t p = 0; p < cfg_.contexto; ++p) {
            const auto id = static_cast<std::size_t>(contexto[p]);
            const auto emb = C_.linha(id);
            for (std::size_t e = 0; e < E; ++e) {
                buf.x[p * E + e] = emb[e];
            }
        }

        // 2. Camada oculta: z1 = x * W1 + b1, seguido de h = tanh(z1)
        std::copy(b1_.begin(), b1_.end(), buf.h.begin());
        for (std::size_t d = 0; d < D; ++d) {
            const float xd = buf.x[d];
            const auto linha_w1 = W1_.linha(d);
            for (std::size_t j = 0; j < H; ++j) {
                buf.h[j] += xd * linha_w1[j];
            }
        }
        for (std::size_t j = 0; j < H; ++j) {
            buf.h[j] = std::tanh(buf.h[j]);
        }

        // 3. Camada de saida: logits = h * W2 + b2
        std::copy(b2_.begin(), b2_.end(), buf.logits.begin());
        for (std::size_t j = 0; j < H; ++j) {
            const float hj = buf.h[j];
            const auto linha_w2 = W2_.linha(j);
            for (std::size_t v = 0; v < V; ++v) {
                buf.logits[v] += hj * linha_w2[v];
            }
        }

        // 4. Probabilidades via softmax
        softmax(buf.logits, buf.probs, 1.0f);
    }

    // Calcula forward + loss + acumula gradientes (backward) para um exemplo.
    float forward_backward_exemplo(const ExemploContexto& ex, BuffersMLP& buf,
                                   GradientesMLP& grad) const {
        forward_exemplo(ex.contexto, buf);

        const auto y = static_cast<std::size_t>(ex.alvo);
        const float perda = -std::log(std::max(buf.probs[y], 1e-12f));

        const std::size_t E = cfg_.dim_emb;
        const std::size_t D = cfg_.dim_entrada();
        const std::size_t H = cfg_.dim_oculta;
        const std::size_t V = cfg_.vocab;

        // 1. Gradiente nos logits: dL/dlogits = probs - one_hot(y)
        for (std::size_t v = 0; v < V; ++v) {
            buf.dlogits[v] = buf.probs[v];
        }
        buf.dlogits[y] -= 1.0f;

        // 2. Gradientes de b2, W2 e propagacao para h
        for (std::size_t v = 0; v < V; ++v) {
            grad.db2[v] += buf.dlogits[v];
        }
        for (std::size_t j = 0; j < H; ++j) {
            const float hj = buf.h[j];
            auto linha_dw2 = grad.dW2.linha(j);
            const auto linha_w2 = W2_.linha(j);
            float soma_dh = 0.0f;
            for (std::size_t v = 0; v < V; ++v) {
                const float dl = buf.dlogits[v];
                linha_dw2[v] += hj * dl;
                soma_dh += dl * linha_w2[v];
            }
            buf.dh[j] = soma_dh;
        }

        // 3. Derivada da tanh: d(tanh(z))/dz = 1 - h^2
        for (std::size_t j = 0; j < H; ++j) {
            buf.dz1[j] = buf.dh[j] * (1.0f - buf.h[j] * buf.h[j]);
            grad.db1[j] += buf.dz1[j];
        }

        // 4. Gradientes de W1 e propagacao para a entrada concatenada x
        for (std::size_t d = 0; d < D; ++d) {
            const float xd = buf.x[d];
            auto linha_dw1 = grad.dW1.linha(d);
            const auto linha_w1 = W1_.linha(d);
            float soma_dx = 0.0f;
            for (std::size_t j = 0; j < H; ++j) {
                const float dz = buf.dz1[j];
                linha_dw1[j] += xd * dz;
                soma_dx += dz * linha_w1[j];
            }
            buf.dx[d] = soma_dx;
        }

        // 5. Espalhamento (scatter-add) do gradiente de volta na tabela de embeddings C.
        // Usamos += pois o mesmo token pode aparecer multiplas vezes no contexto.
        for (std::size_t p = 0; p < cfg_.contexto; ++p) {
            const auto id = static_cast<std::size_t>(ex.contexto[p]);
            auto linha_dc = grad.dC.linha(id);
            for (std::size_t e = 0; e < E; ++e) {
                linha_dc[e] += buf.dx[p * E + e];
            }
        }

        return perda;
    }

    // Calcula a perda media e o gradiente medio sobre um conjunto de indices de exemplos.
    float calcular_perda_e_gradiente(const std::vector<ExemploContexto>& exemplos,
                                     const std::vector<std::size_t>& indices,
                                     GradientesMLP& grad_saida) const {
        grad_saida.zerar();
        if (indices.empty()) {
            return 0.0f;
        }

        float perda_total = 0.0f;

#ifdef USAR_OPENMP
#pragma omp parallel reduction(+ : perda_total)
        {
            GradientesMLP grad_local(cfg_);
            BuffersMLP buf_local(cfg_);

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < indices.size(); ++i) {
                perda_total += forward_backward_exemplo(exemplos[indices[i]], buf_local, grad_local);
            }

#pragma omp critical
            {
                grad_saida.somar_de(grad_local);
            }
        }
#else
        BuffersMLP buf(cfg_);
        for (std::size_t idx : indices) {
            perda_total += forward_backward_exemplo(exemplos[idx], buf, grad_saida);
        }
#endif

        const float inv_n = 1.0f / static_cast<float>(indices.size());
        auto escalar = [inv_n](std::vector<float>& v) {
            for (float& x : v) x *= inv_n;
        };
        escalar(grad_saida.dC.dados());
        escalar(grad_saida.dW1.dados());
        escalar(grad_saida.db1);
        escalar(grad_saida.dW2.dados());
        escalar(grad_saida.db2);

        return perda_total * inv_n;
    }

    // Atualiza todos os pesos com SGD + Momentum (e leve weight decay L2 opcional).
    void aplicar_gradientes(const GradientesMLP& grad, float taxa_aprendizado,
                            float momentum = 0.9f, float decaimento_peso = 1e-4f) {
        auto atualizar = [&](std::vector<float>& peso, std::vector<float>& vel,
                             const std::vector<float>& g, bool aplicar_wd) {
            for (std::size_t i = 0; i < peso.size(); ++i) {
                const float g_eff = aplicar_wd ? (g[i] + decaimento_peso * peso[i]) : g[i];
                vel[i] = momentum * vel[i] + g_eff;
                peso[i] -= taxa_aprendizado * vel[i];
            }
        };

        atualizar(C_.dados(), vel_C_.dados(), grad.dC.dados(), true);
        atualizar(W1_.dados(), vel_W1_.dados(), grad.dW1.dados(), true);
        atualizar(b1_, vel_b1_, grad.db1, false);
        atualizar(W2_.dados(), vel_W2_.dados(), grad.dW2.dados(), true);
        atualizar(b2_, vel_b2_, grad.db2, false);
    }

    // Avalia a perda media em um conjunto (ex.: validacao) sem alterar pesos.
    float avaliar(const std::vector<ExemploContexto>& exemplos) const {
        if (exemplos.empty()) {
            return 0.0f;
        }
        float perda_total = 0.0f;

#ifdef USAR_OPENMP
#pragma omp parallel reduction(+ : perda_total)
        {
            BuffersMLP buf(cfg_);
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < exemplos.size(); ++i) {
                forward_exemplo(exemplos[i].contexto, buf);
                const auto y = static_cast<std::size_t>(exemplos[i].alvo);
                perda_total += -std::log(std::max(buf.probs[y], 1e-12f));
            }
        }
#else
        BuffersMLP buf(cfg_);
        for (const auto& ex : exemplos) {
            forward_exemplo(ex.contexto, buf);
            const auto y = static_cast<std::size_t>(ex.alvo);
            perda_total += -std::log(std::max(buf.probs[y], 1e-12f));
        }
#endif

        return perda_total / static_cast<float>(exemplos.size());
    }

    // Gera texto autoregressivamente a partir de uma lista inicial de tokens.
    std::string gerar(const TokenizadorUTF8& tok, const std::vector<int>& tokens_prompt,
                      int id_pad, std::size_t max_tokens, float temperatura,
                      GeradorAleatorio& rng, bool parar_em_quebra_linha = false) const {
        std::vector<int> janela(cfg_.contexto, id_pad);
        // Preenche o fim da janela com os ultimos tokens do prompt
        const std::size_t copiar = std::min(cfg_.contexto, tokens_prompt.size());
        const std::size_t inicio_prompt = tokens_prompt.size() - copiar;
        for (std::size_t i = 0; i < copiar; ++i) {
            janela[cfg_.contexto - copiar + i] = tokens_prompt[inicio_prompt + i];
        }

        BuffersMLP buf(cfg_);
        std::vector<int> gerados;
        gerados.reserve(max_tokens);
        const int id_quebra = tok.obter_id("\n");

        for (std::size_t passo = 0; passo < max_tokens; ++passo) {
            forward_exemplo(janela, buf);
            softmax(buf.logits, buf.probs, temperatura);
            const int proximo = static_cast<int>(rng.amostrar_categorica(buf.probs));

            if (parar_em_quebra_linha && proximo == id_quebra && !gerados.empty()) {
                break;
            }
            gerados.push_back(proximo);

            // Desliza a janela de contexto para incluir o token recem-gerado
            for (std::size_t p = 0; p + 1 < cfg_.contexto; ++p) {
                janela[p] = janela[p + 1];
            }
            janela[cfg_.contexto - 1] = proximo;
        }

        return tok.decodificar(gerados);
    }

    // Salva configuracao e pesos em arquivo binario simples (.vibes).
    void salvar(const std::string& caminho) const {
        std::ofstream arq(caminho, std::ios::binary);
        if (!arq.is_open()) {
            throw std::runtime_error("Nao foi possivel abrir arquivo para salvar: " + caminho);
        }
        const char magico[8] = {'V', 'B', 'M', 'L', 'P', '0', '1', '\0'};
        arq.write(magico, 8);

        auto escrever_u64 = [&](std::uint64_t v) {
            arq.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };
        escrever_u64(cfg_.vocab);
        escrever_u64(cfg_.contexto);
        escrever_u64(cfg_.dim_emb);
        escrever_u64(cfg_.dim_oculta);

        auto escrever_vec = [&](const std::vector<float>& v) {
            arq.write(reinterpret_cast<const char*>(v.data()),
                      static_cast<std::streamsize>(v.size() * sizeof(float)));
        };
        escrever_vec(C_.dados());
        escrever_vec(W1_.dados());
        escrever_vec(b1_);
        escrever_vec(W2_.dados());
        escrever_vec(b2_);
    }

    // Carrega pesos de arquivo binario .vibes (exige mesma configuracao estrutural).
    void carregar(const std::string& caminho) {
        std::ifstream arq(caminho, std::ios::binary);
        if (!arq.is_open()) {
            throw std::runtime_error("Nao foi possivel abrir arquivo para carregar: " + caminho);
        }
        char magico[8]{};
        arq.read(magico, 8);
        if (std::string(magico, 7) != "VBMLP01") {
            throw std::runtime_error("Formato de checkpoint invalido em: " + caminho);
        }

        auto ler_u64 = [&]() {
            std::uint64_t v = 0;
            arq.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        };
        const auto v = ler_u64();
        const auto ctx = ler_u64();
        const auto emb = ler_u64();
        const auto hid = ler_u64();

        if (v != cfg_.vocab || ctx != cfg_.contexto || emb != cfg_.dim_emb ||
            hid != cfg_.dim_oculta) {
            throw std::runtime_error("Dimensoes do checkpoint nao correspondem ao modelo atual.");
        }

        auto ler_vec = [&](std::vector<float>& vec) {
            arq.read(reinterpret_cast<char*>(vec.data()),
                     static_cast<std::streamsize>(vec.size() * sizeof(float)));
        };
        ler_vec(C_.dados());
        ler_vec(W1_.dados());
        ler_vec(b1_);
        ler_vec(W2_.dados());
        ler_vec(b2_);

        if (!arq) {
            throw std::runtime_error("Arquivo de checkpoint corrompido ou truncado: " + caminho);
        }
    }

    // Acessores diretos usados em testes automatizados (ex.: gradient checking).
    const Matriz& C() const { return C_; }
    const Matriz& W1() const { return W1_; }
    const std::vector<float>& b1() const { return b1_; }
    const Matriz& W2() const { return W2_; }
    const std::vector<float>& b2() const { return b2_; }

private:
    ConfigMLP cfg_;

    // Pesos aprendiveis
    Matriz C_;
    Matriz W1_;
    std::vector<float> b1_;
    Matriz W2_;
    std::vector<float> b2_;

    // Velocidades do Momentum
    Matriz vel_C_;
    Matriz vel_W1_;
    std::vector<float> vel_b1_;
    Matriz vel_W2_;
    std::vector<float> vel_b2_;
};

// Verificacao numerica de gradiente (diferencas finitas centradas).
// Avalia a perda em precisao dupla (double) para evitar erro de cancelamento
// numerico tipico de (f(x+eps) - f(x-eps)) / (2*eps) em float32.
// Retorna o maior erro relativo observado entre gradiente analitico e numerico.
inline double avaliar_perda_double(const ModeloMLP& modelo,
                                   const std::vector<ExemploContexto>& exemplos,
                                   const std::vector<float>* ptr_modificado,
                                   std::size_t idx_modificado, double novo_valor) {
    const auto& cfg = modelo.config();
    const std::size_t E = cfg.dim_emb;
    const std::size_t D = cfg.dim_entrada();
    const std::size_t H = cfg.dim_oculta;
    const std::size_t V = cfg.vocab;

    auto ler = [&](const std::vector<float>& vec, std::size_t idx) -> double {
        if (&vec == ptr_modificado && idx == idx_modificado) {
            return novo_valor;
        }
        return static_cast<double>(vec[idx]);
    };

    const auto& C = modelo.C().dados();
    const auto& W1 = modelo.W1().dados();
    const auto& b1 = modelo.b1();
    const auto& W2 = modelo.W2().dados();
    const auto& b2 = modelo.b2();

    std::vector<double> x(D), h(H), logits(V);
    double perda_total = 0.0;

    for (const auto& ex : exemplos) {
        for (std::size_t p = 0; p < cfg.contexto; ++p) {
            const auto id = static_cast<std::size_t>(ex.contexto[p]);
            for (std::size_t e = 0; e < E; ++e) {
                x[p * E + e] = ler(C, id * E + e);
            }
        }
        for (std::size_t j = 0; j < H; ++j) {
            double soma = ler(b1, j);
            for (std::size_t d = 0; d < D; ++d) {
                soma += x[d] * ler(W1, d * H + j);
            }
            h[j] = std::tanh(soma);
        }
        double maior = -1e300;
        for (std::size_t v = 0; v < V; ++v) {
            double soma = ler(b2, v);
            for (std::size_t j = 0; j < H; ++j) {
                soma += h[j] * ler(W2, j * V + v);
            }
            logits[v] = soma;
            if (soma > maior) maior = soma;
        }
        double soma_exp = 0.0;
        for (std::size_t v = 0; v < V; ++v) {
            soma_exp += std::exp(logits[v] - maior);
        }
        const auto y = static_cast<std::size_t>(ex.alvo);
        const double log_prob = (logits[y] - maior) - std::log(soma_exp);
        perda_total += -log_prob;
    }

    return perda_total / static_cast<double>(exemplos.size());
}

inline float verificar_gradiente_numerico(ModeloMLP& modelo,
                                          const std::vector<ExemploContexto>& exemplos,
                                          std::size_t max_pesos_por_tensor = 16,
                                          double eps = 1e-5) {
    std::vector<std::size_t> indices(exemplos.size());
    for (std::size_t i = 0; i < exemplos.size(); ++i) {
        indices[i] = i;
    }

    GradientesMLP grad(modelo.config());
    modelo.calcular_perda_e_gradiente(exemplos, indices, grad);

    double maior_erro_relativo = 0.0;

    auto checar_vetor = [&](const std::vector<float>& pesos, const std::vector<float>& grads) {
        const std::size_t passo = std::max<std::size_t>(1, pesos.size() / max_pesos_por_tensor);
        for (std::size_t i = 0; i < pesos.size(); i += passo) {
            const double original = static_cast<double>(pesos[i]);
            const double perda_mais =
                avaliar_perda_double(modelo, exemplos, &pesos, i, original + eps);
            const double perda_menos =
                avaliar_perda_double(modelo, exemplos, &pesos, i, original - eps);

            const double grad_num = (perda_mais - perda_menos) / (2.0 * eps);
            const double grad_ana = static_cast<double>(grads[i]);
            const double diff = std::abs(grad_ana - grad_num);
            const double denom = std::max(std::abs(grad_ana) + std::abs(grad_num), 1e-4);
            const double erro_rel = diff / denom;

            if (erro_rel > maior_erro_relativo) {
                maior_erro_relativo = erro_rel;
            }
        }
    };

    checar_vetor(modelo.C().dados(), grad.dC.dados());
    checar_vetor(modelo.W1().dados(), grad.dW1.dados());
    checar_vetor(modelo.b1(), grad.db1);
    checar_vetor(modelo.W2().dados(), grad.dW2.dados());
    checar_vetor(modelo.b2(), grad.db2);

    return static_cast<float>(maior_erro_relativo);
}

}  // namespace vibes
