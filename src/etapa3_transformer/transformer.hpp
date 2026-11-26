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

// Hiperparametros estruturais do Mini-Transformer (arquitetura decoder-only estilo GPT).
struct ConfigTransformer {
    std::size_t vocab = 0;           // Tamanho do vocabulario (V)
    std::size_t contexto_max = 64;   // Comprimento maximo da sequencia (T_max)
    std::size_t dim_modelo = 64;     // Dimensao interna dos embeddings/estados (D)
    std::size_t num_cabecas = 4;     // Numero de cabecas de atencao (Nh)
    std::size_t num_camadas = 2;     // Numero de blocos Transformer (L)
    std::size_t dim_ff = 256;        // Dimensao interna da Feed-Forward (4 * D)

    std::size_t dim_cabeca() const { return dim_modelo / num_cabecas; }
};

// Uma sequencia de treinamento: entrada[0..T-1] -> alvo[0..T-1] (deslocado em 1 token).
struct SequenciaTreino {
    std::vector<int> entrada;
    std::vector<int> alvo;
};

// -----------------------------------------------------------------------------
// Funcoes auxiliares de ativacao (GELU aproximacao GPT-2) e LayerNorm
// -----------------------------------------------------------------------------
inline float gelu_f(float x) {
    constexpr float kRaiz2SobrePi = 0.7978845608028654f;
    const float a = kRaiz2SobrePi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(a));
}

inline float gelu_derivada_f(float x) {
    constexpr float kRaiz2SobrePi = 0.7978845608028654f;
    const float x2 = x * x;
    const float a = kRaiz2SobrePi * (x + 0.044715f * x * x2);
    const float t = std::tanh(a);
    const float da_dx = kRaiz2SobrePi * (1.0f + 3.0f * 0.044715f * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * da_dx;
}

inline double gelu_d(double x) {
    constexpr double kRaiz2SobrePi = 0.7978845608028654;
    const double a = kRaiz2SobrePi * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(a));
}

// -----------------------------------------------------------------------------
// Estruturas de Pesos e Gradientes de um Bloco Transformer
// -----------------------------------------------------------------------------
struct PesosBloco {
    // Pre-LayerNorm 1
    std::vector<float> ln1_g;
    std::vector<float> ln1_b;
    // Projecoes da Atencao Causal Multi-Head (D x D)
    Matriz Wq, Wk, Wv, Wo;
    std::vector<float> bq, bk, bv, bo;
    // Pre-LayerNorm 2
    std::vector<float> ln2_g;
    std::vector<float> ln2_b;
    // Feed-Forward Network: D -> D_ff -> D
    Matriz Wfc1, Wfc2;
    std::vector<float> bfc1, bfc2;

    explicit PesosBloco(const ConfigTransformer& cfg, float valor_gamma = 1.0f)
        : ln1_g(cfg.dim_modelo, valor_gamma),
          ln1_b(cfg.dim_modelo, 0.0f),
          Wq(cfg.dim_modelo, cfg.dim_modelo, 0.0f),
          Wk(cfg.dim_modelo, cfg.dim_modelo, 0.0f),
          Wv(cfg.dim_modelo, cfg.dim_modelo, 0.0f),
          Wo(cfg.dim_modelo, cfg.dim_modelo, 0.0f),
          bq(cfg.dim_modelo, 0.0f),
          bk(cfg.dim_modelo, 0.0f),
          bv(cfg.dim_modelo, 0.0f),
          bo(cfg.dim_modelo, 0.0f),
          ln2_g(cfg.dim_modelo, valor_gamma),
          ln2_b(cfg.dim_modelo, 0.0f),
          Wfc1(cfg.dim_modelo, cfg.dim_ff, 0.0f),
          Wfc2(cfg.dim_ff, cfg.dim_modelo, 0.0f),
          bfc1(cfg.dim_ff, 0.0f),
          bfc2(cfg.dim_modelo, 0.0f) {}
};

// Armazena todos os tensores (usado tanto para Pesos quanto para Gradientes / m / v).
struct ConjuntoTensoresGPT {
    Matriz w_tok;                  // [V x D]
    Matriz w_pos;                  // [T_max x D]
    std::vector<PesosBloco> blocos;
    std::vector<float> ln_f_g;     // [D]
    std::vector<float> ln_f_b;     // [D]
    Matriz w_head;                 // [D x V]
    std::vector<float> b_head;     // [V]

    explicit ConjuntoTensoresGPT(const ConfigTransformer& cfg, float valor_gamma = 1.0f)
        : w_tok(cfg.vocab, cfg.dim_modelo, 0.0f),
          w_pos(cfg.contexto_max, cfg.dim_modelo, 0.0f),
          ln_f_g(cfg.dim_modelo, valor_gamma),
          ln_f_b(cfg.dim_modelo, 0.0f),
          w_head(cfg.dim_modelo, cfg.vocab, 0.0f),
          b_head(cfg.vocab, 0.0f) {
        blocos.reserve(cfg.num_camadas);
        for (std::size_t l = 0; l < cfg.num_camadas; ++l) {
            blocos.emplace_back(cfg, valor_gamma);
        }
    }

    // Visita todos os vetores de parametros em ordem deterministica.
    // O segundo argumento (bool) indica se o tensor e matriz 2D (elegivel para Weight Decay).
    template <typename Fn>
    void visitar(Fn&& fn) {
        fn(w_tok.dados(), true);
        fn(w_pos.dados(), true);
        for (auto& b : blocos) {
            fn(b.ln1_g, false);
            fn(b.ln1_b, false);
            fn(b.Wq.dados(), true);
            fn(b.bq, false);
            fn(b.Wk.dados(), true);
            fn(b.bk, false);
            fn(b.Wv.dados(), true);
            fn(b.bv, false);
            fn(b.Wo.dados(), true);
            fn(b.bo, false);
            fn(b.ln2_g, false);
            fn(b.ln2_b, false);
            fn(b.Wfc1.dados(), true);
            fn(b.bfc1, false);
            fn(b.Wfc2.dados(), true);
            fn(b.bfc2, false);
        }
        fn(ln_f_g, false);
        fn(ln_f_b, false);
        fn(w_head.dados(), true);
        fn(b_head, false);
    }

    template <typename Fn>
    void visitar(Fn&& fn) const {
        fn(w_tok.dados(), true);
        fn(w_pos.dados(), true);
        for (const auto& b : blocos) {
            fn(b.ln1_g, false);
            fn(b.ln1_b, false);
            fn(b.Wq.dados(), true);
            fn(b.bq, false);
            fn(b.Wk.dados(), true);
            fn(b.bk, false);
            fn(b.Wv.dados(), true);
            fn(b.bv, false);
            fn(b.Wo.dados(), true);
            fn(b.bo, false);
            fn(b.ln2_g, false);
            fn(b.ln2_b, false);
            fn(b.Wfc1.dados(), true);
            fn(b.bfc1, false);
            fn(b.Wfc2.dados(), true);
            fn(b.bfc2, false);
        }
        fn(ln_f_g, false);
        fn(ln_f_b, false);
        fn(w_head.dados(), true);
        fn(b_head, false);
    }

    void zerar() {
        visitar([](std::vector<float>& vec, bool) { std::fill(vec.begin(), vec.end(), 0.0f); });
    }

    void somar_de(const ConjuntoTensoresGPT& outro) {
        std::vector<std::vector<float>*> dsts;
        visitar([&](std::vector<float>& v, bool) { dsts.push_back(&v); });
        std::size_t idx = 0;
        outro.visitar([&](const std::vector<float>& src, bool) {
            auto& dst = *dsts[idx++];
            for (std::size_t i = 0; i < dst.size(); ++i) {
                dst[i] += src[i];
            }
        });
    }

    std::size_t total_parametros() const {
        std::size_t total = 0;
        visitar([&](const std::vector<float>& v, bool) { total += v.size(); });
        return total;
    }
};

// Cache de ativacoes de um bloco Transformer (guardado no forward para uso no backward).
struct CacheBloco {
    Matriz x_in;
    Matriz ln1_xhat;
    std::vector<float> ln1_inv_std;
    Matriz ln1_out;
    Matriz Q, K, V;
    std::vector<float> attn_probs;  // [Nh * T * T]
    Matriz ctx;
    Matriz x_mid;
    Matriz ln2_xhat;
    std::vector<float> ln2_inv_std;
    Matriz ln2_out;
    Matriz ff_pre;
    Matriz ff_act;
};

struct CacheTransformer {
    std::vector<CacheBloco> blocos;
    Matriz x_final_in;
    Matriz lnf_xhat;
    std::vector<float> lnf_inv_std;
    Matriz lnf_out;
    Matriz logits;  // [T x V]
    Matriz probs;   // [T x V]
};

// -----------------------------------------------------------------------------
// Operacoes basicas de camadas (Linear e LayerNorm) - Forward e Backward
// -----------------------------------------------------------------------------
namespace detalhe_gpt {

inline void linear_forward(const Matriz& X, const Matriz& W, const std::vector<float>& b,
                           Matriz& Y) {
    const std::size_t T = X.linhas();
    const std::size_t Cin = X.colunas();
    const std::size_t Cout = W.colunas();
    Y = Matriz(T, Cout, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        auto y_t = Y.linha(t);
        std::copy(b.begin(), b.end(), y_t.begin());
        const auto x_t = X.linha(t);
        for (std::size_t i = 0; i < Cin; ++i) {
            const float xti = x_t[i];
            const auto w_i = W.linha(i);
            for (std::size_t o = 0; o < Cout; ++o) {
                y_t[o] += xti * w_i[o];
            }
        }
    }
}

inline void linear_backward(const Matriz& dY, const Matriz& X, const Matriz& W, Matriz& dX,
                            Matriz& dW, std::vector<float>& db) {
    const std::size_t T = X.linhas();
    const std::size_t Cin = X.colunas();
    const std::size_t Cout = W.colunas();
    dX = Matriz(T, Cin, 0.0f);

    for (std::size_t t = 0; t < T; ++t) {
        const auto dy_t = dY.linha(t);
        const auto x_t = X.linha(t);
        auto dx_t = dX.linha(t);

        for (std::size_t o = 0; o < Cout; ++o) {
            db[o] += dy_t[o];
        }
        for (std::size_t i = 0; i < Cin; ++i) {
            const float xti = x_t[i];
            const auto w_i = W.linha(i);
            auto dw_i = dW.linha(i);
            float soma_dx = 0.0f;
            for (std::size_t o = 0; o < Cout; ++o) {
                const float dyo = dy_t[o];
                dw_i[o] += xti * dyo;
                soma_dx += dyo * w_i[o];
            }
            dx_t[i] = soma_dx;
        }
    }
}

inline void layernorm_forward(const Matriz& X, const std::vector<float>& gamma,
                              const std::vector<float>& beta, Matriz& Y, Matriz& xhat,
                              std::vector<float>& inv_std) {
    const std::size_t T = X.linhas();
    const std::size_t D = X.colunas();
    constexpr float eps = 1e-5f;

    Y = Matriz(T, D, 0.0f);
    xhat = Matriz(T, D, 0.0f);
    inv_std.assign(T, 0.0f);

    for (std::size_t t = 0; t < T; ++t) {
        const auto x_t = X.linha(t);
        float media = 0.0f;
        for (std::size_t i = 0; i < D; ++i) media += x_t[i];
        media /= static_cast<float>(D);

        float var = 0.0f;
        for (std::size_t i = 0; i < D; ++i) {
            const float diff = x_t[i] - media;
            var += diff * diff;
        }
        var /= static_cast<float>(D);

        const float r = 1.0f / std::sqrt(var + eps);
        inv_std[t] = r;

        auto xh_t = xhat.linha(t);
        auto y_t = Y.linha(t);
        for (std::size_t i = 0; i < D; ++i) {
            const float xh = (x_t[i] - media) * r;
            xh_t[i] = xh;
            y_t[i] = gamma[i] * xh + beta[i];
        }
    }
}

inline void layernorm_backward(const Matriz& dY, const Matriz& xhat,
                               const std::vector<float>& inv_std, const std::vector<float>& gamma,
                               Matriz& dX, std::vector<float>& dgamma, std::vector<float>& dbeta) {
    const std::size_t T = dY.linhas();
    const std::size_t D = dY.colunas();
    const float Df = static_cast<float>(D);
    dX = Matriz(T, D, 0.0f);

    for (std::size_t t = 0; t < T; ++t) {
        const auto dy_t = dY.linha(t);
        const auto xh_t = xhat.linha(t);
        auto dx_t = dX.linha(t);
        const float r = inv_std[t];

        float soma_dxh = 0.0f;
        float soma_dxh_xh = 0.0f;
        for (std::size_t i = 0; i < D; ++i) {
            dbeta[i] += dy_t[i];
            dgamma[i] += dy_t[i] * xh_t[i];
            const float dxh = dy_t[i] * gamma[i];
            soma_dxh += dxh;
            soma_dxh_xh += dxh * xh_t[i];
        }

        const float fator = r / Df;
        for (std::size_t i = 0; i < D; ++i) {
            const float dxh = dy_t[i] * gamma[i];
            dx_t[i] = fator * (Df * dxh - soma_dxh - xh_t[i] * soma_dxh_xh);
        }
    }
}

}  // namespace detalhe_gpt

// -----------------------------------------------------------------------------
// Modelo Mini-Transformer Conversacional (Decoder-Only / GPT)
// -----------------------------------------------------------------------------
class ModeloTransformer {
public:
    ModeloTransformer(const ConfigTransformer& cfg, GeradorAleatorio& rng)
        : cfg_(cfg),
          pesos_(cfg, 1.0f),
          adam_m_(cfg, 0.0f),
          adam_v_(cfg, 0.0f) {
        if (cfg.dim_modelo == 0 || cfg.num_cabecas == 0 ||
            cfg.dim_modelo % cfg.num_cabecas != 0) {
            throw std::invalid_argument("dim_modelo deve ser multiplo positivo de num_cabecas.");
        }

        // Inicializacao estilo GPT-2:
        // Pesos 2D com N(0, 0.02), e projecoes residuais (Wo, Wfc2) escaladas por 1/sqrt(2*L).
        const float std_padrao = 0.02f;
        const float std_residual =
            std_padrao / std::sqrt(2.0f * static_cast<float>(cfg.num_camadas));

        auto init_normal = [&](std::vector<float>& vec, float desvio) {
            for (float& x : vec) x = rng.normal(0.0f, desvio);
        };

        init_normal(pesos_.w_tok.dados(), std_padrao);
        init_normal(pesos_.w_pos.dados(), std_padrao);
        for (auto& b : pesos_.blocos) {
            init_normal(b.Wq.dados(), std_padrao);
            init_normal(b.Wk.dados(), std_padrao);
            init_normal(b.Wv.dados(), std_padrao);
            init_normal(b.Wo.dados(), std_residual);
            init_normal(b.Wfc1.dados(), std_padrao);
            init_normal(b.Wfc2.dados(), std_residual);
        }
        init_normal(pesos_.w_head.dados(), std_padrao);
    }

    const ConfigTransformer& config() const { return cfg_; }
    const ConjuntoTensoresGPT& pesos() const { return pesos_; }
    std::size_t total_parametros() const { return pesos_.total_parametros(); }

    // Forward completo de uma sequencia de tokens [0..T-1].
    void forward_sequencia(const std::vector<int>& entrada, CacheTransformer& cache) const {
        const std::size_t T = entrada.size();
        assert(T > 0 && T <= cfg_.contexto_max);
        const std::size_t D = cfg_.dim_modelo;
        const std::size_t Nh = cfg_.num_cabecas;
        const std::size_t dh = cfg_.dim_cabeca();
        const float escala_attn = 1.0f / std::sqrt(static_cast<float>(dh));

        // 1. Embeddings de token + posicao
        Matriz x(T, D, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            const auto tok = static_cast<std::size_t>(entrada[t]);
            const auto e_tok = pesos_.w_tok.linha(tok);
            const auto e_pos = pesos_.w_pos.linha(t);
            auto x_t = x.linha(t);
            for (std::size_t d = 0; d < D; ++d) {
                x_t[d] = e_tok[d] + e_pos[d];
            }
        }

        // 2. L Blocos Transformer (Pre-LN -> Causal Multi-Head Attention -> Residual -> Pre-LN -> FFN -> Residual)
        cache.blocos.resize(cfg_.num_camadas);
        for (std::size_t l = 0; l < cfg_.num_camadas; ++l) {
            const auto& blk = pesos_.blocos[l];
            auto& cb = cache.blocos[l];
            cb.x_in = x;

            // 2a. Pre-LayerNorm 1
            detalhe_gpt::layernorm_forward(cb.x_in, blk.ln1_g, blk.ln1_b, cb.ln1_out, cb.ln1_xhat,
                                           cb.ln1_inv_std);

            // 2b. Projecoes Q, K, V
            detalhe_gpt::linear_forward(cb.ln1_out, blk.Wq, blk.bq, cb.Q);
            detalhe_gpt::linear_forward(cb.ln1_out, blk.Wk, blk.bk, cb.K);
            detalhe_gpt::linear_forward(cb.ln1_out, blk.Wv, blk.bv, cb.V);

            // 2c. Causal Multi-Head Self-Attention
            cb.attn_probs.assign(Nh * T * T, 0.0f);
            cb.ctx = Matriz(T, D, 0.0f);
            std::vector<float> scores(T, 0.0f);

            for (std::size_t h = 0; h < Nh; ++h) {
                const std::size_t oh = h * dh;
                for (std::size_t t = 0; t < T; ++t) {
                    // Mascara causal: posicao t so enxerga posicoes k <= t
                    float maior = -1e30f;
                    for (std::size_t k = 0; k <= t; ++k) {
                        float dot = 0.0f;
                        for (std::size_t d = 0; d < dh; ++d) {
                            dot += cb.Q(t, oh + d) * cb.K(k, oh + d);
                        }
                        scores[k] = dot * escala_attn;
                        if (scores[k] > maior) maior = scores[k];
                    }

                    // Softmax sobre k in [0..t]
                    float soma_exp = 0.0f;
                    float* probs_ht = &cb.attn_probs[(h * T + t) * T];
                    for (std::size_t k = 0; k <= t; ++k) {
                        probs_ht[k] = std::exp(scores[k] - maior);
                        soma_exp += probs_ht[k];
                    }
                    const float inv_soma = 1.0f / soma_exp;
                    for (std::size_t k = 0; k <= t; ++k) {
                        probs_ht[k] *= inv_soma;
                    }

                    // Soma ponderada de V
                    auto ctx_t = cb.ctx.linha(t);
                    for (std::size_t k = 0; k <= t; ++k) {
                        const float a = probs_ht[k];
                        const auto v_k = cb.V.linha(k);
                        for (std::size_t d = 0; d < dh; ++d) {
                            ctx_t[oh + d] += a * v_k[oh + d];
                        }
                    }
                }
            }

            // 2d. Projecao de saida + conexao residual
            Matriz attn_out;
            detalhe_gpt::linear_forward(cb.ctx, blk.Wo, blk.bo, attn_out);
            cb.x_mid = Matriz(T, D, 0.0f);
            for (std::size_t i = 0; i < cb.x_mid.total_elementos(); ++i) {
                cb.x_mid.dados()[i] = cb.x_in.dados()[i] + attn_out.dados()[i];
            }

            // 2e. Pre-LayerNorm 2 + Feed-Forward (GELU) + conexao residual
            detalhe_gpt::layernorm_forward(cb.x_mid, blk.ln2_g, blk.ln2_b, cb.ln2_out, cb.ln2_xhat,
                                           cb.ln2_inv_std);
            detalhe_gpt::linear_forward(cb.ln2_out, blk.Wfc1, blk.bfc1, cb.ff_pre);
            cb.ff_act = Matriz(T, cfg_.dim_ff, 0.0f);
            for (std::size_t i = 0; i < cb.ff_pre.total_elementos(); ++i) {
                cb.ff_act.dados()[i] = gelu_f(cb.ff_pre.dados()[i]);
            }
            Matriz ff_out;
            detalhe_gpt::linear_forward(cb.ff_act, blk.Wfc2, blk.bfc2, ff_out);

            x = Matriz(T, D, 0.0f);
            for (std::size_t i = 0; i < x.total_elementos(); ++i) {
                x.dados()[i] = cb.x_mid.dados()[i] + ff_out.dados()[i];
            }
        }

        // 3. LayerNorm final + LM Head + Softmax
        cache.x_final_in = x;
        detalhe_gpt::layernorm_forward(cache.x_final_in, pesos_.ln_f_g, pesos_.ln_f_b,
                                       cache.lnf_out, cache.lnf_xhat, cache.lnf_inv_std);
        detalhe_gpt::linear_forward(cache.lnf_out, pesos_.w_head, pesos_.b_head, cache.logits);

        cache.probs = Matriz(T, cfg_.vocab, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            softmax(cache.logits.linha(t), cache.probs.linha(t), 1.0f);
        }
    }

    // Forward + Cross-Entropy Loss + Backward acumulando gradientes em `grad`.
    float forward_backward_sequencia(const SequenciaTreino& seq, CacheTransformer& cache,
                                     ConjuntoTensoresGPT& grad) const {
        const std::size_t T = seq.entrada.size();
        const std::size_t D = cfg_.dim_modelo;
        const std::size_t Nh = cfg_.num_cabecas;
        const std::size_t dh = cfg_.dim_cabeca();
        const std::size_t V = cfg_.vocab;
        const float inv_T = 1.0f / static_cast<float>(T);
        const float escala_attn = 1.0f / std::sqrt(static_cast<float>(dh));

        forward_sequencia(seq.entrada, cache);

        // 1. Perda media na sequencia e dLogits
        float perda = 0.0f;
        Matriz dlogits(T, V, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            const auto y = static_cast<std::size_t>(seq.alvo[t]);
            const auto p_t = cache.probs.linha(t);
            auto dl_t = dlogits.linha(t);
            perda += -std::log(std::max(p_t[y], 1e-12f));
            for (std::size_t v = 0; v < V; ++v) {
                dl_t[v] = p_t[v] * inv_T;
            }
            dl_t[y] -= inv_T;
        }
        perda *= inv_T;

        // 2. Backward da LM Head e LayerNorm final
        Matriz d_lnf_out;
        detalhe_gpt::linear_backward(dlogits, cache.lnf_out, pesos_.w_head, d_lnf_out,
                                     grad.w_head, grad.b_head);
        Matriz dx;
        detalhe_gpt::layernorm_backward(d_lnf_out, cache.lnf_xhat, cache.lnf_inv_std,
                                        pesos_.ln_f_g, dx, grad.ln_f_g, grad.ln_f_b);

        // 3. Backward dos L Blocos Transformer (do ultimo para o primeiro)
        std::vector<float> dA(T, 0.0f);
        std::vector<float> dscores(T, 0.0f);

        for (std::size_t l = cfg_.num_camadas; l-- > 0;) {
            const auto& blk = pesos_.blocos[l];
            auto& gblk = grad.blocos[l];
            const auto& cb = cache.blocos[l];

            // 3a. Backward FFN + Residual 2
            Matriz d_ff_act;
            detalhe_gpt::linear_backward(dx, cb.ff_act, blk.Wfc2, d_ff_act, gblk.Wfc2, gblk.bfc2);
            Matriz d_ff_pre(T, cfg_.dim_ff, 0.0f);
            for (std::size_t i = 0; i < d_ff_pre.total_elementos(); ++i) {
                d_ff_pre.dados()[i] = d_ff_act.dados()[i] * gelu_derivada_f(cb.ff_pre.dados()[i]);
            }
            Matriz d_ln2_out;
            detalhe_gpt::linear_backward(d_ff_pre, cb.ln2_out, blk.Wfc1, d_ln2_out, gblk.Wfc1,
                                         gblk.bfc1);
            Matriz d_xmid_branch;
            detalhe_gpt::layernorm_backward(d_ln2_out, cb.ln2_xhat, cb.ln2_inv_std, blk.ln2_g,
                                            d_xmid_branch, gblk.ln2_g, gblk.ln2_b);

            // Soma o caminho residual com o caminho da FFN
            Matriz d_xmid(T, D, 0.0f);
            for (std::size_t i = 0; i < d_xmid.total_elementos(); ++i) {
                d_xmid.dados()[i] = dx.dados()[i] + d_xmid_branch.dados()[i];
            }

            // 3b. Backward Projecao de Saida da Atencao
            Matriz d_ctx;
            detalhe_gpt::linear_backward(d_xmid, cb.ctx, blk.Wo, d_ctx, gblk.Wo, gblk.bo);

            // 3c. Backward Causal Multi-Head Self-Attention
            Matriz dQ(T, D, 0.0f), dK(T, D, 0.0f), dV(T, D, 0.0f);
            for (std::size_t h = 0; h < Nh; ++h) {
                const std::size_t oh = h * dh;
                for (std::size_t t = 0; t < T; ++t) {
                    const float* probs_ht = &cb.attn_probs[(h * T + t) * T];
                    const auto dctx_t = d_ctx.linha(t);

                    // dA[k] e dV[k]
                    float dot_A_dA = 0.0f;
                    for (std::size_t k = 0; k <= t; ++k) {
                        const float a = probs_ht[k];
                        const auto v_k = cb.V.linha(k);
                        auto dv_k = dV.linha(k);
                        float soma_da = 0.0f;
                        for (std::size_t d = 0; d < dh; ++d) {
                            const float dc = dctx_t[oh + d];
                            soma_da += dc * v_k[oh + d];
                            dv_k[oh + d] += a * dc;
                        }
                        dA[k] = soma_da;
                        dot_A_dA += a * soma_da;
                    }

                    // Backward do Softmax na linha t -> dscores[k]
                    for (std::size_t k = 0; k <= t; ++k) {
                        dscores[k] = probs_ht[k] * (dA[k] - dot_A_dA);
                    }

                    // dQ[t] e dK[k]
                    auto dq_t = dQ.linha(t);
                    const auto q_t = cb.Q.linha(t);
                    for (std::size_t k = 0; k <= t; ++k) {
                        const float ds = dscores[k] * escala_attn;
                        const auto k_k = cb.K.linha(k);
                        auto dk_k = dK.linha(k);
                        for (std::size_t d = 0; d < dh; ++d) {
                            dq_t[oh + d] += ds * k_k[oh + d];
                            dk_k[oh + d] += ds * q_t[oh + d];
                        }
                    }
                }
            }

            // 3d. Backward das Projecoes Q, K, V -> soma em d_ln1_out
            Matriz d_ln1_q, d_ln1_k, d_ln1_v;
            detalhe_gpt::linear_backward(dQ, cb.ln1_out, blk.Wq, d_ln1_q, gblk.Wq, gblk.bq);
            detalhe_gpt::linear_backward(dK, cb.ln1_out, blk.Wk, d_ln1_k, gblk.Wk, gblk.bk);
            detalhe_gpt::linear_backward(dV, cb.ln1_out, blk.Wv, d_ln1_v, gblk.Wv, gblk.bv);

            Matriz d_ln1_out(T, D, 0.0f);
            for (std::size_t i = 0; i < d_ln1_out.total_elementos(); ++i) {
                d_ln1_out.dados()[i] =
                    d_ln1_q.dados()[i] + d_ln1_k.dados()[i] + d_ln1_v.dados()[i];
            }

            // 3e. Backward Pre-LayerNorm 1 + Residual 1
            Matriz d_xin_branch;
            detalhe_gpt::layernorm_backward(d_ln1_out, cb.ln1_xhat, cb.ln1_inv_std, blk.ln1_g,
                                            d_xin_branch, gblk.ln1_g, gblk.ln1_b);
            dx = Matriz(T, D, 0.0f);
            for (std::size_t i = 0; i < dx.total_elementos(); ++i) {
                dx.dados()[i] = d_xmid.dados()[i] + d_xin_branch.dados()[i];
            }
        }

        // 4. Backward dos Embeddings (scatter-add em w_tok e soma em w_pos)
        for (std::size_t t = 0; t < T; ++t) {
            const auto tok = static_cast<std::size_t>(seq.entrada[t]);
            const auto dx_t = dx.linha(t);
            auto dwt = grad.w_tok.linha(tok);
            auto dwp = grad.w_pos.linha(t);
            for (std::size_t d = 0; d < D; ++d) {
                dwt[d] += dx_t[d];
                dwp[d] += dx_t[d];
            }
        }

        return perda;
    }

    // Calcula perda media e gradientes medios sobre um mini-batch de sequencias (OpenMP).
    float calcular_perda_e_gradiente(const std::vector<SequenciaTreino>& batch,
                                     ConjuntoTensoresGPT& grad_saida) const {
        grad_saida.zerar();
        if (batch.empty()) return 0.0f;

        float perda_total = 0.0f;

#ifdef USAR_OPENMP
#pragma omp parallel reduction(+ : perda_total)
        {
            ConjuntoTensoresGPT grad_local(cfg_, 0.0f);
            CacheTransformer cache_local;

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < batch.size(); ++i) {
                perda_total += forward_backward_sequencia(batch[i], cache_local, grad_local);
            }

#pragma omp critical
            {
                grad_saida.somar_de(grad_local);
            }
        }
#else
        CacheTransformer cache;
        for (const auto& seq : batch) {
            perda_total += forward_backward_sequencia(seq, cache, grad_saida);
        }
#endif

        const float inv_b = 1.0f / static_cast<float>(batch.size());
        grad_saida.visitar([inv_b](std::vector<float>& g, bool) {
            for (float& x : g) x *= inv_b;
        });
        return perda_total * inv_b;
    }

    // Otimizador AdamW com Gradient Clipping global (L2 norm).
    void passo_adamw(ConjuntoTensoresGPT& grad, float taxa_aprendizado,
                     float weight_decay = 0.01f, float max_norm = 1.0f, float beta1 = 0.9f,
                     float beta2 = 0.999f, float eps = 1e-8f) {
        ++passo_otimizador_;

        // 1. Gradient Clipping global
        double soma_quadrados = 0.0;
        grad.visitar([&](const std::vector<float>& g, bool) {
            for (float x : g) soma_quadrados += static_cast<double>(x) * static_cast<double>(x);
        });
        const float norma = static_cast<float>(std::sqrt(soma_quadrados));
        const float escala_clip = (norma > max_norm && norma > 0.0f) ? (max_norm / norma) : 1.0f;

        // 2. Atualizacao AdamW com correcao de vies
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(passo_otimizador_));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(passo_otimizador_));

        std::vector<std::vector<float>*> ps, gs, ms, vs;
        std::vector<bool> wds;
        pesos_.visitar([&](std::vector<float>& v, bool wd) {
            ps.push_back(&v);
            wds.push_back(wd);
        });
        grad.visitar([&](std::vector<float>& v, bool) { gs.push_back(&v); });
        adam_m_.visitar([&](std::vector<float>& v, bool) { ms.push_back(&v); });
        adam_v_.visitar([&](std::vector<float>& v, bool) { vs.push_back(&v); });

        for (std::size_t t = 0; t < ps.size(); ++t) {
            auto& p = *ps[t];
            const auto& g = *gs[t];
            auto& m = *ms[t];
            auto& v = *vs[t];
            const bool aplicar_wd = wds[t];

            for (std::size_t i = 0; i < p.size(); ++i) {
                const float gi = g[i] * escala_clip;
                m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
                v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;

                const float m_hat = m[i] / bc1;
                const float v_hat = v[i] / bc2;

                if (aplicar_wd) {
                    p[i] -= taxa_aprendizado * weight_decay * p[i];
                }
                p[i] -= taxa_aprendizado * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }

    // Avalia perda media sobre um conjunto de sequencias sem alterar pesos.
    float avaliar(const std::vector<SequenciaTreino>& sequencias) const {
        if (sequencias.empty()) return 0.0f;
        double perda_total = 0.0;

#ifdef USAR_OPENMP
#pragma omp parallel reduction(+ : perda_total)
        {
            CacheTransformer cache;
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < sequencias.size(); ++i) {
                const auto& seq = sequencias[i];
                forward_sequencia(seq.entrada, cache);
                double p_seq = 0.0;
                for (std::size_t t = 0; t < seq.entrada.size(); ++t) {
                    const auto y = static_cast<std::size_t>(seq.alvo[t]);
                    p_seq += -std::log(std::max(cache.probs(t, y), 1e-12f));
                }
                perda_total += p_seq / static_cast<double>(seq.entrada.size());
            }
        }
#else
        CacheTransformer cache;
        for (const auto& seq : sequencias) {
            forward_sequencia(seq.entrada, cache);
            double p_seq = 0.0;
            for (std::size_t t = 0; t < seq.entrada.size(); ++t) {
                const auto y = static_cast<std::size_t>(seq.alvo[t]);
                p_seq += -std::log(std::max(cache.probs(t, y), 1e-12f));
            }
            perda_total += p_seq / static_cast<double>(seq.entrada.size());
        }
#endif
        return static_cast<float>(perda_total / static_cast<double>(sequencias.size()));
    }

    // Gera texto autoregressivamente usando Temperatura e Top-K.
    std::string gerar(const TokenizadorUTF8& tok, std::vector<int> contexto,
                      std::size_t max_tokens, float temperatura, std::size_t top_k,
                      GeradorAleatorio& rng, bool parar_em_quebra_linha = false) const {
        if (contexto.empty()) {
            const int id_inicio = tok.obter_id("\n") >= 0 ? tok.obter_id("\n") : 0;
            contexto.push_back(id_inicio);
        }

        CacheTransformer cache;
        std::vector<float> probs(cfg_.vocab, 0.0f);
        std::vector<int> gerados;
        gerados.reserve(max_tokens);
        const int id_quebra = tok.obter_id("\n");

        for (std::size_t passo = 0; passo < max_tokens; ++passo) {
            // Mantem apenas os ultimos contexto_max tokens
            if (contexto.size() > cfg_.contexto_max) {
                contexto.erase(contexto.begin(),
                               contexto.begin() +
                                   static_cast<std::ptrdiff_t>(contexto.size() - cfg_.contexto_max));
            }

            forward_sequencia(contexto, cache);
            const std::size_t ultimo = contexto.size() - 1;
            softmax(cache.logits.linha(ultimo), probs, temperatura);
            const int proximo = static_cast<int>(rng.amostrar_top_k(probs, top_k));

            if (parar_em_quebra_linha && proximo == id_quebra && !gerados.empty()) {
                break;
            }
            gerados.push_back(proximo);
            contexto.push_back(proximo);
        }

        return tok.decodificar(gerados);
    }

    // Salva configuracao e pesos em arquivo binario .vibes ("VBGPT01").
    void salvar(const std::string& caminho) const {
        std::ofstream arq(caminho, std::ios::binary);
        if (!arq.is_open()) {
            throw std::runtime_error("Nao foi possivel abrir arquivo para salvar: " + caminho);
        }
        const char magico[8] = {'V', 'B', 'G', 'P', 'T', '0', '1', '\0'};
        arq.write(magico, 8);

        auto escrever_u64 = [&](std::uint64_t v) {
            arq.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };
        escrever_u64(cfg_.vocab);
        escrever_u64(cfg_.contexto_max);
        escrever_u64(cfg_.dim_modelo);
        escrever_u64(cfg_.num_cabecas);
        escrever_u64(cfg_.num_camadas);
        escrever_u64(cfg_.dim_ff);

        pesos_.visitar([&](const std::vector<float>& vec, bool) {
            arq.write(reinterpret_cast<const char*>(vec.data()),
                      static_cast<std::streamsize>(vec.size() * sizeof(float)));
        });

        if (!arq) {
            throw std::runtime_error("Falha ao gravar checkpoint em: " + caminho);
        }
    }

    // Carrega pesos de arquivo binario .vibes ("VBGPT01").
    void carregar(const std::string& caminho) {
        std::ifstream arq(caminho, std::ios::binary);
        if (!arq.is_open()) {
            throw std::runtime_error("Nao foi possivel abrir arquivo para carregar: " + caminho);
        }
        char magico[8]{};
        arq.read(magico, 8);
        if (std::string(magico, 7) != "VBGPT01") {
            throw std::runtime_error("Formato de checkpoint GPT invalido em: " + caminho);
        }

        auto ler_u64 = [&]() {
            std::uint64_t v = 0;
            arq.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        };
        if (ler_u64() != cfg_.vocab || ler_u64() != cfg_.contexto_max ||
            ler_u64() != cfg_.dim_modelo || ler_u64() != cfg_.num_cabecas ||
            ler_u64() != cfg_.num_camadas || ler_u64() != cfg_.dim_ff) {
            throw std::runtime_error("Dimensoes do checkpoint nao correspondem ao Transformer.");
        }

        pesos_.visitar([&](std::vector<float>& vec, bool) {
            arq.read(reinterpret_cast<char*>(vec.data()),
                     static_cast<std::streamsize>(vec.size() * sizeof(float)));
        });

        if (!arq) {
            throw std::runtime_error("Arquivo de checkpoint corrompido ou truncado: " + caminho);
        }
    }

private:
    ConfigTransformer cfg_;
    ConjuntoTensoresGPT pesos_;
    ConjuntoTensoresGPT adam_m_;
    ConjuntoTensoresGPT adam_v_;
    std::size_t passo_otimizador_ = 0;
};

// -----------------------------------------------------------------------------
// Verificacao numerica de gradiente em precisao dupla (double) para o Transformer
// -----------------------------------------------------------------------------
inline double avaliar_perda_gpt_double(const ModeloTransformer& modelo,
                                       const std::vector<SequenciaTreino>& batch,
                                       const std::vector<float>* ptr_mod, std::size_t idx_mod,
                                       double novo_valor) {
    const auto& cfg = modelo.config();
    const auto& P = modelo.pesos();
    const std::size_t D = cfg.dim_modelo;
    const std::size_t Nh = cfg.num_cabecas;
    const std::size_t dh = cfg.dim_cabeca();
    const std::size_t V = cfg.vocab;
    const double escala = 1.0 / std::sqrt(static_cast<double>(dh));

    auto ler = [&](const std::vector<float>& vec, std::size_t idx) -> double {
        if (&vec == ptr_mod && idx == idx_mod) return novo_valor;
        return static_cast<double>(vec[idx]);
    };

    auto ln_d = [&](const std::vector<double>& X, std::size_t T, const std::vector<float>& g,
                    const std::vector<float>& b) {
        std::vector<double> Y(T * D, 0.0);
        for (std::size_t t = 0; t < T; ++t) {
            double media = 0.0;
            for (std::size_t i = 0; i < D; ++i) media += X[t * D + i];
            media /= static_cast<double>(D);
            double var = 0.0;
            for (std::size_t i = 0; i < D; ++i) {
                const double df = X[t * D + i] - media;
                var += df * df;
            }
            var /= static_cast<double>(D);
            const double r = 1.0 / std::sqrt(var + 1e-5);
            for (std::size_t i = 0; i < D; ++i) {
                Y[t * D + i] = ler(g, i) * ((X[t * D + i] - media) * r) + ler(b, i);
            }
        }
        return Y;
    };

    auto lin_d = [&](const std::vector<double>& X, std::size_t T, std::size_t Cin, std::size_t Cout,
                     const Matriz& W, const std::vector<float>& b) {
        std::vector<double> Y(T * Cout, 0.0);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t o = 0; o < Cout; ++o) {
                double soma = ler(b, o);
                for (std::size_t i = 0; i < Cin; ++i) {
                    soma += X[t * Cin + i] * ler(W.dados(), i * Cout + o);
                }
                Y[t * Cout + o] = soma;
            }
        }
        return Y;
    };

    double perda_batch = 0.0;
    for (const auto& seq : batch) {
        const std::size_t T = seq.entrada.size();
        std::vector<double> x(T * D, 0.0);
        for (std::size_t t = 0; t < T; ++t) {
            const auto tok = static_cast<std::size_t>(seq.entrada[t]);
            for (std::size_t d = 0; d < D; ++d) {
                x[t * D + d] = ler(P.w_tok.dados(), tok * D + d) + ler(P.w_pos.dados(), t * D + d);
            }
        }

        for (std::size_t l = 0; l < cfg.num_camadas; ++l) {
            const auto& blk = P.blocos[l];
            const auto ln1 = ln_d(x, T, blk.ln1_g, blk.ln1_b);
            const auto Q = lin_d(ln1, T, D, D, blk.Wq, blk.bq);
            const auto K = lin_d(ln1, T, D, D, blk.Wk, blk.bk);
            const auto Val = lin_d(ln1, T, D, D, blk.Wv, blk.bv);

            std::vector<double> ctx(T * D, 0.0);
            for (std::size_t h = 0; h < Nh; ++h) {
                const std::size_t oh = h * dh;
                for (std::size_t t = 0; t < T; ++t) {
                    std::vector<double> sc(t + 1, 0.0);
                    double maior = -1e300;
                    for (std::size_t k = 0; k <= t; ++k) {
                        double dot = 0.0;
                        for (std::size_t d = 0; d < dh; ++d) {
                            dot += Q[t * D + oh + d] * K[k * D + oh + d];
                        }
                        sc[k] = dot * escala;
                        if (sc[k] > maior) maior = sc[k];
                    }
                    double soma = 0.0;
                    for (std::size_t k = 0; k <= t; ++k) {
                        sc[k] = std::exp(sc[k] - maior);
                        soma += sc[k];
                    }
                    for (std::size_t k = 0; k <= t; ++k) {
                        const double a = sc[k] / soma;
                        for (std::size_t d = 0; d < dh; ++d) {
                            ctx[t * D + oh + d] += a * Val[k * D + oh + d];
                        }
                    }
                }
            }

            const auto attn_out = lin_d(ctx, T, D, D, blk.Wo, blk.bo);
            std::vector<double> x_mid(T * D, 0.0);
            for (std::size_t i = 0; i < T * D; ++i) x_mid[i] = x[i] + attn_out[i];

            const auto ln2 = ln_d(x_mid, T, blk.ln2_g, blk.ln2_b);
            auto ff1 = lin_d(ln2, T, D, cfg.dim_ff, blk.Wfc1, blk.bfc1);
            for (double& u : ff1) u = gelu_d(u);
            const auto ff2 = lin_d(ff1, T, cfg.dim_ff, D, blk.Wfc2, blk.bfc2);
            for (std::size_t i = 0; i < T * D; ++i) x[i] = x_mid[i] + ff2[i];
        }

        const auto lnf = ln_d(x, T, P.ln_f_g, P.ln_f_b);
        const auto logits = lin_d(lnf, T, D, V, P.w_head, P.b_head);

        double perda_seq = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            double maior = -1e300;
            for (std::size_t v = 0; v < V; ++v) {
                maior = std::max(maior, logits[t * V + v]);
            }
            double soma = 0.0;
            for (std::size_t v = 0; v < V; ++v) {
                soma += std::exp(logits[t * V + v] - maior);
            }
            const auto y = static_cast<std::size_t>(seq.alvo[t]);
            perda_seq += -((logits[t * V + y] - maior) - std::log(soma));
        }
        perda_batch += perda_seq / static_cast<double>(T);
    }

    return perda_batch / static_cast<double>(batch.size());
}

inline float verificar_gradiente_numerico_gpt(const ModeloTransformer& modelo,
                                              const std::vector<SequenciaTreino>& batch,
                                              std::size_t max_pesos_por_tensor = 4,
                                              double eps = 1e-5) {
    ConjuntoTensoresGPT grad(modelo.config(), 0.0f);
    modelo.calcular_perda_e_gradiente(batch, grad);

    std::vector<const std::vector<float>*> ps, gs;
    modelo.pesos().visitar([&](const std::vector<float>& v, bool) { ps.push_back(&v); });
    grad.visitar([&](const std::vector<float>& v, bool) { gs.push_back(&v); });

    double maior_erro = 0.0;
    for (std::size_t k = 0; k < ps.size(); ++k) {
        const auto& peso = *ps[k];
        const auto& gr = *gs[k];
        const std::size_t passo = std::max<std::size_t>(1, peso.size() / max_pesos_por_tensor);
        for (std::size_t i = 0; i < peso.size(); i += passo) {
            const double orig = static_cast<double>(peso[i]);
            const double l_mais = avaliar_perda_gpt_double(modelo, batch, &peso, i, orig + eps);
            const double l_menos = avaliar_perda_gpt_double(modelo, batch, &peso, i, orig - eps);
            const double g_num = (l_mais - l_menos) / (2.0 * eps);
            const double g_ana = static_cast<double>(gr[i]);
            const double denom = std::max(std::abs(g_ana) + std::abs(g_num), 1e-4);
            const double erro = std::abs(g_ana - g_num) / denom;
            if (erro > maior_erro) maior_erro = erro;
        }
    }

    return static_cast<float>(maior_erro);
}

}  // namespace vibes
